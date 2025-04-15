# Optimized script to add SWOT data to postgis database with multiprocessing
# imports
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
from tqdm import tqdm
import time
import argparse
import sys
import multiprocessing as mp
from functools import partial
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, Float, String, Text, inspect, select, text
from sqlalchemy.dialects.postgresql import insert
import hashlib
import psycopg2

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process SWOT water mask pixel cloud data')
    parser.add_argument('--start', type=int, default=0, help='Start file index')
    parser.add_argument('--end', type=int, default=None, help='End file index')
    parser.add_argument('--pause', type=float, default=0.5, help='Pause between files in seconds')
    parser.add_argument('--chunk-size', type=int, default=50000, help='Number of rows to process at once')
    parser.add_argument('--cores', type=int, default=6, help='Number of CPU cores to use')
    parser.add_argument('--verbose', action='store_true', help='Show detailed progress')
    return parser.parse_args()

def create_uid(lat, lon, height):
    """Create a unique hash from latitude, longitude, and height"""
    input_str = f"{lat:.6f}_{lon:.6f}_{height:.6f}"
    return hashlib.md5(input_str.encode()).hexdigest()

def process_file(filepath, args, config, file_metadata_table, pixel_cloud_table, db_uri):
    """Process a single SWOT file"""
    start_time = time.time()
    try:
        # Create a new engine for this process
        engine = create_engine(db_uri)
        
        # Open the NetCDF file
        with h5py.File(filepath, 'r') as f:
            # Extract metadata
            metadata_values = {
                'file_name': filepath.name,
                'cycle_number': int(f.attrs['cycle_number'][0]),
                'pass_number': int(f.attrs['pass_number'][0]),
                'tile_number': int(f.attrs['tile_number'][0]),
                'tile_name': f.attrs['tile_name'].decode(),
                'time_granule_start': f.attrs['time_granule_start'].decode(),
                'time_granule_end': f.attrs['time_granule_end'].decode(),
                'geospatial_lon_min': float(f.attrs['geospatial_lon_min'][0]),
                'geospatial_lon_max': float(f.attrs['geospatial_lon_max'][0]),
                'geospatial_lat_min': float(f.attrs['geospatial_lat_min'][0]),
                'geospatial_lat_max': float(f.attrs['geospatial_lat_max'][0])
            }
            
            # Insert metadata and get file_id
            with engine.connect() as conn:
                # Check if file already exists
                file_query = select(file_metadata_table.c.id).where(
                    file_metadata_table.c.file_name == filepath.name
                )
                existing_file = conn.execute(file_query).fetchone()
                
                if existing_file:
                    file_id = existing_file[0]
                    if args.verbose:
                        print(f"File {filepath.name} already in database with id {file_id}")
                else:
                    result = conn.execute(file_metadata_table.insert().values(metadata_values))
                    file_id = result.inserted_primary_key[0]
                    if args.verbose:
                        print(f"Inserted metadata with file_id: {file_id}")
            
            # Get pixel cloud data
            pixel_cloud = f['pixel_cloud']
            
            # Filter classification data first to get only classes 3 or 4
            classification = pixel_cloud['classification'][:]
            mask = np.isin(classification, [3, 4])
            filtered_indices = np.where(mask)[0]
            
            if len(filtered_indices) == 0:
                print(f"No valid data (classification 3 or 4) found in {filepath.name}")
                return
            
            total_rows = len(filtered_indices)
            if args.verbose:
                print(f"Found {total_rows} rows with classification 3 or 4 (out of {len(classification)} total)")
            
            # Process pixel data in chunks
            chunk_size = args.chunk_size
            num_chunks = (total_rows + chunk_size - 1) // chunk_size
            
            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min((chunk_idx + 1) * chunk_size, total_rows)
                
                # Get indices for this chunk
                chunk_indices = filtered_indices[start_idx:end_idx]
                
                # Extract data only for points with classification 3 or 4
                latitude = pixel_cloud['latitude'][chunk_indices]
                longitude = pixel_cloud['longitude'][chunk_indices]
                height = pixel_cloud['height'][chunk_indices]
                water_frac = pixel_cloud['water_frac'][chunk_indices]
                classification_chunk = classification[chunk_indices]
                sig0 = pixel_cloud['sig0'][chunk_indices]
                pixel_area = pixel_cloud['pixel_area'][chunk_indices]
                
                # Create unique IDs for each point
                point_uids = [create_uid(lat, lon, h) for lat, lon, h in zip(latitude, longitude, height)]
                
                # Prepare data for insertion
                pixel_data = pd.DataFrame({
                    'file_id': file_id,
                    'point_uid': point_uids,
                    'latitude': latitude,
                    'longitude': longitude,
                    'height': height,
                    'water_frac': water_frac,
                    'classification': classification_chunk,
                    'sig0': sig0,
                    'pixel_area': pixel_area
                })
                
                # Insert or update pixel cloud data using the point_uid as a key
                with engine.connect() as conn:
                    # Check which records already exist
                    existing_uids = pd.read_sql(
                        select(pixel_cloud_table.c.point_uid)
                        .where(pixel_cloud_table.c.point_uid.in_(point_uids)),
                        conn
                    )['point_uid'].tolist()
                    
                    # Filter out existing records and only insert new ones
                    if existing_uids:
                        pixel_data = pixel_data[~pixel_data['point_uid'].isin(existing_uids)]
                    
                    if not pixel_data.empty:
                        # Bulk insert new data
                        pixel_data.to_sql('pixel_cloud', conn, if_exists='append', index=False)
                        if args.verbose:
                            print(f"  Inserted {len(pixel_data)} new rows into pixel_cloud table")
                    elif args.verbose:
                        print(f"  All {len(point_uids)} points already in database, skipping")
        
        elapsed_time = time.time() - start_time
        return f"Processed {filepath.name} in {elapsed_time:.2f} seconds"
    
    except Exception as e:
        return f"ERROR processing file {filepath.name}: {str(e)}"

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Load configuration
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        print(f"Successfully loaded configuration from config.json")
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)
    
    # Get a list of all NetCDF files in the pixel cloud directory
    file_list = sorted(list(Path(config['water_mask_pixel_cloud_dir']).glob('*.nc')))
    total_files = len(file_list)
    print(f"Found {total_files} files in {config['water_mask_pixel_cloud_dir']}")
    
    # Slice files based on arguments
    end_idx = args.end if args.end is not None else total_files
    file_list = file_list[args.start:end_idx]
    print(f"Processing files {args.start} to {end_idx-1} ({len(file_list)} files)")
    
    # Database connection string
    db_uri = 'postgresql://ben:1234@localhost:5432/swotdb'
    
    # Database connection
    try:
        engine = create_engine(db_uri)
        metadata = MetaData()
        print("Successfully connected to database")
    except Exception as e:
        print(f"Error connecting to database: {e}")
        sys.exit(1)
    
    # Define database tables
    file_metadata_table = Table(
        'file_metadata', metadata,
        Column('id', Integer, primary_key=True),
        Column('file_name', String, nullable=False),
        Column('cycle_number', Integer),
        Column('pass_number', Integer),
        Column('tile_number', Integer),
        Column('tile_name', String),
        Column('time_granule_start', String),
        Column('time_granule_end', String),
        Column('geospatial_lon_min', Float),
        Column('geospatial_lon_max', Float),
        Column('geospatial_lat_min', Float),
        Column('geospatial_lat_max', Float)
    )
    
    pixel_cloud_table = Table(
        'pixel_cloud', metadata,
        Column('id', Integer, primary_key=True),
        Column('file_id', Integer),
        Column('point_uid', String(32)),  # MD5 hash of lat+lon+height
        Column('latitude', Float),
        Column('longitude', Float),
        Column('height', Float),
        Column('water_frac', Float),
        Column('classification', Integer),
        Column('sig0', Float),
        Column('pixel_area', Float)
    )
    
    # Create tables if they don't exist
    try:
        # Check if tables exist first
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        
        if 'file_metadata' not in tables:
            file_metadata_table.create(engine)
            print("Created file_metadata table")
        
        if 'pixel_cloud' not in tables:
            pixel_cloud_table.create(engine)
            print("Created pixel_cloud table")
        
        # Check if our point_uid column exists, and add it if not
        if 'pixel_cloud' in tables:
            existing_columns = {col['name'] for col in inspector.get_columns('pixel_cloud')}
            if 'point_uid' not in existing_columns:
                # Use direct psycopg2 connection for DDL operations
                with psycopg2.connect(
                    dbname="swotdb",
                    user="ben",
                    password="1234",
                    host="localhost",
                    port="5432"
                ) as conn:
                    conn.autocommit = True
                    with conn.cursor() as cur:
                        # Add point_uid column
                        cur.execute("ALTER TABLE pixel_cloud ADD COLUMN point_uid VARCHAR(32)")
                        print("Added point_uid column to existing pixel_cloud table")
                        
                        # Populate point_uid for existing data
                        cur.execute("""
                            UPDATE pixel_cloud 
                            SET point_uid = md5(
                                CAST(ROUND(latitude::numeric, 6) AS TEXT) || '_' || 
                                CAST(ROUND(longitude::numeric, 6) AS TEXT) || '_' || 
                                CAST(ROUND(height::numeric, 6) AS TEXT)
                            )
                            WHERE point_uid IS NULL
                        """)
                        print("Populated point_uid for existing data")
                        
                        # Add unique index
                        cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_pixel_cloud_uid ON pixel_cloud(point_uid)")
                        print("Added unique index on point_uid")
        
        print("Database structure verified")
    except Exception as e:
        print(f"Error setting up database tables: {str(e)}")
        sys.exit(1)
    
    # Create a pool of workers (using the specified number of cores)
    pool = mp.Pool(processes=min(args.cores, len(file_list)))
    
    # Process files in parallel
    process_func = partial(
        process_file, 
        args=args, 
        config=config, 
        file_metadata_table=file_metadata_table, 
        pixel_cloud_table=pixel_cloud_table, 
        db_uri=db_uri
    )
    
    print(f"Starting parallel processing with {min(args.cores, len(file_list))} workers")
    
    # Process files and collect results
    results = []
    for result in tqdm(pool.imap_unordered(process_func, file_list), total=len(file_list)):
        if result:
            print(result)
        time.sleep(args.pause)  # Short pause to prevent database contention
    
    # Clean up
    pool.close()
    pool.join()
    
    print(f"\nProcessing complete!")

if __name__ == "__main__":
    main()