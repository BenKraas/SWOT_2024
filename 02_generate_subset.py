import os
import yaml
import numpy as np
import pandas as pd
import geopandas as gpd
import netCDF4 as nc
import h5py
import xarray as xr
import re

from tqdm import tqdm
from shapely.geometry import Point
import logging
import json
from pathlib import Path
from datetime import datetime


def load_dataset_description(yaml_file):
    """Load and parse the YAML configuration file."""
    with open(yaml_file, 'r') as f:
        description = yaml.safe_load(f)
    logger.debug(f"Loaded dataset description from {yaml_file}")
    return description

def check_variable_type(var_data, expected_type):
    """
    Check if variable data matches the expected type.
    Supported types: 'int', 'float', 'string', 'float64', 'float32', 'datetime64[ns]'.
    """
    if expected_type == 'int':
        return np.issubdtype(var_data.dtype, np.integer)
    elif expected_type == 'float':
        return np.issubdtype(var_data.dtype, np.floating)
    elif expected_type == 'string':
        return np.issubdtype(var_data.dtype, np.character)
    elif expected_type == 'float64':
        return var_data.dtype == np.float64
    elif expected_type == 'float32':
        return var_data.dtype == np.float32
    elif expected_type == 'datetime64[ns]':
        return np.issubdtype(var_data.dtype, np.datetime64)
    elif expected_type == 'np.ndarray':
        return isinstance(var_data, np.ndarray)
    else:
        logger.warning(f"Unknown type check for {expected_type}. Check supported types in check_variable_type() function.")
        return True

def filter_by_constraints(data, var_name, constraints):
    """Apply constraints to filter the data."""
    if data is None or len(data) == 0:
        return data
    
    # Check min constraint
    if 'min' in constraints:
        min_val = constraints['min']
        data = data[data[var_name] >= min_val]
    
    # Check max constraint
    if 'max' in constraints:
        max_val = constraints['max']
        data = data[data[var_name] <= max_val]
    
    # Check include constraint
    if 'include' in constraints:
        include_values = constraints['include']
        data = data[data[var_name].isin(include_values)]
    
    return data

def process_netcdf_file_nc(file_path, config):
    """Process a single netCDF file according to configuration."""
    logger.debug(f"Processing file: {file_path} using netCDF4")
    
    try:
        # Open the netCDF file
        dataset = nc.Dataset(file_path, 'r')
        
        # Extract variables from the specified groups
        all_data = {}
        
        # Focus on Pixel Cloud Keys group as specified in the YAML
        pixel_cloud_config = config.get('Pixel Cloud Keys', {})
        
        if 'pixel_cloud' not in dataset.groups:
            logger.error(f"Required group 'pixel_cloud' not found in {file_path}")
            return None
        
        pixel_cloud_group = dataset.groups['pixel_cloud']
        
        # Extract each variable defined in the config
        for var_name, var_config in pixel_cloud_config.items():
            if var_name not in pixel_cloud_group.variables:
                logger.warning(f"Variable {var_name} not found in pixel_cloud group")
                continue
            
            # Get the variable data
            var_data = pixel_cloud_group.variables[var_name][:]
            
            # Check the variable type
            expected_type = var_config.get('type')
            if not check_variable_type(var_data, expected_type):
                error_msg = f"Variable {var_name} has incorrect type. Expected {expected_type}, got {var_data.dtype}"
                logger.error(error_msg)
                raise TypeError(error_msg)
            
            # Store the data
            all_data[var_name] = var_data
        
        # Create dataframe from the collected data
        if not all_data:
            logger.warning(f"No valid data extracted from {file_path}")
            return None
        
        df = pd.DataFrame(all_data)
        
        # Apply filtering based on constraints
        for var_name, var_config in pixel_cloud_config.items():
            if var_name in df.columns:
                constraints = {k: v for k, v in var_config.items() if k not in ['type', 'description']}
                if constraints:
                    df = filter_by_constraints(df, var_name, constraints)
        
        # Convert to GeoDataFrame
        if 'latitude' in df.columns and 'longitude' in df.columns:
            geometry = [Point(lon, lat) for lon, lat in zip(df['longitude'], df['latitude'])]
            gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
            return gdf
        else:
            logger.warning(f"Cannot create GeoDataFrame: missing latitude/longitude columns in {file_path}")
            return None
    
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        return None
    finally:
        if 'dataset' in locals():
            dataset.close()

def process_netcdf_file_xr(file_path, dataset_description, clip_shp=None, clip_id_name=None):
    """Process a single netCDF file according to configuration using xarray."""
    logger.debug(f"Processing file: {file_path} using xarray")
    
    try:
        # Open the netCDF file with xarray
        with h5py.File(file_path, 'r') as xr_dataset:
            # get all attribute values as a dictionary
            dataset_info = dict(xr_dataset.attrs)

            # get the pixel cloud group
            pixel_cloud_group = xr.open_dataset(file_path, group='pixel_cloud')
            # unload the dataset
            del xr_dataset
            
            # Focus on Pixel Cloud Keys group as specified in the YAML
            pixel_cloud_config = dataset_description.get('pixel_cloud_keys', {})
            
            if pixel_cloud_group is None:
                logger.error(f"Required group 'pixel_cloud' not found in {file_path}")
                return None
            
            # Create a subset with only the variables we need
            variables_to_extract = list(pixel_cloud_config.keys())
            
            # Check if all required variables exist
            missing_vars = [var for var in variables_to_extract if var not in pixel_cloud_group.variables]
            if missing_vars:
                for var in missing_vars:
                    logger.warning(f"Variable {var} not found in pixel_cloud group")
                
                # Update the list of variables to extract
                variables_to_extract = [var for var in variables_to_extract if var not in missing_vars]
                
                if not variables_to_extract:
                    logger.warning(f"No valid variables to extract from {file_path}")
                    return None
            
            # Extract the subset of data
            data_subset = pixel_cloud_group[variables_to_extract]

            # unload pixel_cloud_group
            del pixel_cloud_group
           
            # Ensure the dataset only contains the specified variables
            data_subset = data_subset[variables_to_extract]

            # Validate variable types
            invalid_columns = []
            for var_name, var_config in pixel_cloud_config.items():
                if var_name in data_subset.variables:
                    expected_type = var_config.get('type')
                    if not check_variable_type(data_subset[var_name].values, expected_type):
                        logger.warning(f"Variable {var_name} has incorrect type. Expected {expected_type}, got {data_subset[var_name].dtype}")
                        invalid_columns.append(var_name)
            
            # Drop invalid columns
            if invalid_columns:
                logger.info(f"Dropping invalid columns: {invalid_columns}")
                data_subset = data_subset.drop_vars(invalid_columns)
            
            new_dataset = pd.DataFrame()
            for var_name in data_subset.variables:
                try:
                    # Convert to DataFrame
                    var_data = data_subset[var_name].values
                    print(f"Var name: {var_name} shape: {var_data.shape}")
                    new_dataset[var_name] = var_data
                except Exception as e:
                    logger.error(f"Error converting variable {var_name} to DataFrame: {str(e)}. It will be skipped.")
                    continue

            # Convert to dataframe
            logger.debug("Converting to DataFrame.")
            df = new_dataset.reset_index(drop=True)
            
            # Apply filtering based on constraints
            for var_name, var_config in pixel_cloud_config.items():
                if var_name in df.columns:
                    constraints = {k: v for k, v in var_config.items() if k not in ['type', 'description']}
                    if constraints:
                        df = filter_by_constraints(df, var_name, constraints)
            

            # Convert to GeoDataFrame if latitude and longitude are available
            if 'latitude' in df.columns and 'longitude' in df.columns:
                # Use GeoPandas' built-in functionality for creating geometries
                geometry = gpd.points_from_xy(df['longitude'], df['latitude'])
                gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
                logger.debug(f"Created GeoDataFrame with {len(gdf)} rows.")
            else:
                logger.warning(f"Cannot create GeoDataFrame: missing latitude/longitude columns in {file_path}")
                return None
            
            # If clip_shp is provided, clip the GeoDataFrame to every geometry in the shapefile and assign the clip_id_name to the clipped geometries
            if clip_shp is not None:
                logger.debug(f"Clipping GeoDataFrame to shapefile: {clip_shp}")
                clip_gdf = gpd.read_file(clip_shp)
                
                # Check if the clip shapefile is empty
                if clip_gdf.empty:
                    logger.warning(f"Clip shapefile {clip_shp} is empty.")
                    return None
                
                if not clip_id_name:
                    logger.error("clip_id_name is not specified. It should specify the column name in the clip shapefile identifying the respective clipping geometries.")
                    return None
                
                # Ensure the clip shapefile contains the clip_id_name column
                if clip_id_name not in clip_gdf.columns:
                    logger.error(f"Column '{clip_id_name}' not found in the clip shapefile.")
                    logger.error(f"Available columns: {clip_gdf.columns.tolist()}")
                    return None
                
                # Perform the spatial join and retain the clip_id_name column
                _num_rows_before = len(gdf)
                gdf = gpd.sjoin(gdf, clip_gdf[[clip_id_name, 'geometry']], how='inner', predicate='intersects')
                _num_rows_after = len(gdf)
                logger.debug(f"Clipped GeoDataFrame to {clip_shp} using {clip_id_name}. Rows before: {_num_rows_before}, after: {_num_rows_after}")
                
                # Rename the clip_id_name column to avoid conflicts
                gdf.rename(columns={clip_id_name: f"{clip_id_name}_clip"}, inplace=True)

                # Check if the resulting GeoDataFrame is empty after clipping
                if gdf.empty:
                    logger.debug(f"GeoDataFrame is empty after clipping with {clip_shp}")
                    return None
            
            # populate the dataframe with the file_parameters specified in the yaml file post filtering
            required_attributes = dataset_description.get('file_parameters', [])
            for attr in required_attributes:
                # check if the attribute exists in the dataset
                if attr in dataset_info:
                    # check if the attribute complies with the expected type
                    attr_type = required_attributes.get(attr, {}).get('type')
                    attr_content = dataset_info[attr]
                    attr_cast = required_attributes.get(attr, {}).get('cast', None)
                    
                    if attr_type:
                        if not check_variable_type(attr_content, attr_type):
                            logger.warning(f"Attribute {attr} has incorrect type. Expected {attr_type}, got {type(attr_content)}")
                            continue
                    
                    # check if the attribute needs to be casted and attempt if so
                    if attr_cast:
                        _cast_type_dict = {'int': int,'float': float,'str': str,'bool': bool,'list': list,'np.ndarray': np.ndarray}
                        # handle ndim > 0 to scalar
                        if isinstance(attr_content, np.ndarray) and attr_content.ndim > 0:
                            attr_content = attr_content.flatten()
                            if len(attr_content) == 1:
                                attr_content = attr_content[0]
                        try:
                            attr_content = _cast_type_dict[attr_cast](attr_content)
                        except Exception as e:
                            logger.warning(f"Failed to cast attribute {attr} to {attr_cast}: {str(e)}")
                            continue
                    
                    # add the attribute to the GeoDataFrame depending on the type
                    if isinstance(attr_content, (str, int, float)):
                        gdf[attr] = attr_content
                    elif isinstance(attr_content, list):
                        gdf[attr] = [attr_content] * len(gdf)
                    elif isinstance(attr_content, np.ndarray):
                        gdf[attr] = [attr_content.tolist()] * len(gdf)
                    elif isinstance(attr_content, np.bytes_):
                        gdf[attr] = attr_content.decode('utf-8')
                    else:
                        logger.warning(f"Unsupported attribute type for {attr}: {type(attr_content)}. Expected str, int, float, list, np.ndarray, or np.bytes_.")
                        continue
                    logger.debug(f"Added attribute {attr} with {attr_content} to GeoDataFrame")
                else:
                    logger.warning(f"Attribute {attr} not found in dataset info")

            # Check if the resulting GeoDataFrame is empty
            if gdf.empty:
                logger.debug(f"GeoDataFrame is empty after processing {file_path}")
                return None
            
            # return the GeoDataFrame
            logger.debug(f"Processed {file_path} successfully.")
            return gdf
    
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        return None

def process_netcdf_files(file_paths, config, output_file=None, clip_shp=None, clip_id_name=None):
    """Process multiple netCDF files and combine the results."""
    all_gdfs = []
    
    for file_path in tqdm(file_paths, desc="Processing netCDF files"):
        gdf = process_netcdf_file_xr(file_path, config, clip_shp, clip_id_name)
        if gdf is not None:
            all_gdfs.append(gdf)
        else:
            logger.debug(f"Skipping empty or invalid GeoDataFrame from {file_path}")
            continue
    
    if not all_gdfs:
        logger.warning("No valid data extracted from any file")
        return None
    
    # Combine all GeoDataFrames
    combined_gdf = pd.concat(all_gdfs, ignore_index=True)
    
    # Save to file if specified
    if output_file:
        file_ext = os.path.splitext(output_file)[1].lower()
        if file_ext == '.shp':
            combined_gdf.to_file(output_file)
        elif file_ext == '.csv':
            combined_gdf.to_csv(output_file, index=False)
        elif file_ext == '.geojson':
            combined_gdf.to_file(output_file, driver='GeoJSON')
        else:
            combined_gdf.to_pickle(output_file)
        logger.info(f"Saved combined GeoDataFrame to {output_file}")
    
    # logger print all columns and their types
    logger.debug("Combined GeoDataFrame columns and types:")
    for col in combined_gdf.columns:
        logger.debug(f"{col}: {combined_gdf[col].dtype}")
    
    return combined_gdf

def load_configuration():
    with open('config.json', 'r') as f:
        config = json.load(f)
    return config

def initialize_logging(print_level=logging.INFO, file_level=logging.DEBUG):
    """
    Initialize logging configuration for both console and file outputs.

    Args:
        print_level (int): Logging level for console output (e.g., logging.INFO).
        file_level (int): Logging level for file output (e.g., logging.DEBUG).

    Returns:
        logging.Logger: Configured logger instance.
    """
    global logger

    # Create a logger instance
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)  # Set the base logging level to DEBUG

    # Define a timestamp for the log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    log_filename = log_dir / f'process_netcdf_{timestamp}.log'

    # Create a file handler for logging to a file
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(file_level)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # Add the file handler only if it hasn't been added already
    if not any(isinstance(handler, logging.FileHandler) and handler.baseFilename == file_handler.baseFilename for handler in logger.handlers):
        logger.addHandler(file_handler)

    # Create a console handler for logging to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(print_level)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)

    # Add the console handler
    logger.addHandler(console_handler)

    return logger

def main():
    """Main function to run the script."""

    # initialization
    config = load_configuration()
    output_dir = Path(config['data_dir']) / 'output'
    output_dir.mkdir(exist_ok=True)
    logger = initialize_logging(logging.INFO, logging.DEBUG)
    # Load configuration
    column_definition = load_dataset_description("define_columns.yaml")


    ##############
    # Script start
    ##############

    logger.info("Starting the netCDF processing script.")
    
    # Make a list of all netCDF files
    netcdf_files = list(Path(config['pixel_cloud_dataset_folder']).glob('*.nc'))
    netcdf_files = netcdf_files[:2] # Limit to first 2 files for testing

    # Setup output file name with timestamp
    output_file = output_dir / f"combined_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.geojson"

    # Process files
    result = process_netcdf_files(netcdf_files, column_definition, output_file, config["clip_shp_path"], config["clip_id_name"])
    
    if result is not None:
        logger.info(f"Successfully processed {len(netcdf_files)} files. Result contains {len(result)} rows.")
    else:
        logger.error("Failed to process files.")

    logger.info("Script finished.")

if __name__ == "__main__":
    main()