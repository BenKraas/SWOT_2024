# SWOT_2024
Custom workflow for SWOT analysis.

# Requirements
- Python 3.12.2
- An environment with the required packages installed (see environment.yml)

Conda install with `conda env create -f environment_anc.yml`
Mamba install with `mamba env create -f environment_mmb.yml`

# Goal
The goal of this project is to create a custom workflow for SWOT analysis. 

# Workflow
1. **Data Preparation**: 
   - Download and prepare the SWOT data.
   - This is done using the `01 Download_SWOT_data.ipynb` script.
2. **Data Preprocessing**: 
   - Preprocess the SWOT data 
   - This is done using the `02_generate_subset.py` script.
   - Call the script with the following command:
	 ```bash
	 python 02_generate_subset.py
	 ```
   - Columns to keep, types, casts etc. must be defined in the `define_columns.yaml` file.
   - Automatic preprocessing includes:
     - Performantly loading data from `data/source/SWOT_L2_HR_PIXC_2.0_20230101_20241231`.
	 - Data validation, type checks, type casts
	 - Range limiting
	 - Conversion into a common geodata format (`geojson`, `shapefile`, `csv`, `pickle`)
3. **Data Analysis**: 
   - Perform data analysis on the preprocessed data.
   - This is done using the `03 Boxplots Focus Regions.ipynb` script.
   
   - This script generates boxplots for the focus regions

# Interesting events:
Damm breakage (date pending)
Volumenberechnung?