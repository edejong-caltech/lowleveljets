# lowleveljets
Repository of code and selected data resources for JAS publication.

## Downloading Data
Data used in these analysis comes from three sources and will need to be downloaded by the user. Data file paths in some notebooks may need to be updated accordingly.
1. NYSERDA floating lidar buoy data, hourly and 10min resolution at the North E05 and South E06 buoys: https://oswbuoysny.resourcepanorama.dnv.com/
2. Doppler lidar data from the ARM SGP site C1 from 2018-06-08 through 2018-06-20: https://www.arm.gov/data/data-sources
3. WRF model run for the mid-Atlantic region, hourly data from 2020: https://data.openei.org/submissions/4500

## Running the analysis notebooks
You may use the included `environment.yml` file to create a python environment and run the notebooks included here.

`conda env create -f environment.yml`

Be sure that `ipykernel` is installed and set up within your environment.
