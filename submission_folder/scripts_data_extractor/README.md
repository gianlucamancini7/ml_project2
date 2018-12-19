# scripts_data_extractor README

Machine learning Project 2 – CS-433 – EPFL

Fall 2018

Authors: Gianluca Mancini, Tullio Nutta and Tianchu Zhang

## Scripts info
- **data_extractor.ipynb:** converts all the raw input file into averaged results. Takes zipped folders into from ``/data/ zip_files`` and saved the results in `` /data/average_data``. The time over which the average is taken can be changed; defaults is 5 minutes. Target directory should be empty before running the script as the result file will be not overwritten.

- **regression_mat_builder.ipynb:** converts all the averaged file into a single matrix saved as `` regression_mat_year``. Takes csv files from ``/data/average_data`` and saved a csv in `` /data``. No pre-existing file with same name should be present in the target directory before running the script as the result file will be not overwritten but more lines will be added at it.

*IMPORTANT: All the scripts are set to run with the current folder tree architecture.  Any changes in the structure should be incorporated into the scripts in their respective paths definition.*