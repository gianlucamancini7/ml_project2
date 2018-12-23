# MoTUS machine learning approach - Machine Learning Project 2

Machine learning Project 2 Option A – CS-433 – EPFL<br>
Fall 2018<br>
*Authors:* Gianluca Mancini, Tullio Nutta and Tianchu Zhang<br>
*Laboratory:* LESO-PB – Solar Energy and Building Physics Laboratory<br>
*Supervisors:* Roberto Castello and Dasaraden Mauree

## Scripts outline

- **data_extractor_v1 to v3.ipynb:** averages the raw data starting from non compressed, `.zip` and `.7z` respectively. v3 is unstable (probable unexpected error at some point) due to high compresion ratio of `.7z` files. Input file should be places in `/data_zip`. v2 version used. 

- **regression_mat_builder_v1.ipynb:** takes file in `/average_data_year` and creates `regression_mat_year.csv` (~200MB) cleaning the data.

- **data_extractor_7zip and _zip.ipynb:** `.py` files of the datat extractor v3 and v2 respectively. Ready to run on the server.

## Folders and files outline
- *average_data_year:* contains already a preprocessed files for data of almost a year (January to mid November). 

- *data_ext_original:* all the scripts already provided to extract the data

- *data_zip:* empty folder. Set input data path for data extractor files.

- *regression_mat_year.csv:* averaged (on a 5 minutes time interval) and condensed data. Consider data from January to mid November.


