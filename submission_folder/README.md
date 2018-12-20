# Machine Learning Project 2

Machine learning Project 2 Option A – CS-433 – EPFL<br>
Fall 2018<br>
*Authors:* Gianluca Mancini, Tullio Nutta and Tianchu Zhang<br>
*Laboratory:* LESO-PB – Solar Energy and Building Physics Laboratory<br>
*Supervisors:* Roberto Castello and Dasaraden Mauree


## Directory tree
```
│   README.md
│
├───data
│   ├───average_data
│   └───zip_files
├───results
│   ├───baseline
│   ├───neural_network
│   ├───random_forest
│   ├───results_analysis
│   └───ridge_regression
├───scripts_data_extractor
│   │   data_extractor.ipynb
│   │   README.md
│   │   regression_mat_builder.ipynb
│   │
│   └───.ipynb_checkpoints
│           data_extractor-checkpoint.ipynb
│           regression_mat_builder-checkpoint.ipynb
│
├───scripts_features_selection
└───scripts_regression
```

*IMPORTANT: All the scripts are set to run with the current folder tree architecture. The directory structure is self-contained. Any changes in the structure should be incorporated into the scripts in their respective paths definition.*

## Data download
All the required data to run the algorithms can be found at [this link](https://enacshare.epfl.ch/d6GU2cHxX8pti3W7VSkPu) (download only). Additional useful information about the download can be found at the `/data` folder [README.md](./data/README.md).


## Hardware requirements
Random forest and ridge regression require at least 32GB of RAM to run with the current settings. To run them with lower memory, follow the instruction of the `README.md` in `/scripts` folder. All the other scripts (neural network included) can operate with 4GB of RAM, but 8GB is advised. 

## Required packages
 - numpy - 1.14.3
 - pandas - 0.23.0
 - scipy - 1.1.0
 - matplotlib.pyplot - 2.2.2
 - pytorch-cpu - 0.4.1
 - scikit-learn - 0.20.0
 - scikit-image - 0.13.1
 
 **All the instruction concerning single scripts are contained in the READMEs in the subfolders.**
 