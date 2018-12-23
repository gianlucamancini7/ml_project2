# MoTUS machine learning approach - Machine Learning Project 2

Machine learning Project 2 Option A – CS-433 – EPFL<br>
Fall 2018<br>
*Authors:* Gianluca Mancini, Tullio Nutta and Tianchu Zhang<br>
*Laboratory:* LESO-PB – Solar Energy and Building Physics Laboratory<br>
*Supervisors:* Roberto Castello and Dasaraden Mauree

## Abstract
The purpose of the project was to select the most important features and investigate possible machine learning algorithm on the 2018 data collected through the MoTUS setup. 

## Directory tree
```
├───data_extractor_scripts
│   ├───.ipynb_checkpoints
│   ├───average_data_year
│   └───data_zip
├───exploratory_analysis
├───final_submission_scripts
│   ├───.ipynb_checkpoints
│   ├───images
│   ├───py_scripts
│   │   └───.ipynb_checkpoints
│   └───__pycache__
├───intermediate_work_scripts
├───report
└───submission_folder
    ├───data
    │   ├───average_data
    │   └───zip_files
    ├───results
    │   ├───baseline
    │   │   └───images
    │   ├───neural_network
    │   ├───random_forest
    │   ├───results_analysis
    │   └───ridge_regression
    │       └───images
    ├───scripts_data_extractor
    ├───scripts_features_selection
    └───scripts_regression
        ├───.ipynb_checkpoints
        └───__pycache__
```

## Overview
This repository, other than the definitive results and scripts, contains all the intermediate goals for possible future investigation of the project development. The general content of the files (mainly `.ipynb`) and folders of this directory is shown below. Subfolder readmes will be present.

- **data_extractor_scripts:** contains all the different versions of the code used to convert the raw data into the regression matrix used in the machine learning algorithms. A final regression matrix is also included.

- **exploratory_analysis:** includes all the scripts used firstly to investigate the raw data. No notable operations are performed with these scripts.

- **final_submission_scripts:** contains the scripts able to efficiently run the 3 selected regression methods. It includes all the accessory files and `.py` conversion of the notebooks used run the scripts on the laboratory server. 

- **intermediate_work_scripts:** includes the intermediate file used to run the explorative version of the regression. 

- **report:** contains the final report for the CS-433 course, together with its appendix and the guidelines for the project.

- **submission_folder:** includes the final version of all the script necessary to produce the conclusion present in the report starting from the raw data.

- *SCRIPTS_RULES.txt:* general rules and procedures used throughout the project.

**IMPORTANT: For working scripts and clean results always refer to `/submission_folder`. All codes have appropriate readmes and are adequately commented.**
