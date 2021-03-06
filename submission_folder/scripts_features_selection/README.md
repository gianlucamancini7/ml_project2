#  README scripts_features_selection

Machine learning Project 2 Option A – CS-433 – EPFL<br>
Fall 2018<br>
*Authors:* Gianluca Mancini, Tullio Nutta and Tianchu Zhang<br>
*Laboratory:* LESO-PB – Solar Energy and Building Physics Laboratory<br>
*Supervisors:* Roberto Castello and Dasaraden Mauree

## Scripts info
- **feature_selection_stepwise.ipynb:** provides an exploratory analysis of a stepwise feature selection. The script doesn't output any files. This script was not used to select the most important features in this investigation. The rationale behind this method is explained in the notebook itself, but the method's success depends on the regression's success at each iteration. In fact the results that are produced in the code are not accurate since the feature selection is based on a regression carried out with least square which doesn't provide an accurate fit of the data. In order to make it accurate a ridge regression should be performed and the algorithm would be computationally not feasible.

- **helpers.py:** contains all the functions called in the scripts. It needs to have the same location of the notebooks. 

**REMARK:** The feature selection of the random forest is performed inside ``/scripts_regression/random_forest.ipynb``. Please refer to that file for random forest feature selection. ``/scripts_regression/random_forest.ipynb`` output also `` feature_for_ridge_season.txt`` in ``/results/results_analysis``. To use the new file in the baseline and in the ridge regression, move manually and overwrite it in ``/scripts_regression/``.

*IMPORTANT: All the scripts are set to run with the current folder tree architecture.  Any changes in the structure should be incorporated into the scripts in their respective paths definition.*