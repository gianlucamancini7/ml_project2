#  README scripts_features_selection

Machine learning Project 2 Option A – CS-433 – EPFL<br>
Fall 2018<br>
*Authors:* Gianluca Mancini, Tullio Nutta and Tianchu Zhang<br>
*Laboratory:* LESO-PB – Solar Energy and Building Physics Laboratory<br>
*Supervisors:* Roberto Castello and Dasaraden Mauree

## Scripts info
- **feature_selection_stepwise.ipynb:** Provides an exploratory analysis of a stepwise feature selection. The script doesn't output any code. This script was not used to select the most important features in this investigation. The rationale behind this method is explained in the notebook itself, but the method's success depends on the regression's success at each iteration. In fact the results that are produced in the code are not accurate since the feature selection is based on a regression carried out with least square which doesn't provide an accurate fit of the data.

*IMPORTANT: All the scripts are set to run with the current folder tree architecture.  Any changes in the structure should be incorporated into the scripts in their respective paths definition.*