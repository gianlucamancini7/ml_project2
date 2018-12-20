#  README scripts_regression

Machine learning Project 2 Option A – CS-433 – EPFL<br>
Fall 2018<br>
*Authors:* Gianluca Mancini, Tullio Nutta and Tianchu Zhang<br>
*Laboratory:* LESO-PB – Solar Energy and Building Physics Laboratory<br>
*Supervisors:* Roberto Castello and Dasaraden Mauree

## Scripts info
- **ridge_regression.ipynb:** perform the cross- validated ridge regression. Description of all the settings of the scripts is contained inside under *settings* markdown cell. By default, results (plots and 4 ``.txt`` files) are saved in ``/results/ ridge_regression``. Be sure that an empty image folder is already present in the result target directory. Saved plot will be overwritten; data will be added on the ``.txt`` file if already present.

- **baseline.ipynb:** perform the baseline ridge regression. Description of all the settings of the scripts is contained inside under *settings* markdown cell. By default, results (plots and 4 ``.txt`` files) are saved in ``/results/baseline``. Be sure that an empty image folder is already present in the result target directory. Saved plot will be overwritten; data will be added on the ``.txt`` file if already present.

- **neural_network.ipynb:**

- **random_forest.ipynb:**

- **result_analysis.ipynb:**

- **helpers.py:** contains all the functions used in the regression scripts. It needs to have the same location at the regression notebooks. 

## Files info
- **feature_for_ridge.txt:** contains the feature list to run one option of `` ridge_regression.ipynb`` and `` baseline.ipynb``. can be created through ``/scripts_features_selection/random_forest.ipynb``. 


*IMPORTANT: All the scripts are set to run with the current folder tree architecture.  Any changes in the structure should be incorporated into the scripts in their respective paths definition.*