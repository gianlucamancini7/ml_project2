#  README scripts_regression

Machine learning Project 2 Option A – CS-433 – EPFL<br>
Fall 2018<br>
*Authors:* Gianluca Mancini, Tullio Nutta and Tianchu Zhang<br>
*Laboratory:* LESO-PB – Solar Energy and Building Physics Laboratory<br>
*Supervisors:* Roberto Castello and Dasaraden Mauree

## Scripts info
- **ridge_regression.ipynb:** performs the cross- validated ridge regression. Description of all the settings of the scripts is contained inside under *Settings* markdown cell. By default, results (plots and 5 ``.txt``, whose content and structure is descibed in the notebook) are saved in ``/results/ ridge_regression``. Be sure that an empty image folder is already present in the result target directory. Saved plot will be overwritten; data will be added on the ``.txt`` file if already present, if not a new file is created. Detailed information on how to deal with possible ``MemoryError`` are in the script under *Settings* markdown cell. Generally, it is useful to reduce the training matrix dimension. 

- **baseline.ipynb:** performs the baseline ridge regression. Description of all the settings of the scripts is contained inside under *Settings* markdown cell. By default, results (plots and 5 ``.txt`` files, whose content and structure is descibed in the notebook) are saved in ``/results/baseline``. Be sure that an empty image folder is already present in the result target directory. Saved plot will be overwritten; data will be added on the ``.txt`` file if already present, if not a new file is created. Detailed information on how to deal with possible ``MemoryError`` are in the script under *Settings* markdown cell. Generally, it is useful to reduce the training matrix dimension. 

- **neural_network.ipynb:** provides a simple neural network architecture. Description of all the settings of the scripts is contained inside under *Settings* markdown cell. By default the notebook output the mean squared errors and $R^2$ for different anemometers for different seasons. On top of that, the mean values of the wind speed at each anemometer is also saved in order to produce the wind speed profile inside the urban canyon. Also the evolution of the train and evaluation mse during the training of the network are saved as well as the visualization of the fitting of the netowork for different anemometers. The results are saved in ``/results/neural_network``.  Saved plot will be overwritten.

- **random_forest.ipynb:**

- **result_analysis.ipynb:**

- **helpers.py:** contains all the functions called in the scripts. It needs to have the same location of the notebooks. 

## Files info
- **feature_for_ridge.txt:** contains the feature list to run one option of `` ridge_regression.ipynb`` and `` baseline.ipynb``. can be created through ``/scripts_features_selection/random_forest.ipynb``. 


*IMPORTANT: All the scripts are set to run with the current folder tree architecture.  Any changes in the structure should be incorporated into the scripts in their respective paths definition.*