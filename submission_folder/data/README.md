#  README data

Machine learning Project 2 Option A – CS-433 – EPFL<br>
Fall 2018<br>
*Authors:* Gianluca Mancini, Tullio Nutta and Tianchu Zhang<br>
*Laboratory:* LESO-PB – Solar Energy and Building Physics Laboratory<br>
*Supervisors:* Roberto Castello and Dasaraden Mauree

## Download instructions

- All the required data to run the algorithms can be found at [this link](https://enacshare.epfl.ch/d6GU2cHxX8pti3W7VSkPu) (download only). **REMARK:** The files downloaded from the link will be compressed into a ``.zip`` or ``.tar.gz``. Before placing them in the correct place, don’t forget to unzip the archive downloaded (so ``.._Mesures_MoTUS.zip`` need all to be separate in ``/data/zip_files`` as the ``.csv``’s divided in the described locations. 
- ``regression_mat_year`` (~200MB) should be placed in ``/data``. **This in the only file required to run regression algorithms, all the rest is raw and intermediate processed files**.
- All the ``.zip`` (~50GB) folders shoud be downloaded in ``/data/zip_files``.
- The others ``.csv`` (~90MB) (``anem1`` to ``anem7``, ``radiometer`` and ``surf_temp`` should be downloaded in ``/data/average_data`` (intermediate pipeline results).

The scripts provided in ``/scripts_data_extractor`` can construct the final matrix from the zip files. However, also intermediate results are provided (``/data/average_data``). If intermediate pipeline script is chosen to run be sure that the target directory is empty. Additional useful information about the scripts can be found at the `/ scripts_data_extractor ` folder [README.md](./ scripts_data_extractor /README.md).


*IMPORTANT: All the scripts are set to run with the current folder tree architecture.  Any changes in the structure should be incorporated into the scripts in their respective paths definition.*