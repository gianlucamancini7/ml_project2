
# coding: utf-8

# In[7]:


#Imports
import pandas as pd
import numpy as np
from IPython.core import display as ICD
import glob
import os
from zipfile import ZipFile
pd.set_option('display.max_columns', 100)
import py7zlib


# In[8]:


print('### --- MoTUS Data Extractor ---###')
print('')


# In[5]:


# General timestamp
# ----------------------------------- #
#List of some available timestamp, it is possible to include more
# B       business day frequency
# D       calendar day frequency
# W       weekly frequency
# M       month end frequency
# SM      semi-month end frequency (15th and end of month)
# MS      month start frequency
# SMS     semi-month start frequency (1st and 15th)
# Q       quarter end frequency
# QS      quarter start frequency
# A       year end frequency
# AS      year start frequency
# H       hourly frequency
# T       minutely frequency
# S       secondly frequency
# L       milliseonds
# U       microseconds
# N       nanoseconds
time_int='T'
#Data folder
DATA_FOLDER = r'/Users/gianlucamancini/Desktop/7zip_files/'
AVERAGE_RES_FOLDER = r'/Users/gianlucamancini/Desktop/'
name_file="guru99.txt"

print('Time interval: ',time_int)
print('Folder with zip files: ',DATA_FOLDER)
print('Folder for save results: ',AVERAGE_RES_FOLDER)
print('')


# In[6]:


###  Anemometer averaging data  ###
# ----------------------------------- #
print('Anemometers in progress...')
items_anem = np.arange(1,8)
columns_anem=[1,2,3,6,7]
time_stp_position_anem=-1 #Respect to columns_anem

zip_folders=os.listdir(DATA_FOLDER)


for zip_folder in zip_folders:
    
    print("In progress: {}".format(zip_folder), end="\r")
    
    content=open(DATA_FOLDER+zip_folder ,"rb")
    archive = py7zlib.Archive7z(content)
    

    for day_file in archive.getnames():
        for i in items_anem:
            if day_file.endswith("anem" + str(i) +'_20Hz'+'.txt'):
                data = archive.getmember(day_file).read().decode('utf-8')
                f= open(AVERAGE_RES_FOLDER+"guru99.txt","w+")
                f.write(data)
                f.close()
                df_temp=pd.read_csv(AVERAGE_RES_FOLDER+name_file, header=None, comment=',', error_bad_lines=False,usecols=columns_anem, index_col=time_stp_position_anem)
                df_temp.index=pd.to_datetime(df_temp.index,format='%d.%m.%Y %H:%M:%S',errors='coerce')
                df_temp=df_temp.resample(time_int).mean()[:-1]
                df_temp.to_csv(AVERAGE_RES_FOLDER+'anem'+ str(i)+'.csv', header=None, index=True, sep=',', mode='a')
print('Anemometers OK') 
print('')


# In[8]:


###  Temperature averaging data  ###
# ----------------------------------- #
print('Temperatures in progress...')
items_temp = [['Mat','North','East','South','West'],['Sensor Ground temperature [°C]','North temperature [°C]','East temperature [°C]','South temperature [°C]','West temperature [°C]']]
columns_temp=[0,2]
time_stp_position_temp=-1 #Respect to columns_anem

zip_folders=os.listdir(DATA_FOLDER)

for day,zip_folder in enumerate(zip_folders):
    print("In progress: {}".format(zip_folder), end="\r")
    content=open(DATA_FOLDER+zip_folder ,"rb")
    archive = py7zlib.Archive7z(content)
    k=0
    for day_file in archive.getnames():

        for i in items_temp[0]:

            if day_file.endswith('Temp'+i+'.txt'):
                
                data = archive.getmember(day_file).read().decode('utf-8')
                f= open(AVERAGE_RES_FOLDER+"guru99.txt","w+")
                f.write(data)
                f.close()
                
                df_temp=pd.read_csv(AVERAGE_RES_FOLDER+name_file, header=None, comment=',', error_bad_lines=False,usecols=columns_temp, index_col=time_stp_position_temp)
                df_temp.index=pd.to_datetime(df_temp.index,format='%d.%m.%Y %H:%M:%S',errors='coerce')
                df_temp=df_temp.resample(time_int).mean()
                df_temp.columns = [items_temp[1][k]]
                if k==0:
                    df_tot=df_temp
                else:
                    df_tot=df_tot.merge(df_temp,how='left', left_index=True,right_index=True)
                k=k+1
    if day==0:
        df_tot.to_csv(AVERAGE_RES_FOLDER+'surf_temp'+'.csv', index=True, sep=',', mode='a')
    else:
        df_tot.to_csv(AVERAGE_RES_FOLDER+'surf_temp'+'.csv', index=True,header=None,sep=',', mode='a')
print('Temperatures OK')
print('')


# In[10]:


###  Radiometer averaging data  ###
# ----------------------------------- #
print('Radiometer in progress...')
columns_radio=list(range(10))+[11]
time_stp_position_temp=-1 #Respect to columns_radio
items_radio=['Pyranometer Upper Irradiance [W/m$^2$]','Pyranometer Lower Irradiance [W/m$^2$]',             'Pyrgeometer Upper Irradiance [W/m$^2$]','Pyrgeometer Lower Irradiance [W/m$^2$]',             'Albedo [-]','Net Solar radiation [W/m$^2$]','Net (total) radiation [W/m$^2$]',             'Net Far Infrared radiation [W/m$^2$]','Sky temperature [°C]','Radiometer Ground temperature [°C]']

zip_folders=os.listdir(DATA_FOLDER)
for day,zip_folder in enumerate(zip_folders):
    print("In progress: {}".format(zip_folder), end="\r")
    
    content=open(DATA_FOLDER+zip_folder ,"rb")
    archive = py7zlib.Archive7z(content)
    
    for day_file in archive.getnames():
        if day_file.endswith('radiometre'+'.txt'):
            
            data = archive.getmember(day_file).read().decode('utf-8')
            f= open(AVERAGE_RES_FOLDER+"guru99.txt","w+")
            f.write(data)
            f.close()
                
            df_temp=pd.read_csv(AVERAGE_RES_FOLDER+name_file, header=None, comment=',', error_bad_lines=False,usecols=columns_radio, index_col=time_stp_position_temp)
            df_temp.index=pd.to_datetime(df_temp.index,format='%d.%m.%Y %H:%M:%S',errors='coerce')
            df_temp=df_temp.resample(time_int).mean()
            df_temp.columns = items_radio
            if day==0:
                df_temp.to_csv(AVERAGE_RES_FOLDER+'radiometer'+'.csv', index=True, sep=',', mode='a')
            else:
                df_temp.to_csv(AVERAGE_RES_FOLDER+'radiometer'+'.csv', index=True,header=None, sep=',', mode='a')

print('Temperatures OK')
print('JOB Finished')

