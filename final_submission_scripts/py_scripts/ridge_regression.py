
# coding: utf-8

# In[ ]:


import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from IPython.core import display as ICD
import seaborn as sns
import glob
import scipy
import os
pd.set_option('display.max_columns', 100)


# In[ ]:


# import scikit learn packages

from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA

from sklearn.pipeline import make_pipeline


# In[ ]:


from helpers import *


# **Parameters**

# In[ ]:


seasonwise = True
feature_selection = False
alphas = np.logspace(-10,5,200)
deg = 3
train_dim = 0.6
test_dim = 0.2
validate_dim = 0.2

consistency_splitting(train_dim, test_dim ,validate_dim)

model = make_pipeline(StandardScaler(),PolynomialFeatures(deg,include_bias=True), RidgeCV(alphas))


# In[ ]:


#Local
# DATA_FOLDER = r'../data_extractor_scripts/'
# RESULTS_FOLDER = r'./'
#Server
DATA_FOLDER = '~/scripts/'
RESULTS_FOLDER = '/raid/motus/results/ridgeregression/'


# In[ ]:


tot_df=pd.read_csv(DATA_FOLDER+'regression_mat_year.csv',index_col=0)


# Transform absolute value and direction in vector components

# In[ ]:


tot_df = vectorize_wind_speed(tot_df)


# Split season by season

# In[ ]:


if seasonwise:
    df_spring, df_summer, df_autumn, df_winter = season_splitter(tot_df)
    season_dfs=[df_spring, df_summer, df_autumn, df_winter]
    
else:
    season_dfs=[tot_df]


# In[ ]:


#Empty dataframes
mses_results=[]
r_2s_results=[]
mag_avg_pred_results=[]
mag_avg_true_results=[]


# In[ ]:


for index,df in enumerate(season_dfs):
    if len(season_dfs)>2:
        names=['spring','summer','autumn','winter']
    else:
        names=['allyear']
    
    #Printing progress
    print('Period under optimization: ',names[index])
    
    if feature_selection:
        ### To be implemented (use idx to cleane the dataframes)
        raise NotImplementedError('Not yet Implemented')
    
    #Dividing X and y
    X = np.array(tot_df.drop(columns=['u_x', 'u_y','u_z']))
    y = np.array(tot_df[['u_x', 'u_y']])
    
    #Splitting Matrices
    X_tr, X_temp, y_tr, y_temp = train_test_split(X, y, test_size=test_dim+validate_dim, random_state=12)
    X_te, X_va, y_te, y_va = train_test_split(X_temp, y_temp, test_size=validate_dim/(test_dim+validate_dim), random_state=12)
    X_temp = None
    y_temp = None
    
    #Make a list with differet heights
    X_te_hs, y_te_hs = split_hs_test(X_te,y_te)
    
    #Fit the model 
    model.fit(X_tr,y_tr)
    
    y_pred_hs=[]
    mag_avg_pred=[]
    for hs in X_te_hs:
        ans=model.predict(hs)
        y_pred_hs.append(ans)
        ans=np.sqrt(ans[:,0]**2+ans[:,1]**2).mean()
        mag_avg_pred.append(ans)
    mag_avg_pred.append(names[index])
    mag_avg_pred_results.append(mag_avg_pred)

    
    mses=[]
    mag_avg_true=[]
    for idx,i in enumerate(y_pred_hs):
        mses.append(mean_squared_error(y_te_hs[idx],i))
        ans=y_te_hs[idx]
        ans=np.sqrt(ans[:,0]**2+ans[:,1]**2).mean()
        mag_avg_true.append(ans)
    mses.append(names[index])
    mag_avg_true.append(names[index])
    mses_results.append(mses)
    mag_avg_true_results.append(mag_avg_true)
    
    r_2s=[]
    for idx,i in enumerate(y_pred_hs):
        r_2s.append(r2_score(y_te_hs[idx],i))
    r_2s.append(names[index])
    r_2s_results.append(r_2s)
    
    plot_ys(y_pred_hs,y_te_hs,RESULTS_FOLDER+'/images/',save=True,name=(names[index]+'-'))


# In[ ]:


mses_results_df=pd.DataFrame(mses_results)
mses_results_df.to_csv(RESULTS_FOLDER+'mses_results.txt',header=None, sep=',', mode='a')


# In[ ]:


r_2s_results_df=pd.DataFrame(r_2s_results)
r_2s_results_df.to_csv(RESULTS_FOLDER+'r_2s_results.txt',header=None, sep=',', mode='a')


# In[ ]:


pd.DataFrame(mag_avg_pred_results).to_csv(RESULTS_FOLDER+'magnitude_average_pred.txt',header=None, sep=',', mode='a')
pd.DataFrame(mag_avg_true_results).to_csv(RESULTS_FOLDER+'magnitude_average_true.txt',header=None, sep=',', mode='a')


# In[ ]:


print('JOB Finished')

