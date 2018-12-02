
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import glob
import scipy

# In[ ]:


# import scikit learn packages

from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

from sklearn.pipeline import make_pipeline


# In[ ]:


from helpers import *


# In[ ]:


DATA_FOLDER = '~/scripts/'
RESULTS_FOLDER = '/raid/motus/results/ridgeregression/'


# In[ ]:


tot_df=pd.read_csv(DATA_FOLDER+'regression_mat_year.csv',index_col=0)


# Transform absolute value and direction in vector components

# In[ ]:


tot_df = vectorize_wind_speed(tot_df)


# Split train and test and take columns considered

# In[ ]:


X = np.array(tot_df.drop(columns=['u_x', 'u_y','u_z']))
y = np.array(tot_df[['u_x', 'u_y']])


# In[ ]:


X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)


# Split the test for different heighs

# In[ ]:


X_te_hs, y_te_hs = split_hs_test(X_te,y_te)


# Regression

# In[ ]:


alphas=np.logspace(-10,10,200)
degrees=np.arange(1,7)

mses_results=[]
r_2s_results=[]
for deg in degrees:
    print('Degree in processing: ',deg)
    #define pipeline
    model = make_pipeline(StandardScaler(),PolynomialFeatures(deg), RidgeCV(alphas))

    model.fit(X_tr,y_tr)
    y_pred_hs=[]
    for hs in X_te_hs:
        y_pred_hs.append(model.predict(hs))
    
    mses=[]
    for idx,i in enumerate(y_pred_hs):
        mses.append(mean_squared_error(y_te_hs[idx],i))
    mses.append(deg)
    mses_results.append(mses)
    
    r_2s=[]
    for idx,i in enumerate(y_pred_hs):
        r_2s.append(r2_score(y_te_hs[idx],i))
    r_2s.append(deg)
    r_2s_results.append(r_2s)
    
    plot_ys(y_pred_hs,y_te_hs,RESULTS_FOLDER+'/images/',save=True,name=('deg'+str(deg)+'-'))


# In[ ]:


mses_results_df=pd.DataFrame(mses_results)
mses_results_df.to_csv(RESULTS_FOLDER+'mses_results.txt',header=None, sep=',', mode='a')


# In[ ]:


r_2s_results_df=pd.DataFrame(r_2s_results)
r_2s_results_df.to_csv(RESULTS_FOLDER+'r_2s_results.txt',header=None, sep=',', mode='a')

print('JOB Finished')