
# coding: utf-8

# In[ ]:


import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', 100)


# In[ ]:


# import scikit learn packages
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

from sklearn.pipeline import make_pipeline


# In[ ]:


from helpers import *


# **Settings**
# 
# *Boleans*
# - ``seasonwise``: If set to ``True`` perform 4 different regression splitting the dataset season by season. Otherwise perform one single regression.
# - ``feature_selection``: If set to ``True`` filter only season selected by random forest features selection (80% importance, 13 out of 20 features). Otherwise utilise all the 20 features.
# - ``output_y_only``: If set to ``True`` perform regression considering only ``u_y`` as output. Otherwise consider speed in both $x$ and $y$.
# 
# *Parameters*
# - ``rnd_state``: Seed state used in the splitting of the dataset. Default in all algorithms is $50$.
# - ``alphas``: Possible choice of regularization parameter optimized by the ridge regression. Default is `` np.logspace(-10,5,200)``.
# - ``deg``: Degree of the polynomial expansion. Default is $3$.
# - ``train_dim``,``test_dim``,``validate_dim``: Dimension of the splitting. Default are respectively $0.6$, $0.2$ and $0.2$.
# 
# *Memory problem:* If ``MemoryError`` arise (with current parameters and 32GB of ram would be very unlikely), different changes can be done to make the script less RAM heavy. With  `` seasonwise = True`` the regression is performed seasonally and the dataset on which the regression in performed is $1/4$ in dimension. Other matrix dimension reduction can be done maintaining the regression yearly lowering the polynomial degree of expansion (``deg``) or lowering  the dimension of training dataset (``train_dim``). The latter operations reduce overall performance of the algorithm.

# In[ ]:


seasonwise = True
feature_selection = False
output_y_only = False
rnd_state=50
alphas = np.logspace(-10,5,200)
deg = 3
train_dim = 0.6
test_dim = 0.2
validate_dim = 0.2

print('-seasonwise: ',seasonwise,'    -feature_selection: ',feature_selection,'    -output_y_only: ',output_y_only)

seasons_list = ['spring','summer','autumn','winter']

consistency_splitting(train_dim, test_dim ,validate_dim)

model = make_pipeline(StandardScaler(),PolynomialFeatures(deg,include_bias=True), RidgeCV(alphas))
print(model)
print('')


# In[ ]:


if seasonwise==False:
    feature_selection = False


# Paths of the scripts. 

# In[ ]:


#Local
# DATA_FOLDER = r'../data/'
# RESULTS_FOLDER = r'../results/ridge_regression/'
#Server
DATA_FOLDER = '~/scripts/'
RESULTS_FOLDER = '/raid/motus/results/ridgeregression/'

# In[ ]:


tot_df=pd.read_csv(DATA_FOLDER+'regression_mat_year.csv',index_col=0)


# Imports of the features selected by the random forest.

# In[ ]:


if feature_selection:
    df_features_sel = pd.read_csv('feature_for_ridge_season.txt',header=None)
    df_features_sel = df_features_sel.drop(df_features_sel.columns[0], axis=1)
    lst_features_sel=np.array(df_features_sel).tolist()


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
std_true_results=[]


# In[ ]:


for index,df in enumerate(season_dfs):
    if len(season_dfs)>2:
        names=seasons_list
    else:
        names=['allyear']
    
    #Printing progress
    print('Period under optimization: ',names[index])
    
    #Dividing X and y
    X = df.drop(columns=['u_x', 'u_y','u_z'])
    if feature_selection:
        print('Features considered: ',lst_features_sel[index])
        print('')
        X=np.array(X[lst_features_sel[index]])
        h_position=lst_features_sel[index].index('h')
    else:
        X=np.array(X)
    
    #Chose 1D or 2D output
    if output_y_only:
        y = np.array(df['u_y']) 
    else:
        y = np.array(df[['u_x', 'u_y']])
    
    
    #Splitting Matrices
    X_tr, X_temp, y_tr, y_temp = train_test_split(X, y, test_size=test_dim+validate_dim, random_state=rnd_state)
    X_te, X_va, y_te, y_va = train_test_split(X_temp, y_temp, test_size=validate_dim/(test_dim+validate_dim), random_state=rnd_state)
    X_temp = None
    y_temp = None
    
    #Make a list with differet heights
    if feature_selection:
        X_te_hs, y_te_hs = split_hs_test(X_te,y_te,h_pos=h_position)
    else:
        X_te_hs, y_te_hs = split_hs_test(X_te,y_te)
    #print('X_shape: ',X.shape)
    
    #Fit the model 
    model.fit(X_tr,y_tr)
    
    y_pred_hs=[]
    mag_avg_pred=[]
    #Computing and saving the predictions and the average magnitude
    for hs in X_te_hs:
        ans=model.predict(hs)
        y_pred_hs.append(ans)
        ans=magnitude_avg(ans)
        #print(ans.shape)
        mag_avg_pred.append(ans)
    mag_avg_pred.append(names[index])
    mag_avg_pred_results.append(mag_avg_pred)

    
    mses=[]
    mag_avg_true=[]
    std_true=[]
    #Computing and saving mse, true magnitude averaged, true std
    for idx,i in enumerate(y_pred_hs):
        mses.append(mean_squared_error(y_te_hs[idx],i))
        Ans=y_te_hs[idx]
        #ans=np.sqrt(np.sum(np.square(ans),axis=1)).mean()
        ans=magnitude_avg(Ans)
        mag_avg_true.append(ans)
        ans=std_avg(Ans)
        std_true.append(ans)
    mses.append(names[index])
    mag_avg_true.append(names[index])
    std_true.append(names[index])
    mses_results.append(mses)
    mag_avg_true_results.append(mag_avg_true)
    std_true_results.append(std_true)
    
    r_2s=[]
    #Computing and saving r_2 metric
    for idx,i in enumerate(y_pred_hs):
        r_2s.append(r2_score(y_te_hs[idx],i))
    r_2s.append(names[index])
    r_2s_results.append(r_2s)
    
    #Plot and save graphs. Save or show can be chosen in bolean save.
    if output_y_only:
        plot_ys_single(y_pred_hs,y_te_hs,RESULTS_FOLDER+'/images/',save=True,name=(names[index]+'-'))
    else:
        plot_ys(y_pred_hs,y_te_hs,RESULTS_FOLDER+'/images/',save=True,name=(names[index]+'-'))
    


# Save results in .txt in the target results folder. 

# In[ ]:


mses_results_df=pd.DataFrame(mses_results)
mses_results_df.to_csv(RESULTS_FOLDER+'mses_u_seasons.txt',header=None, sep=',', mode='a')


# In[ ]:


r_2s_results_df=pd.DataFrame(r_2s_results)
r_2s_results_df.to_csv(RESULTS_FOLDER+'rsquared_u_seasons.txt',header=None, sep=',', mode='a')


# In[ ]:


pd.DataFrame(mag_avg_pred_results).to_csv(RESULTS_FOLDER+'magnitude_average_pred.txt',header=None, sep=',', mode='a')
pd.DataFrame(mag_avg_true_results).to_csv(RESULTS_FOLDER+'magnitude_average_true.txt',header=None, sep=',', mode='a')
pd.DataFrame(std_true_results).to_csv(RESULTS_FOLDER+'magnitude_std_true.txt',header=None, sep=',', mode='a')


# In[ ]:


print('JOB Finished')


# Plot names and title on the picture itself explain their content. An interval is chosen randomly to visualize the behaviour on the true value and the prediction. The MSE, $R^2$, average prediction, average true values and average standard deviations are all saved on ``.txt`` format. The last entry of each line identifies the period (season or all year), while every number not considering the first one is referring to the mast anemometer from 1 to 6 in this order.
