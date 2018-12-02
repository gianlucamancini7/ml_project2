
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import KBinsDiscretizer

pd.set_option('display.max_columns', 100)


# In[ ]:
INPUT_PATH = '~/scripts/'
OUTPUT_PATH = '/raid/motus/results/randomforest/'

def season_splitter(df):
    df.index = pd.to_datetime(df.index)
    df_spring = df[(df.index > '2018-03-20') & (df.index <= '2018-06-20')]
    df_summer = df[(df.index > '2018-06-21') & (df.index <= '2018-09-21')]
    df_autumn = df[(df.index > '2018-09-21') & (df.index <= '2018-12-21')]
    df_winter = df[(df.index > '2018-12-21') + (df.index <= '2018-03-20')]
    return df_spring, df_summer, df_autumn, df_winter


# # Load and preprocess the data

# In[65]:


tot_df=pd.read_csv('regression_mat_year.csv',index_col=0)


# In[66]:


tot_df=pd.read_csv(INPUT_PATH + 'regression_mat_year.csv',index_col=0)
# create columns with coordinate velocities output
tot_df['u_x']=tot_df['u']*np.cos(np.radians(tot_df['direction']))
tot_df['u_y']=tot_df['u']*np.sin(np.radians(tot_df['direction']))
# create columns with coordinate velocities input top mast anemometer
tot_df['u_top_x']=tot_df['u_top']*np.cos(np.radians(tot_df['direction_top']))
tot_df['u_top_y']=tot_df['u_top']*np.sin(np.radians(tot_df['direction_top']))
# drop the columns which are not used anymore
tot_df=tot_df.drop(columns=['u', 'u_top', 'direction', 'direction_top'])
tot_df=tot_df.iloc[0:,:]


# # Random forest feature selection
# Pipeline: Discretize the output -> Random forest
# <br>Output: u_x, u_y, z

# ## Prepare the input and output

# In[12]:


x = np.array(tot_df.drop(columns=['u_x', 'u_y','u_z']))
y_continue = np.array(tot_df[['u_x', 'u_y']])


# ## Discretize the output

# In[13]:


discretizer = KBinsDiscretizer(n_bins=20, encode='ordinal', strategy='uniform')
discretizer.fit(y_continue)
y_disc = discretizer.transform(y_continue)


# ## Split train and test

# In[14]:


x_tr, x_te, y_tr, y_te = train_test_split(x, y_disc, test_size=0.3, random_state=42)


# ## Random forest

# In[15]:


#y_tr_cont = discretizer.inverse_transform(y_tr)


# In[16]:


rf = RandomForestClassifier(n_estimators=1000, max_depth=None, criterion='gini', random_state=0)
rf.fit(x_tr, y_tr)


# ## Print the result(feature importance)

# In[18]:


feat_labels = tot_df.drop(columns=['u_x', 'u_y','u_z']).columns


# In[74]:


importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
important_features = []
importance_accum = 0
#open("feature_importance.txt", 'w').close
filetxt = open(OUTPUT_PATH + "FS_RF_full.txt", "w")
filetxt.write("\n For the full year: \n")
for f in range(x_tr.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 50, feat_labels[indices[f]], importances[indices[f]]))
    filetxt.write("%2d) %-*s %f \n" % (f + 1, 50, feat_labels[indices[f]], importances[indices[f]]))
    if importance_accum < 0.80:
        importance_accum = importance_accum + importances[indices[f]]
        important_features.append(feat_labels[indices[f]])
filetxt.write("\n The top 80% important features are: \n")
for i in range(len(important_features)):
    filetxt.write("%s \n" % important_features[i])
filetxt.write("%i features on %i" % (len(important_features), x_tr.shape[1]))
filetxt.close()

