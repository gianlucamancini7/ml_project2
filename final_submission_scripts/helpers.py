
# coding: utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def season_splitter(df):
    """Returs 4 different datasets for each astronomical season in year 2018""" 
    df.index = pd.to_datetime(df.index)
    df_spring = df[(df.index > '2018-03-20') & (df.index <= '2018-06-20')]
    df_summer = df[(df.index > '2018-06-21') & (df.index <= '2018-09-21')]
    df_autumn = df[(df.index > '2018-09-21') & (df.index <= '2018-12-21')]
    df_winter = df[(df.index > '2018-12-21') + (df.index <= '2018-03-20')]
    return df_spring, df_summer, df_autumn, df_winter


def vectorize_wind_speed(tot_df):
    """Converts u and direction into vectorized wind speed u_x and u_y""" 
    # create columns with coordinate velocities output
    tot_df['u_x']=tot_df['u']*np.cos(np.radians(tot_df['direction']))
    tot_df['u_y']=tot_df['u']*np.sin(np.radians(tot_df['direction']))
    # create columns with coordinate velocities input top mast anemometer
    tot_df['u_top_x']=tot_df['u_top']*np.cos(np.radians(tot_df['direction_top']))
    tot_df['u_top_y']=tot_df['u_top']*np.sin(np.radians(tot_df['direction_top']))
    # drop the columns which are not used anymore
    tot_df=tot_df.drop(columns=['u', 'u_top', 'direction', 'direction_top'])
    return tot_df


def split_hs_test(X_te,y_te,hs=np.arange(1.5,22,4)):
    """Creates a list of arrays for every different anemometer on the mast (not the top one)"""
    X_te_hs=[]
    y_te_hs=[]
    for i in hs:
        X_te_hs.append(X_te[X_te[:,1]==i])
        y_te_hs.append(y_te[X_te[:,1]==i])
    return X_te_hs, X_te_hs


def plot_ys(y_pred,y_te,save=False,interval=[100,200],name='graph'):
"""Plot comparison between predicted and real values. Possibility of saving the result"""
    for idx,i in enumerate(y_pred):
        fig=plt.figure(figsize=(16,12))
        plt.subplot(221)
        plt.gca().set_title('u_x')
        plt.plot(i[interval[0]:interval[1],0],'r-',label='u_x_pred')
        plt.plot(y_te[idx][interval[0]:interval[1],0],'b-',label='u_x_test')
        plt.xlabel('t')
        plt.ylabel('u_x')
        plt.legend()
        plt.subplot(222)
        plt.gca().set_title('u_y')
        plt.plot(i[interval[0]:interval[1],1],'r-',label='u_y_pred')
        plt.plot(y_te[idx][interval[0]:interval[1],1],'b-',label='u_y_test')
        plt.xlabel('t')
        plt.ylabel('u_y')
        plt.legend()
        if save:
            savefig(name+str(idx)+'png')
        else
            plt.show()
        
        
        