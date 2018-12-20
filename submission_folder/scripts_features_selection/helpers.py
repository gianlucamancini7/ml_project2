
# coding: utf-8
import numpy as np
import pandas as pd
import matplotlib
#matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split


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

def compute_mse_rsq(y_te_hs, y_pred_hs):
    
    '''Compute the mse and r square of the prediction with several outputs, 
    in the format of a list seperated by height'''
    
    mse = np.zeros([y_te_hs[0].shape[1],len(y_te_hs)])
    rsq = np.zeros([y_te_hs[0].shape[1],len(y_te_hs)])
    for i in range(mse.shape[0]):
        for j in range(mse.shape[1]):
            mse[i,j] = np.mean(np.square(y_te_hs[j][:,i]-y_pred_hs[j][:,i]))
            rsq[i,j] = 1-mse[i,j]/(np.std(y_te_hs[j][:,i]))**2
    return mse, rsq

#------------------------------------------------------------------------------------##
#-------------------------- Splitting the data --------------------------------------##
#------------------------------------------------------------------------------------##

def split_data(X, ratio, seed=1):
    # set seed
    np.random.seed(seed)
    # generate random indices
    num_row = X.shape[0]
    indices = np.random.permutation(num_row)
    index_split = int(np.floor(ratio * num_row))
    index_tr = indices[: index_split]
    index_te = indices[index_split:]
    # create split
    x_tr = X.iloc[index_tr]
    x_te = X.iloc[index_te]
    return x_tr, x_te
        

def consistency_splitting(train_dim, test_dim ,validate_dim):
    if train_dim+test_dim+validate_dim != 1:
        raise ValueError('SplittingConsistecyError: Check splitting of the dataframe')
    else:
        print('Consistent Split')
        print('')
    return
      
def split_train_evaluation_test(x,y,ratio_train,ratio_validation,ratio_test,random_state = 50):
    
    '''Split the data into train, evaluation, test set, 
      according to the global ratios.'''
    
    if ratio_train+ratio_validation+ratio_test != 1:
        raise ValueError('SplittingConsistecyError: Check splitting of the dataframe')
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=ratio_test, random_state = random_state)
    x_tr, x_ev, y_tr, y_ev = train_test_split(x_tr, y_tr, test_size = ratio_validation/(1.0 - ratio_test), random_state = random_state)
    return x_tr,y_tr,x_ev,y_ev,x_te,y_te

def split_hs_test(X_te,y_te,h_pos=1,hs=np.arange(1.5,22,4)):
    """Creates a list of arrays for every different anemometer on the mast (not the top one)"""
    X_te_hs=[]
    y_te_hs=[]
    for i in hs:
        X_te_hs.append(X_te[X_te[:,h_pos]==i])
        y_te_hs.append(y_te[X_te[:,h_pos]==i])
    return X_te_hs, y_te_hs

def season_splitter(df):
    """Returs 4 different datasets for each astronomical season in year 2018""" 
    df.index = pd.to_datetime(df.index)
    df_spring = df[(df.index > '2018-03-20') & (df.index <= '2018-06-20')]
    df_summer = df[(df.index > '2018-06-21') & (df.index <= '2018-09-21')]
    df_autumn = df[(df.index > '2018-09-21') & (df.index <= '2018-12-21')]
    df_winter = df[(df.index > '2018-12-21') + (df.index <= '2018-03-20')]
    return df_spring, df_summer, df_autumn, df_winter

#------------------------------------------------------------------------------------##
#------------------------ Plot the predictions --------------------------------------##
#------------------------------------------------------------------------------------##

def plot_ys(y_pred,y_te,path,save=False,interval=[100,200],name='graph'):
    
    """Plot comparison between predicted and real values. Possibility of saving the result"""

    for idx,i in enumerate(y_pred):
        plt.figure(figsize=(16,7), dpi=300)
        #plt.gca().set_title('Anemometer '+str(idx+1)+' - '+'%s'%name[:-1])
        plt.subplot(121)
        plt.plot(i[interval[0]:interval[1],0],'r-',label='$u_x$ pred')
        plt.plot(y_te[idx][interval[0]:interval[1],0],'b-',label='$u_x$ test')
        plt.grid()
        plt.xlabel('timestamp')
        plt.ylabel('$u_x$ [m/s]')
        plt.legend()
        plt.subplot(122)
        #plt.gca().set_title('$u_y$ %s'%name[:-1])
        plt.plot(i[interval[0]:interval[1],1],'r-',label='$u_y$ pred')
        plt.plot(y_te[idx][interval[0]:interval[1],1],'b-',label='$u_y$ test')
        plt.grid()
        plt.xlabel('timestamp')
        plt.ylabel('$u_y$ [m/s]')
        plt.legend()
        plt.suptitle('Anemometer '+str(idx+1)+' - '+'%s'%name)
        if save:
            plt.savefig(path+name+'anem'+str(idx+1)+'.eps',dpi=300)
            plt.close()
        else:
            plt.show()
    return

def plot_ys_single(y_pred,y_te,path,save=False,interval=[100,200],name='graph'):

    """Plot comparison between predicted and real value. Possibility of saving the result"""
    for idx,i in enumerate(y_pred):
        plt.figure(figsize=(8,7), dpi=300)
        plt.plot(i[interval[0]:interval[1]],'r-',label='$u$ pred')
        plt.plot(y_te[idx][interval[0]:interval[1]],'b-',label='$u$ test')
        plt.grid()
        plt.xlabel('timestamp')
        plt.ylabel('$u$ [m/s]')
        plt.legend()
        if save:
            plt.savefig(path+name+'anem'+str(idx+1)+'.eps',dpi=300)
            plt.close()
        else:
            plt.show()
            plt.close() 
    return

#------------------------------------------------------------------------------------##
#------------------ Random forest feature selection and regression ------------------##
#------------------------------------------------------------------------------------##

def bins_proposal(y_continue,precision = 0.1):
    
    """
    For random forest discretizing the output, 
    compute the appropriate bins according to the desired precision
    """
    
    bins = []
    for i in range(y_continue.shape[1]):
        bins.append(int(math.ceil((np.max(y_continue[:,i])-np.min(y_continue[:,i]))/precision)))
    return bins

def extract_important_features(feat_labels,importances,threshold):
    
    """ Extract the list of the most important features according to the threshold """
    
    if threshold > 1 or threshold < 0:
        raise ValueError('Threshold should between 0 and 1')
    indices = np.argsort(importances)[::-1]
    importance_accum = 0
    important_features = []
    for f in range(len(importances)):
        if importance_accum < threshold:
            importance_accum = importance_accum + importances[indices[f]]
            important_features.append(feat_labels[indices[f]])
    return important_features

def write_feature_importance(filetxt, importances, feat_labels, important_features):
    
    """ Plot the histogram indicating the relative importance of each feature """
    
    indices = np.argsort(importances)[::-1]
    for f in range(len(importances)):
        filetxt.write("%2d) %-*s %f \n" % (f + 1, 50, feat_labels[indices[f]], importances[indices[f]]))
    filetxt.write("\n The top 80% important features are: \n")
    for i in range(len(important_features)):
        filetxt.write("%s \n" % important_features[i])
    filetxt.write("%i features on %i" % (len(important_features), len(importances)))
    return

def plot_feature_importance(path, feature_importance, name = 'feature_importance', save = 'True'):
    
    """ Plot the histogram indicating the relative importance of each feature """
    #feat_labels = feature_importance.columns.values.reshape(1,-1)
    feature_importance = feature_importance.values
    plt.figure(figsize=(16,8), dpi=300)
    plt.suptitle('Feature importances')
    for i, season in enumerate(['Spring','Summer','Autumn','Winter']):
        importances = feature_importance[i,:]
        ax = plt.subplot(221+i)
        plt.gca().set_title(season)
        plt.bar(range(len(importances)), importances, color="b")#, align="center")
        plt.xticks(np.arange(len(importances)), fontsize=7)
        plt.subplots_adjust(bottom=0.45)
        plt.tight_layout()
        plt.xlabel('Feature index')
        plt.ylabel('Relative feature importance')
        plt.grid()
    if save:
        plt.savefig(path+name+'.eps',dpi=300)
    plt.show()
    
    return
        
def write_rf_prediction(OUTPUT_FOLDER,bins,mse,rsq,season):
    
    """ 
    Write the performance of the rf prediction
    Output: 3 files, reapectively storing the bins, mses, rsquared. 
    In each file the 4 rows are representive of 4 seasons. 
    In mses and rsquared, the columns are for each anemometers.
    """
    
    df = pd.DataFrame(bins).T
    df.join(pd.DataFrame({len(bins):[season]}))
    df.to_csv(OUTPUT_FOLDER + "/bins_proposal.txt", mode='a',header=None)
    
    mse = np.mean(mse,axis = 0)
    df = pd.DataFrame(mse).T
    df.join(pd.DataFrame({6:[season]}))
    df.to_csv(OUTPUT_FOLDER + "/mses_u_seasons.txt", mode='a',header=None)
    
    rsq = np.mean(rsq,axis = 0)
    df = pd.DataFrame(rsq).T
    df.join(pd.DataFrame({6:[season]}))
    df.to_csv(OUTPUT_FOLDER + "/rsquared_u_seasons.txt", mode='a',header=None)
    return

#------------------------------------------------------------------------------------##
#---------------------------- Global Result Analysis ------------------------------- ##
#------------------------------------------------------------------------------------##

def magnitude_avg(arr):
    
    """ Compute average of magnitude depending on array shape """
    
    if len(arr.shape)==1:
        return arr.mean()
    else:
        return np.sqrt(np.sum(np.square(arr),axis=1)).mean()
    return
 
def plot_height_average(u_compact, u_true, mse_compact, methods, path = '', name="compare_height_average", save=True): 
    
    '''
    To obtain the height profile: 
    The evaluation of the wind speed amplitude with the height of anemometers, 
    averged for each season
    ''' 
    ## u_average, 4 lines for 4 seasons. Increasing height
    ## mse, 4 lines for 4 seasons, Increasing height
    colors = ['brown','orange','c','g']
    height = np.arange(1.5,22,4)
    plt.figure(figsize=(16,12), dpi=300)
    for i, season in enumerate(['Spring','Summer','Autumn','Winter']):
        plt.subplot(221+i)
        plt.gca().set_title(season)
        plt.plot(u_true[i,:], height, label='true', color = 'r')
        for j,method in enumerate(methods):
            plt.errorbar(u_compact[j][i,:], height, xerr = (np.sqrt(mse_compact[j][i,:])),label=(method + 'prediction'), linestyle='--', marker='o',color = colors[j])
        plt.xlabel('u [m/s]')
        plt.ylabel('height [m]')
        plt.legend()
        plt.grid()
    if save:
        plt.savefig(path+name+'.eps',dpi=300)
    plt.show()
    return
    
def load_comparable_reault(INPUT_FOLDER,methods):
    '''
    To load the performance(mse, rsqared) of the 4 regression methods into a compact form
    '''
    u_compact = []
    mse_compact = []
    rsq_compact = []
    u_true = pd.read_table(INPUT_FOLDER + 'baseline'+ "/magnitude_average_true.txt", sep=",",header = None)
    u_true = np.array(u_true)
    u_true = u_true[:,1:7]
    u_true = u_true.astype(np.float)
    for method in methods:
        '''The magnitude_average'''
        u = pd.read_table(INPUT_FOLDER + method + "/magnitude_average_pred.txt", sep=",",header = None)
        u = np.array(u)
        u = u[:,1:7]
        u = u.astype(np.float)
        u_compact.append(u)
        '''The mse and r_squared'''
        mse = pd.read_table(INPUT_FOLDER + method + "/mses_u_seasons.txt", sep=",",header = None)
        mse = np.array(mse)
        mse = mse[:,1:7]
        mse = mse.astype(np.float)
        mse_compact.append(mse)
        rsq = pd.read_table(INPUT_FOLDER + method + "/rsquared_u_seasons.txt", sep=",",header = None)
        rsq = np.array(rsq)
        rsq = rsq[:,1:7]
        rsq = rsq.astype(np.float)
        rsq_compact.append(rsq)
    return u_true,u_compact,mse_compact,rsq_compact

def plot_method_comparasion(mse_compact, methods, ylabel = 'mse',path = '', name = 'compare_mse', save = True):
    '''
    To visualize the performance(mse, rsquared) of each method, being separated for each season.
    ''' 
    ## mse_compact: 3*(4*6) matrix, 3 methods, 4 seasons, 6 anometers
    colors = ['brown','orange','c','g']  # c
    plt.figure(figsize=(16,12), dpi=300)
    plt.suptitle('For $u_x$ and $u_y$')
    anem = np.arange(6) + 1
    for i, season in enumerate(['Spring','Summer','Autumn','Winter']):
        ax = plt.subplot(221+i)
        plt.gca().set_title(season)
        for j,method in enumerate(methods):
            plt.plot(anem, mse_compact[j][i,:], label=method, linestyle='--', marker='o', color = colors[j]) 
        plt.xlabel('Anemometer Number')
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid()
        if name == 'compare_rsq': 
            ax.set_ylim([-1, 1.5])
        else:
            ax.set_ylim([np.min(mse_compact)-0.5, np.max(mse_compact)+0.5])
    if save:
        plt.savefig(path+name+'.eps',dpi=300)
    plt.show()
    return
#------------------------------------------------------------------------------------##
#----------------------------------------------------------------------------------- ##
#------------------------------------------------------------------------------------##
    
