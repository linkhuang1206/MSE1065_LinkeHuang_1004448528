import pandas as pd
import numpy as np
import numpy.matlib as nm
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings("ignore")
import scipy.io as sio
from scipy.constants import R
import scipy
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib as mpl
import seaborn as sns
import sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, QuantileTransformer, Normalizer, RobustScaler
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, roc_auc_score, plot_roc_curve, accuracy_score, precision_score, recall_score 
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, RandomForestRegressor
from sklearn.svm import SVC
from mp_api.client import MPRester
import itertools as it
from Functions import *


# Section 1.0
# create a dictionary for mp-id and the corresponding list of elements of the material 
def add_chemsys_and_label(df):
    mp_list = list(df['mpid'].unique())
    prop = ['material_id','chemsys']
    formu_elem = {}
    with MPRester("JcjACBt45HOv4RNHua4YOSrdWEFqoIJw") as mpr:
        docs = mpr.summary.search(material_ids=mp_list,fields=prop)
        formu_elem['mp_id'] = [str(getattr(doc, 'material_id')) for doc in docs]
        formu_elem['chemsys'] = [getattr(doc,'chemsys').split('-') for doc in docs]
    formu_elem = pd.DataFrame.from_dict(formu_elem) 
    dic = dict(zip(formu_elem.mp_id, formu_elem.chemsys))

    # create a new column of chemical symbols
    df['chemsys'] = df['mpid'].map(dic)
    # move the chemsys column to the third column

    df.insert(0,'label', 0)
    df['label'][(df['energy']< -0.07)&(df['energy']>-0.47)] = 1
    
    return df

# Section 1.1
def plot_adsorption_energy_distribution(df):
        # visualize the energy distribution of the adsorption sites
        print("type of miller indices:",len(df['miller'].unique()))
        print("number of shift values:",len(df['shift'].unique()))

        plt.figure(figsize=(5,3))
        plt.title('Distribution of adsorption energy')
        plt.hist(df.energy.explode().to_list(),bins=500)
        legend = f"mean: {df.energy.mean():.2f} eV\n"\
                +f"std:      {df.energy.std():.2f}\n"\
                +f"min:    {df.energy.min():.2f} eV\n"\
                +f"max:    {df.energy.max():.2f} eV\n"
        plt.legend([legend])
        plt.axvline(x=-0.47, color='red')
        plt.axvline(x=-0.07, color='red')
        plt.xlabel('adsorption energy(eV)')

        plt.show()
        return None


# Section 1.1
def visualize_chemsys_count(df):
    """
    visualize the number of adsorption sites for each chemical system
    """

    # c# create a materix to store the number of surface for each combination of elements
    elements = df.chemsys.explode().unique()                                  # get the unique elements
    chemsys_count = np.zeros((len(elements),len(elements)))                   # create a matrix to store the number of surfaces for each combination of elements
    for i in range(len(elements)):      
        for j in range(len(elements)):
            chemsys_count[i,j] = len(df[df['chemsys'].apply(lambda x: elements[i] in x and elements[j] in x)]) # count the number of surfaces for each combination of elements
    chemsys_count = pd.DataFrame(chemsys_count, index=elements, columns=elements)
    mask = np.tril(np.ones(chemsys_count.shape)).astype(np.bool)                                               # set the lower triangle to zero, including the diagonal
    chemsys_count = chemsys_count.mask(mask).fillna(0).astype(int)

    # visualize the chemsys_count
    fig, ax = plt.subplots(figsize=(6.5,6.5))
    im = ax.imshow(chemsys_count, cmap='viridis', norm=colors.LogNorm(vmin=1, vmax=620))
    ax.set_xticks(np.arange(len(elements)))                                # add xticks
    ax.set_yticks(np.arange(len(elements)))                                # add yticks
    ax.set_xticklabels(elements, fontsize=10)                              # add the elements to xtick labels
    ax.set_yticklabels(elements, fontsize=10)                              # add the elements to ytick labels
    ax.xaxis.tick_top() # move the xticks to the top
    # plot the edge of the heatmap in black
    ax.set_xticks(np.arange(chemsys_count.shape[1]+1)-.51, minor=True)  
    ax.set_yticks(np.arange(chemsys_count.shape[0]+1)-.53, minor=True)
    ax.grid(which="minor", color="k", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)
    # add a diagonal line
    ax.plot([-0.5, len(elements)-0.5], [-0.5, len(elements)-0.5], color='red', linewidth=1)
    # add colorbar
    cbar = ax.figure.colorbar(im, ax=ax,shrink=0.75)
    cbar.ax.set_ylabel("number of samples", rotation=-90, va="bottom",fontsize=15)
    cbar.set_ticks([1, 10, 100, 200, 300, 400, 500, 600])                  # add ticks to the colorbar
    cbar.set_ticklabels([1, 10, 100, 200, 300, 400, 500, 600])             # add ticks labels to the colorbar

    fig.tight_layout()
    plt.show()
        
    return None

# Section 1.2.1
def visuaize_label_PCA(pos,df, PC=[0,1]):
    plt.figure(figsize=(6,4))
    plt.scatter(pos[:,PC[0]],pos[:,PC[1]],s=30, c = df["label"],  alpha=0.5)
    plt.colorbar()
    plt.xticks(fontsize=13)        # set x sticks font size
    plt.yticks(fontsize=13)        # set y sticks font size
    plt.xlabel('PCA 4',fontsize=13)
    plt.ylabel('PCA 6',fontsize=13)
    plt.show()   

    return None

# Section 1.2.1
def visualize_label_sep_PCA(pos,df, PC=[3,5]):
    L1 = np.where(df['label']==1)[0]
    L0 = np.where(df['label']==0)[0]

    plt.figure(figsize=(8,3))
    #----------------------------------------------------------
    plt.subplot(1,2,1)
    plt.scatter(pos[L1,PC[0]],pos[L1,PC[1]],s=20, alpha=0.5, c = 'gold')
    plt.xlabel('PCA'+str(PC[0]+1))
    plt.xlim((-10,9))
    plt.ylim((-10,10))
    plt.ylabel('PCA'+str(PC[1]+1))
    plt.title('Class 1: near-optimal adsorption sites')
    #----------------------------------------------------------
    plt.subplot(1,2,2)
    plt.scatter(pos[L0,PC[0]],pos[L0,PC[1]],s=20, alpha=0.5, c ='darkmagenta')
    plt.xlabel('PCA'+str(PC[0]+1))
    plt.xlim((-10,9))
    plt.ylim((-10,10))
    plt.ylabel('PCA'+str(PC[1]+1))
    plt.title('Class 0: non-optimal adsorption sites')

    plt.show()
    
    return None



# Section 1.2.1
def plot_adsorption_energy_distribution(df):
        # visualize the energy distribution of the adsorption sites
        print("type of miller indices:",len(df['miller'].unique()))
        print("number of shift values:",len(df['shift'].unique()))

        plt.figure(figsize=(5,3))
        plt.title('Distribution of adsorption energy')
        plt.hist(df.energy.explode().to_list(),bins=500)
        legend = f"mean: {df.energy.mean():.2f} eV\n"\
                +f"std:      {df.energy.std():.2f}\n"\
                +f"min:    {df.energy.min():.2f} eV\n"\
                +f"max:    {df.energy.max():.2f} eV\n"
        plt.legend([legend])
        plt.axvline(x=-0.47, color='red')
        plt.axvline(x=-0.07, color='red')
        plt.xlabel('adsorption energy(eV)')

        plt.show()
        return None

def visualize_near_optimal_fraction(df):
    """
    visualize the adsorption site with near ideal adsorption energy
    """
    # create a materix to store the fraction of near-optimal adsorption energy for each combination of elements
    elements = df.chemsys.explode().unique()                                                    # get the unique elements
    energy_opt_ratio = np.zeros((len(elements),len(elements)))                                  # create a empty matrix
    for i in range(len(elements)):
        for j in range(len(elements)):
            subset = df[df['chemsys'].apply(lambda x: elements[i] in x and elements[j] in x)]   
            numb_of_surface = len(subset)                                                       # get the total number of surface for the combination of elements
            opt_surface = len(subset[(subset['energy']>-0.47) & (subset['energy']<-0.07)])      # get the number of near-optimal adsorption energy
            if numb_of_surface == 0:                            
                energy_opt_ratio[i,j] = np.nan                                                  # if no surface for the combination of elements, set the value to nan
            else:
                energy_opt_ratio[i,j] = opt_surface/numb_of_surface*100                         # calculate the fraction(%) of near-optimal adsorption energy

    energy_opt_ratio = pd.DataFrame(energy_opt_ratio, index=elements, columns=elements)
    mask = np.tril(np.ones(energy_opt_ratio.shape)).astype(np.bool)                             # set the lower triangle to zero, including the diagonal
    energy_opt_ratio = energy_opt_ratio.mask(mask)#.

    # visualize the 2d matrix energy_opt_ratio
    fig, ax = plt.subplots(figsize=(6.5,6.5))
    im = ax.imshow(energy_opt_ratio, cmap='viridis', vmin=0, vmax=100)       
    ax.set_xticks(np.arange(len(elements)))                                  # add xticks
    ax.set_yticks(np.arange(len(elements)))                                  # add yticks
    ax.set_xticklabels(elements, fontsize=10)                                # add the elements to xtick labels     
    ax.set_yticklabels(elements, fontsize=10)                                # add the elements to ytick labels                                      
    ax.xaxis.tick_top()
    # plot colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.75)
    cbar.ax.set_ylabel('Fraction of near-optimal surface(%)', rotation=-90, va="bottom", fontsize=15)
    # add a diagonal line
    ax.plot([-0.5, len(elements)-0.5], [-0.5, len(elements)-0.5], color='red', linewidth=1)
    ax.set_xticks(np.arange(energy_opt_ratio.shape[1]+1)-.51, minor=True)
    ax.set_yticks(np.arange(energy_opt_ratio.shape[0]+1)-.53, minor=True)
    ax.grid(which="minor", color="k", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    current_cmap = plt.cm.get_cmap()
    current_cmap.set_bad(color='grey', alpha=1)
    fig.tight_layout()
    plt.show()

    return None


# Section 1.2.2
def visualize_Ead_PCA(pos,df, PC=[3,5]):
    plt.figure(figsize=(6,4))
    # the range of the energy is capped at -1.04 eV and 0.5 eV, to focus more on the dominant & favaorable range
    norm = colors.Normalize(vmin=-1.04, vmax=0.5)  
    plt.scatter(pos[:,3],pos[:,5],s=40, norm=norm, c = df["energy"],  alpha=0.5)
    # add title to colorbar
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Adsorption Energy (eV)', rotation=270, labelpad=20, fontsize=13)
    plt.xlabel('PCA 4',fontsize=11.5)
    plt.ylabel('PCA 6',fontsize=11.5)

    plt.show()  

    return None 

# Section 1.2.2
def plot_fea_target(df, feature):
    plt.figure(figsize=(24,3))
    norm = colors.Normalize(vmin=-1.04, vmax=0.5)  
    for i in range(len(feature)):
        plt.subplot(1,3,i+1)
        plt.scatter(df[feature[i]],df['energy'],s=30,norm=norm, alpha=0.5,c = df['energy'])
        cbar = plt.colorbar()
        cbar.set_label('Adsorption Energy (eV)', rotation=270, labelpad=20, fontsize=12)
        plt.xticks(np.arange(4, 16, 1))
        plt.xlabel(feature[i], fontsize=12)
        plt.ylabel('Adsorption Energy (eV)', fontsize=12)
    plt.show()
    return None

# Section 2.1
def plot_gridsearch_contour_clf(AUC_mean_list, AUC_mean_train_list,tuned_hyperpar):
    # Calculate the difference between AUC of test and train set
    AUC_diff = np.array(AUC_mean_train_list)-np.array(AUC_mean_list)    

    max_depth= tuned_hyperpar['max_depth']
    n_estimators= tuned_hyperpar['n_estimators'] 

    plt.figure(figsize=(15,4))
    plt.subplot(1,2,1)
    M, N = np.meshgrid(max_depth, n_estimators)
    Z = np.array(AUC_mean_list).reshape(5,4).transpose()
    plt.contourf(M, N, Z, 50, )
    plt.colorbar(label='AUC');
    
    contours = plt.contour(M,N, Z, 3,colors='black')
    plt.clabel(contours, inline=True, fontsize=8)
    plt.title('val AUC')
    plt.xlabel('max_depth')
    plt.ylabel('n_estimators')

    #-----------------------------------------------------
    plt.subplot(1,2,2)
    M, N = np.meshgrid(max_depth, n_estimators)
    Z = np.array(AUC_diff).reshape(5,4).transpose()
    plt.contourf(M,N, Z, 50,)
    plt.colorbar(label='AUC_diff')
    
    contours = plt.contour(M,N, Z, 3, colors='black')
    plt.clabel(contours, inline=True, fontsize=8)
    plt.title('AUC_diff between train and val')
    plt.xlabel('max_depth')
    plt.ylabel('n_estimators')
    plt.show()

    return None


# Section 2.1
def plot_randCV_AUC(RandomCV_AUC_score, RandomCV_AUC_score_train):
    # plot RandomCV_AUC_score and RandomCV_AUC_score_train
    print('Training Avg AUC: ', round(np.mean(RandomCV_AUC_score_train),3))
    print('Validation Avg AUC: ', round(np.mean(RandomCV_AUC_score),3))
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(RandomCV_AUC_score, 'o-', label='train')
    plt.plot(RandomCV_AUC_score_train, 'o-', label='test')
    plt.legend()
    plt.title('AUC for Random-CV ')
    plt.xlabel('group')
    plt.ylabel('AUC')
    plt.ylim(0,1)
    plt.show()
    return None

# Section 2.3
def model_eval_clf(RFC_full, y, y_hat_CV, X_test, y_test, y_hat_full):
    #------------------- ----------------------------------------- CV model performance
    conf_matrix = confusion_matrix(y, y_hat_CV)
    # plot the confusion matrix
    plt.figure(figsize=(4.8,4.25))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='viridis')
    plt.title('RFC_CV Confusion matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    print("RF_CV accuracy: ", round(accuracy_score(y, y_hat_CV),3))
    print("RF_CV precision: ", round(precision_score(y, y_hat_CV),3))
    print("RF_CV recall score:",round(recall_score(y, y_hat_CV),3),'\n')
    #------------------------------------------------------------ full data model performance
    plot_confusion_matrix(RFC_full, X_test, y_test)
    title = 'RFC_full confusion matrix '
    print("RFC_full accuracy: ", round(accuracy_score(y_test, y_hat_full),3))
    print("RFC_full precision: ", round(precision_score(y_test, y_hat_full),3))
    print("RFC_full recall score:",round(recall_score(y_test, y_hat_full),3))
    plt.title(title)
    plt.show()
    return None


# Section 2.3
def visualize_pred_prob_distribution(RFC_full, X_test, pos, RFC):
    proba = RFC.predict_proba(X_test)[:,1]*100
    cmap = plt.cm.viridis_r
    norm = plt.Normalize(0, proba.max())
    plt.scatter(pos[:,3],pos[:,5], s=40, c = cmap(norm(proba)), marker='v')
    # show colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, label='probability of near-ideal adsorption site (%)')
    plt.title('RandomForest classification')
    plt.xlabel('PCA 4')
    plt.ylabel('PCA 6')

    plt.show()
    return None

# Section 2.2
def visualize_misclassified_clf_PCA(RandomCV_RF_misclassified, full_RF_misclassified, pos, y, y_test):
    figure = plt.figure(figsize=(12, 4))
    #------------------- ----------------------------------------- CV model misclassified
    plt.subplot(1,2,1)
    title = 'CV_RFC_misclassified '+str(len(RandomCV_RF_misclassified))
    true_label=y[RandomCV_RF_misclassified]
    plt.scatter(pos[:,3],pos[:,5], s=20, c='lightgrey')
    scatter = plt.scatter(pos[RandomCV_RF_misclassified,3],pos[RandomCV_RF_misclassified,5], s=20, alpha=0.6 ,c=true_label, cmap='viridis_r',marker='v')
    plt.legend(scatter.legend_elements()[0],['0','1'])
    plt.title(title)
    plt.xlim(-8,12)
    plt.ylim(-10,12)
    plt.xlabel('PCA 4')
    plt.ylabel('PCA 6')  
    #------------------------------------------------------------ full data model misclassified
    plt.subplot(1,2,2)
    title = 'full_RFC_misclassified '+str(len(full_RF_misclassified))
    true_label=y_test[full_RF_misclassified]
    plt.scatter(pos[:,3],pos[:,5], s=20, c='lightgrey')
    scatter = plt.scatter(pos[full_RF_misclassified,3],pos[full_RF_misclassified,5], s=20, alpha=0.6, c=true_label, cmap='viridis_r',marker='v')
    plt.legend(scatter.legend_elements()[0],['0','1'])
    plt.title(title)
    plt.xlim(-8,12)
    plt.ylim(-10,12)
    plt.xlabel('PCA 4')
    plt.ylabel('PCA 6')  

    plt.show()
    return None

# Section 2.3
def visualize_misclassified_clf_fea(RandomCV_RF_misclassified, full_RF_misclassified, pos, y, y_test, df):
    figure = plt.figure(figsize=(12, 4))
    a_pool = (pos[:,3],df['0group'],df['1group']) 
    b_pool = (pos[:,5],df['energy'],df['0group'],df['1group'])
    a = a_pool[1]
    b = b_pool[3]
    #------------------- ----------------------------------------- CV model misclassified
    plt.subplot(1,2,1)
    title = 'CV_RFC_misclassified '+str(len(RandomCV_RF_misclassified))
    true_label=y[RandomCV_RF_misclassified]
    plt.scatter(a,b, s=20, c='lightgrey')
    scatter = plt.scatter(a[RandomCV_RF_misclassified],b[RandomCV_RF_misclassified], s=50, alpha=0.3 ,c=true_label, cmap='viridis_r',marker='v')
    plt.legend(scatter.legend_elements()[0],['0','1'])
    plt.title(title)
    plt.xlabel('0group')
    plt.ylabel('1group')  
    #------------------------------------------------------------ full data model misclassified
    plt.subplot(1,2,2)
    title = 'full_RFC_misclassified '+str(len(full_RF_misclassified))
    true_label=y_test[full_RF_misclassified]
    plt.scatter(a,b, s=20, c='lightgrey')
    scatter = plt.scatter(a[full_RF_misclassified],b[full_RF_misclassified], s=50, alpha=0.3, c=true_label, cmap='viridis_r',marker='v')
    plt.legend(scatter.legend_elements()[0],['0','1'])
    plt.title(title)
    plt.xlabel('0group')
    plt.ylabel('1group')  

    plt.show()
    return None

# Section 3.1
def plot_gridsearch_contour_reg(R2_mean_list, R2_mean_train_list, tuned_hyperpar):
    print('Training Avg R2: ', round(np.mean(R2_mean_train_list),3))
    print('Validation Avg R2: ', round(np.mean(R2_mean_list),3))
    R2_diff = np.array(R2_mean_train_list)-np.array(R2_mean_list)

    max_depth= tuned_hyperpar['max_depth'] #[6,8,10,12,14]
    n_estimators= tuned_hyperpar['n_estimators'] #[50,100,150,200,250]


    plt.figure(figsize=(15,4))
    plt.subplot(1,2,1)
    M, N = np.meshgrid(max_depth, n_estimators)
    Z = np.array(R2_mean_list).reshape(5,3).transpose()
    plt.contourf(M, N, Z, 50, )
    plt.colorbar(label='R2');
    
    contours = plt.contour(M,N, Z, 3,colors='black')
    plt.clabel(contours, inline=True, fontsize=8)
    plt.title('val R2')
    plt.xlabel('max_depth')
    plt.ylabel('n_estimators')
    #-----------------------------------------------------
    plt.subplot(1,2,2)
    M, N = np.meshgrid(max_depth, n_estimators)
    Z = np.array(R2_diff).reshape(5,3).transpose()
    plt.contourf(M,N, Z, 50,)
    plt.colorbar(label='R2_diff')
    
    contours = plt.contour(M,N, Z, 3, colors='black')
    plt.clabel(contours, inline=True, fontsize=8)
    plt.title('R2_diff between train and val')
    plt.xlabel('max_depth')
    plt.ylabel('n_estimators')
    plt.show()

    return None


# Section 3.1
def plot_randCV_R2(group_R2, group_R2_train):
    # plot group R2 of random-CV for train and test set in the same figure
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(group_R2_train, 'o-', label='train')
    plt.plot(group_R2, 'o-', label='test')
    plt.legend()
    plt.title('R2 for Random-CV')
    plt.xlabel('group')
    plt.ylabel('R2')
    plt.ylim(0,1)
    plt.show()
    return None

# Section 3.3
def model_eval_reg(y, y_hat, y_hat_randCV, std_randCV, std):
    # plot the parity plot for the random CV and full-data model
    plt.figure(figsize=(10,8))
    ###-----------------parity plot for random CV-----------------###
    plt.subplot(1,2,1)
    plt.scatter(y, y_hat_randCV, zorder=10)
    plt.xlabel('Ead actual [eV]')
    plt.ylabel('Ead predicted [eV]')

    plt.errorbar(y, y_hat_randCV, yerr=std_randCV, fmt='o', ecolor='k', zorder=0, elinewidth=0.5) # plot error bar

    plt.plot([-2,2],[-2,2], c='r', label='$R^{2}$='+ str(round(r2_score(y, y_hat_randCV),3)), zorder=20) # plot a line with the slope of 1

    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')
    ###-----------------parity plot for full dataset-----------------###
    plt.subplot(1,2,2)
    plt.scatter(y, y_hat, zorder=10)
    plt.xlabel('Ead actual [eV]')
    plt.ylabel('Ead predicted [eV]')

    plt.errorbar(y, y_hat, yerr=std, fmt='o', ecolor='k', zorder=0, elinewidth=0.5) # plot error bar

    plt.plot([-2,2],[-2,2], c='r', label='$R^{2}$='+ str(round(r2_score(y, y_hat),3)), zorder=20) # plot a line with the slope of 1

    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')

    return None


# Section 4.1.1
def visualize_chemsys_target_PCA(df, pos):
    df3 = df.copy()
    df3.sort_values(by='chemsys', inplace=True)

    # convert chemsys to str
    chemsys_str = df3['chemsys'].apply(lambda x: ''.join(x))
    chemsys_str
    plt.figure(figsize=(15,4))
    #-------------------------------------------------------------
    plt.subplot(1,2,1)
    PC = [0,1]
    #facotrize the mpid
    plt.scatter(pos[df3.index,PC[0]],pos[df3.index,PC[1]],s=20, c = pd.factorize(chemsys_str)[0],  alpha=0.5, cmap='hsv')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('chemsys index', rotation=270, labelpad=20)
    plt.xlabel('PCA {}'.format(PC[0]+1))
    plt.ylabel('PCA {}'.format(PC[1]+1))

    #---------------------------------------------------------------
    plt.subplot(1,2,2)
    # the range of the energy is capped at -1.04 eV and 0.5 eV, to focus more on the dominant & favaorable range
    norm = colors.Normalize(vmin=-1.04, vmax=0.5)  
    plt.scatter(pos[:,PC[0]],pos[:,PC[1]],s=40, norm=norm, c = df["energy"],  alpha=0.5)
    # add title to the colorbar
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Ead [eV]', rotation=270, labelpad=20)
    plt.xlabel('PCA {}'.format(PC[0]+1))
    plt.ylabel('PCA {}'.format(PC[1]+1))

    plt.show()   
    return None



# Section 4.4.1
def plot_AUC_vs_frac(frac_true, AUC_list, AUC_list_rd):
    mean = np.array(AUC_list_rd).reshape(5,10).mean(axis=0)
    std = np.array(AUC_list_rd).reshape(5,10).std(axis=0)

    plt.figure(figsize=(6,4))
    # add error bar of std
    plt.errorbar(frac_true, mean, yerr=std, color='mediumspringgreen', label='random sampling')
    plt.scatter(frac_true, mean, color='mediumspringgreen')
    plt.plot(frac_true, AUC_list, color='lightseagreen',label='systematic sampling')
    plt.scatter(frac_true, AUC_list, color='lightseagreen')
    plt.legend(fontsize=12)
    plt.xlabel('fraction of data used for training', fontsize=12)
    plt.ylabel('AUC', fontsize=12)
    plt.axvline(x=0.16, color='teal', linestyle='--')
    plt.axhline(y=0.938*0.9, color='teal', linestyle='--', label='90% of oringinal performance')
    plt.show()

    return None

# Section 4.4.2
def plot_mae_vs_frac(frac_true, mae_list, mae_list_rd):
    mean2 = np.array(mae_list_rd).reshape(5,10).mean(axis=0)
    std2 = np.array(mae_list_rd).reshape(5,10).std(axis=0)

    plt.figure(figsize=(6,4))
    plt.scatter(frac_true, mean2, color='mediumspringgreen')
    plt.errorbar(frac_true, mean2, yerr=std2, color='mediumspringgreen', label='random sampling')

    plt.plot(frac_true, mae_list, color='lightseagreen', label='systematic sampling')
    plt.scatter(frac_true, mae_list, color='lightseagreen')

    plt.axvline(x=0.79, color='teal', linestyle='--')
    plt.axhline(y=0.086*1.1, color='teal', linestyle='--', label='90% of oringinal performance')
    plt.legend(fontsize=12)
    plt.xlabel('Fraction of data used for training', fontsize=12)
    plt.ylabel('MAE', fontsize=12)
    plt.show()

    return None

# for i in range(5):
#     for frac in tqdm(frac_true):
#         train_index =  np.random.choice(np.arange(0,len(X)), int(len(X)*frac), replace=False)
#         print(i, "idx",np.sort(train_index)[:3])
#         X_train, y_train= scaler.fit_transform(X[train_index]), y[train_index]
#         print(i, 'train', X_train[:3,0])
#         # predict target value for test set
#         print(i, "test", scaler.transform(X)[:3,0])