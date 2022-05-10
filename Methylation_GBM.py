#!/usr/bin/env python
# coding: utf-8

# In[29]:


# Library Import
import pandas as pd
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import numpy as np
from sklearn.decomposition import PCA
from sklearn import linear_model
from sklearn import svm
from sklearn.model_selection import train_test_split
from scipy.cluster.hierarchy import dendrogram
from sklearn import cluster
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import KMeans
from matplotlib import colors
import matplotlib as mpl
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from scipy import stats
import collections


# In[95]:


# Define a path for iterative import of TCGA dataset (GBM)
mypath="C:/Graduate school/Spring22/20.440/gdc_download_20220405_034243.083206"


# In[96]:


# Read in all beta value files from a directory (TCGA-GBM) 
listOfFiles = list()
for (dirpath, dirnames, filenames) in os.walk(mypath):
    listOfFiles += [os.path.join(dirpath, file) for file in filenames if 'level3betas' in file]


# In[97]:


# Process and concatenate all GBM table
table= pd.read_csv(listOfFiles[0], sep="\t",engine='python',header=None)
for i in range(1,len(listOfFiles)):
    element=pd.read_csv(listOfFiles[i], sep="\t",engine='python',header=None)
    table[str(i+1)]=element.iloc[:,1]


# In[92]:


control_table=pd.read_csv('GSE15745-GPL8490_series_matrix.txt',sep='\t',skiprows=97, header=None, engine='python')
control_table=control_table.iloc[0:len(control_table)-1,:]
print(control_table)


# In[6]:


test_table=pd.read_csv('GSE60274_series_matrix.txt',sep='\t',skiprows=76, header=0, engine='python')
test_table=test_table.iloc[0:len(test_table)-1,:]
display(test_table)


# In[11]:


# Index the matrix by the methylation CpG site ID
table_reindex=table.set_index(0)
control_table_reindex=control_table.set_index(0)
test_table_reindex=test_table.set_index('ID_REF')


# In[12]:


# Drop NA values of dataset and keep the common CpG sites shared between all datasets
GBM_final=table_reindex.sort_index()
control_final=control_table_reindex.sort_index()
test_final=test_table_reindex.sort_index()
GBM_nona=GBM_final.dropna()
control_nona=control_final.dropna()
test_nona=test_final.dropna()
common_index=list(set(GBM_nona.index)&set(control_nona.index)&set(test_nona.index))
GBM_final.to_csv('GBM_table.csv')
control_final.to_csv('control_table.csv')
test_final.to_csv('test_table.csv')


# In[98]:


# Take the common CpG sites of all datasets
GBM_total_nona=GBM_nona.loc[common_index,:]
control_total_nona=control_nona.loc[common_index,:]
test_total_nona=test_nona.loc[common_index,:]


# In[99]:


# Rename samples by its label (identity)
GBM_name=dict()
for i in range(0,len(GBM_total_nona.columns)):
    GBM_name[GBM_total_nona.columns[i]]='GBM'+str(i+1)
control_name=dict()
for i in range(0,len(control_total_nona.columns)):
    control_name[control_total_nona.columns[i]]='Control'+str(i+1)
test_name=dict()
# test_total_nona=test_total_nona.drop(['GSM1469038','GSM1469039','GSM1469040','GSM1469041'],axis=1)
for i in range(0,len(test_total_nona.columns)):
    if test_total_nona.columns[i] in ['GSM1469033','GSM1469034','GSM1469035','GSM1469036','GSM1469037']:
        test_name[test_total_nona.columns[i]]='Control'+str(i+1+len(control_total_nona.columns))
    else:
        test_name[test_total_nona.columns[i]]='GBM'+str(i+1+len(GBM_total_nona.columns))


# In[101]:


# Rename the matrix
GBM_renamed=GBM_total_nona.rename(columns=GBM_name)
control_renamed=control_total_nona.rename(columns=control_name)
test_renamed=test_total_nona.rename(columns=test_name)


# In[102]:


# Output final processed data. This can be directly loaded in the future
control_renamed.to_csv('control_table_noNa_final.csv')
GBM_renamed.to_csv('GBM_table_noNa_final.csv')
test_renamed.to_csv('test_table_noNa_final.csv')


# # This section runs the code for processed data matrix

# In[9]:


#Load data
# Control: GSE15745, GBM: TCGA-GBM, Test: GSE60274
control_renamed=pd.read_csv('control_table_noNa_final.csv',index_col=0,engine='python')
GBM_renamed=pd.read_csv('GBM_table_noNa_final.csv',index_col=0,engine='python')
test_renamed=pd.read_csv('test_table_noNa_final.csv',index_col=0,engine='python')


# In[128]:


# Plot global distribution of methylation for control and GBM samples
plt.hist(control_renamed.mean(axis=1),bins=20,color='#6E8B3D')
plt.xlabel('Mean beta value')
plt.ylim(0, 9000)
plt.ylabel('Count')
plt.show()


# In[11]:


# Agglomerative clustering of samples, labels shown as bar below
# Input: matrix containing beta values of all samples and their labels
# Output: dendrogram with labeled bar
def plot_dendrogram(feature_matrix,target):
    distance=pdist(feature_matrix,'euclidean')
    Z=linkage(distance,'average')
    fig=plt.figure(figsize = (15,6))
    dn=dendrogram(Z,labels=target,leaf_rotation=90)
    plt.show()
    fig, ax = plt.subplots()
    fig.set_figheight(2)
    fig.set_figwidth(15)
    if len(set(target))==2:
        cmap = mpl.colors.ListedColormap(['#6E8B3D','#CD3333'])
        bounds = [-0.5,0.5,1.5]
    else:
        cmap = mpl.colors.ListedColormap(['#6E8B3D','#CD3333','#53868B'])
        bounds = [-0.5,0.5,1.5,2.5]
    
    points = [dn['ivl'] for i in range(100)]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    ax.imshow(points, cmap=cmap, norm=norm)
    plt.show()
    return(dn['ivl'])


# In[12]:


# Create joint table of all dataset and training dataset
total_table1=control_renamed.join(GBM_renamed)
total_table2=total_table1.join(test_renamed)


# In[13]:


# Create labels for the matrix and combine them together
GBM_label=[0]*len(total_table2.columns)
GBM_label2=[0]*len(total_table1.columns)
GBM_label_test=[0]*len(test_renamed.columns)
matrix_label=[0]*len(control_renamed.columns)+[1]*len(GBM_renamed.columns)+[2]*len(test_renamed.columns)
for i in range(0,len(total_table2.columns)):
    if 'GBM' in total_table2.columns[i]:
        GBM_label[i]=1
    elif 'Control' in total_table2.columns[i]:
        GBM_label[i]=0
for i in range(0,len(total_table1.columns)):
    if 'GBM' in total_table1.columns[i]:
        GBM_label2[i]=1
    elif 'Control' in total_table1.columns[i]:
        GBM_label2[i]=0
for i in range(0,len(test_renamed.columns)):
    if 'GBM' in test_renamed.columns[i]:
        GBM_label_test[i]=1
    elif 'Control' in test_renamed.columns[i]:
        GBM_label_test[i]=0


# In[14]:


# Agglomerative clustering of samples by label
# Red: GBM, Green: Control
GBM_culster=plot_dendrogram(total_table2.T,GBM_label)


# In[15]:


# Clustering of samples and label by datasets
# Red: GBM, green:Control, Blue: test
# Samples don't cluster by dataset (between sample variance is not too big)
Sample_cluster=plot_dendrogram(total_table2.T,matrix_label)


# In[16]:


# PCA analysis of data
# Standardize the values for each selected feature and perform PCA
def PCAanalysis(x):
    X=preprocessing.StandardScaler().fit_transform(x)
    pca = PCA()
    pca.fit(X)
    print(pca.n_components_)
    principalComponents = pca.fit_transform(X)
    principalDf = pd.DataFrame(data = principalComponents
             , columns = ['PC'+str(i) for i in range(1,pca.n_components_+1)])
    loading_matrix=pca.components_
    return(principalDf,loading_matrix,pca.explained_variance_ratio_)


# In[133]:


# This block adds the name of donors on each point
# Input result is the 'y variable' with donors as row labels
# If uncomment block, plot feature loadings on current x and y axis PCs
def plotPCA(principalDf,selected_feature,loading_matrix,explained_variance):
    for i in range(0,len(principalDf.columns.values)-2):
        for j in range(i+1, len(principalDf.columns.values)-1):
            fig = plt.figure(figsize = (8,8))
            ax = fig.add_subplot(1,1,1) 
            ax.set_xlabel('PC'+str(i+1)+'('+str(round(explained_variance[i]*100,2))+'%)', fontsize = 20)
            ax.set_ylabel('PC'+str(j+1)+'('+str(round(explained_variance[j]*100,2))+'%)', fontsize = 20)
            principalDf_sort=principalDf.sort_values('target',axis=0,ascending=True)
            targets = principalDf_sort['target'].values
            color_list={0:'#6E8B3D',1:'#CD3333'}
            colors = [color_list[c] for c in targets]
            sc = ax.scatter(principalDf_sort.loc[:,'PC'+str(i+1)], principalDf_sort.loc[:,'PC'+str(j+1)], c=colors)
            plt.savefig(str(i)+str(j)+'.jpg')
            plt.show()


# In[18]:


# Run PCA analysis on all features in the training dataset
[principalDf2, loading_matrix,explained_variance]=PCAanalysis(total_table1.T)


# In[19]:


# This is the master run script for visualizing PCA analysis
# Calls PCAanalysis and PlotPCA functions
# feature_matrix: a matrix of physical features to run PCA on, result: label values, name: name of the label used
def PCAmasterrun(principalDf2, loading_matrix,explained_variance,feature_matrix,result):
#     [principalDf2, loading_matrix,explained_variance]=PCAanalysis(feature_matrix)
    plt.figure(figsize=(8,4))
    plt.bar(['PC'+ str(i) for i in range(1,7)],explained_variance[0:6],color='#5F9EA0')
    plt.ylabel('Explained Variance')
    plt.show()
    for i in range(0,len(explained_variance)):
        if sum(explained_variance[0:i+1])>0.8:
            count=i
            break
    principalDf_trimmed=principalDf2.iloc[:,0:6]
    principalDf_trimmed['target']=np.array(result)
    principalDf_trimmed.index=feature_matrix.index
    loading_matrix_trimmed=loading_matrix[:,0:6]
    plotPCA(principalDf_trimmed,feature_matrix.columns.values,loading_matrix_trimmed,explained_variance)


# In[134]:


# Plot score plots for all feature PCA in the training set
PCAmasterrun(principalDf2, loading_matrix,explained_variance,total_table1.T,GBM_label2)


# In[150]:


# Use the loadings from all feature PCA in training set to plot score plot of test set
transformed_test_matrix=np.matmul(loading_matrix,test_renamed)
test_matrix=pd.DataFrame()
for i in range(1,6):
    test_matrix['PC'+str(i)]=transformed_test_matrix.iloc[i-1,:]
targets = GBM_label_test
test_matrix['target']= targets
test_sort=test_matrix.sort_values('target',axis=0,ascending=True)
for i in range(0,len(test_sort.columns.values)-2):
        for j in range(i+1, len(test_sort.columns.values)-1):
            fig = plt.figure(figsize = (8,8))
            ax = fig.add_subplot(1,1,1) 
            ax.set_xlabel('PC'+str(i+1)+'('+str(round(explained_variance[i]*100,2))+'%)', fontsize = 20)
            ax.set_ylabel('PC'+str(j+1)+'('+str(round(explained_variance[j]*100,2))+'%)', fontsize = 20)
            color_list={0:'#6E8B3D',1:'#CD3333'}
            colors = [color_list[c] for c in targets]
            sc = ax.scatter(test_sort.loc[:,'PC'+str(i+1)], test_sort.loc[:,'PC'+str(j+1)], c=colors)
            plt.savefig('test_reduced'+str(i)+str(j)+'.jpg')
            plt.show()


# In[104]:


# Loadings of all features on PC1 and PC2
plt.figure(figsize=(8,4))
plt.hist(loading_matrix[0,:],color='#5F9EA0',bins=20)
plt.ylabel('Frequency')
plt.xlabel('Loadings')
plt.show()
plt.figure(figsize=(8,4))
plt.hist(loading_matrix[1,:],color='#5F9EA0',bins=20)
plt.ylabel('Frequency')
plt.xlabel('Loadings')
plt.show()


# In[41]:


# Violin plot of PC values for GBM and control from all feature analysis
# t-test of difference
# Optional
principalDf2['target']=GBM_label2
ax = sns.violinplot(x='target', y='PC2',
                    data=principalDf2, palette=['#6E8B3D','#CD3333'])
ax.set_xticklabels(['Control','GBM'])
ax.set_xlabel('')
ax.set_ylabel('Values on PC2')
res = stats.ttest_ind(principalDf2['PC2'][0:295], principalDf2['PC2'][295:801], 
                      equal_var=True)

display(res)


# In[23]:


# Lasso regression to eliminate features
def ElasticnetElim(a, x, y):
    clf = linear_model.Lasso(alpha=a)
    clf.fit(x,y)
    selected_feature=[]
    for i in range(0,len(clf.coef_)):
        if clf.coef_[i]!=0:
            selected_feature.append(x.columns.values[i])
    return(selected_feature)


# In[32]:


# Eliminate features using different a values
table_trans=total_table1.T
selected_feature_list=[]
for i in range(10):
    a=(i+1)/100
    selected_feature=ElasticnetElim(a,table_trans,GBM_label2)
    sns.heatmap(table_trans[selected_feature].corr(),cmap="YlGnBu")
    selected_feature_list=selected_feature_list+(selected_feature)
    plt.title('a='+str(a))
    plt.show()


# In[38]:


# Frequency of selected features from a grid search using multiple a values
counter=collections.Counter(selected_feature_list)
plt.figure(figsize=(8,4))
plt.bar(counter.keys(),counter.values(),color='#5F9EA0')
plt.ylabel('Frequency',size=12)
plt.xticks(rotation = 90,size=12)
plt.yticks(size=12)
plt.show()
selected_feature=[list(counter.keys())[i]  for i in range(0,len(counter)) if counter[list(counter.keys())[i]]>2]


# In[146]:


# Violin plot of individual features 
for i in selected_feature:
    table_trans['target']=GBM_label2
    ax = sns.violinplot(x='target', y=i,
                        data=table_trans, palette=['#6E8B3D','#CD3333'])
    ax.set_xticklabels(['Control','GBM'])
    ax.set_xlabel('')
    ax.set_ylabel('Values on '+str(i))
    plt.savefig('test_'+str(i))
    plt.show()
    # Non parametric statistical test
    res = stats.mannwhitneyu(table_trans[i][0:295], table_trans[i][295:801])
    display(res)


# In[39]:


# Features selected from Elastic net elimination where features occured more than 2 times in grid search
print(selected_feature)


# In[136]:


# Run PCA analysis on selected features in the training dataset
[principalDf3, loading_matrix2,explained_variance2]=PCAanalysis(table_trans[selected_feature])
PCAmasterrun(principalDf3, loading_matrix2,explained_variance2,table_trans[selected_feature],GBM_label2)


# In[57]:


# Loading matrix from selected feature PCA
loading_PC=pd.DataFrame(loading_matrix2,index=None, columns=table_trans[selected_feature].columns[:len(table_trans[selected_feature].columns)])


# In[58]:


#Feature loadings on PC1 and PC2
plt.figure(figsize=(8,4))
plt.bar(loading_PC.columns.values,loading_PC.iloc[0,:],color='#5F9EA0')
print(loading_PC.iloc[:,0])
plt.xticks(rotation = 90)
plt.title('Loading on PC1')
plt.show()
plt.figure(figsize=(8,4))
plt.bar(loading_PC.columns.values,loading_PC.iloc[1,:],color='#5F9EA0')
plt.xticks(rotation = 90)
plt.title('Loading on PC2')
plt.show()


# In[143]:


# Support vector machine to separate data
# Out_train and Out_test are from the test dataset (not invovled in training or picking selected feature)
def linearsvm(x,y,fold,out_train,out_test,selected_feature):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=1/fold, random_state=0)
    model = svm.SVC(kernel="linear", probability=True)
    model.fit(X_train, y_train)
#     print(clf.predict(X_test))
    print(model.score(X_test,y_test))
    decision_scores = model.decision_function(X_test)
    fpr, tpr, thres = roc_curve(y_test, decision_scores)
    preds = model.predict(X_test)
    print('AUC: {:.3f}'.format(metrics.roc_auc_score(y_test, preds)))
    print("accuracy: ", metrics.accuracy_score(y_test, preds))
    print("precision: ", metrics.precision_score(y_test, preds)) 
    print("recall: ", metrics.recall_score(y_test, preds))
    print("f1: ", metrics.f1_score(y_test, preds))
    # roc curve
    plt.plot(fpr, tpr, '#5F9EA0', label='Linear SVM')
    plt.plot([0,1],[0,1], "k--", label='Random Guess')
    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.legend(loc="best")
    plt.title("ROC curve")
    plt.show()
    # Get the model parameters
    print('Coefficients of the SVM model')
    print(model.coef_)
    plt.bar(selected_feature,model.coef_[0],color='#5F9EA0')
    plt.xticks(rotation = 90)
    plt.show()
    print('Out of dataset prediction')
    print(model.score(out_train,out_test))
    decision_scores = model.decision_function(out_train)
    fpr, tpr, thres = roc_curve(out_test, decision_scores)
    preds = model.predict(out_train)
    print(preds)
    print('AUC: {:.3f}'.format(metrics.roc_auc_score(out_test, preds)))
    print("accuracy: ", metrics.accuracy_score(out_test, preds))
    print("precision: ", metrics.precision_score(out_test, preds)) 
    print("recall: ", metrics.recall_score(out_test, preds))
    print("f1: ", metrics.f1_score(out_test, preds))
    # roc curve
    plt.plot(fpr, tpr, '#5F9EA0', label='Linear SVM')
    plt.plot([0,1],[0,1], "k--", label='Random Guess')
    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.legend(loc="best")
    plt.title("ROC curve")
    plt.show()
    return(model)


# In[144]:


# Perform SVM on training set and use cross validation/out of dataset samples (testing data) for prediction
test_table_trans=test_renamed.T
model=linearsvm(table_trans[selected_feature],table_trans['target'],5,test_table_trans[selected_feature],GBM_label_test,selected_feature)

