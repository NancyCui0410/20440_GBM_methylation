#!/usr/bin/env python
# coding: utf-8

# In[336]:


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


# In[5]:


mypath="C:/Graduate school/Spring22/20.440/gdc_download_20220405_034243.083206"


# In[12]:


listOfFiles = list()
for (dirpath, dirnames, filenames) in os.walk(mypath):
    listOfFiles += [os.path.join(dirpath, file) for file in filenames if 'level3betas' in file]


# In[34]:


table= pd.read_csv(listOfFiles[0], sep="\t",engine='python',header=None)
for i in range(1,len(listOfFiles)):
    element=pd.read_csv(listOfFiles[i], sep="\t",engine='python',header=None)
    table[str(i+1)]=element.iloc[:,1]


# In[47]:


control_table=pd.read_csv('GSE15745-GPL8490_series_matrix.txt',sep='\t',skiprows=97, header=None, engine='python')


# In[43]:


print(len(control_table))


# In[49]:


control_table=control_table.iloc[0:len(control_table)-1,:]
print(control_table)


# In[57]:


print(len(list(set(control_table.iloc[:,0].values)&set(table.iloc[:,0].values))))


# In[66]:


table_reindex=table.set_index(0)


# In[67]:


control_table_reindex=control_table.set_index(0)


# In[85]:


GBM_final=table_reindex.sort_index()
control_final=control_table_reindex.sort_index()
GBM_final.to_csv('GBM_table.csv')
control_final.to_csv('control_table.csv')


# In[100]:


GBM_nona=GBM_final.dropna()
control_nona=control_final.dropna()
common_index=list(set(GBM_nona.index)&set(control_nona.index))


# In[101]:


GBM_total_nona=GBM_nona.loc[common_index,:]
control_total_nona=control_nona.loc[common_index,:]


# In[102]:


control_total_nona.to_csv('control_table_noNa.csv')
GBM_total_nona.to_csv('GBM_table_noNa.csv')


# In[108]:


GBM_name=dict()
for i in range(0,len(GBM_total_nona.columns)):
    GBM_name[GBM_total_nona.columns[i]]='GBM'+str(i+1)
control_name=dict()
for i in range(0,len(control_total_nona.columns)):
    control_name[control_total_nona.columns[i]]='Control'+str(i+1)


# In[110]:


GBM_renamed=GBM_total_nona.rename(columns=GBM_name)
control_renamed=control_total_nona.rename(columns=control_name)


# In[112]:


control_renamed.to_csv('control_table_noNa_final.csv')
GBM_renamed.to_csv('GBM_table_noNa_final.csv')


# In[243]:


#Agglomerative clustering of patients
# If uncomment lines in this block, shows donor tiers next to names
def plot_dendrogram(feature_matrix,target):
    distance=pdist(feature_matrix,'euclidean')
    Z=linkage(distance,'average')
    fig=plt.figure(figsize = (15,6))
#     target_label=[]
#     for i in range(0,len(target)):
#         target_label.append(feature_matrix.index[i]+'('+str(target[i])+')')
    dn=dendrogram(Z,labels=feature_matrix.index,leaf_rotation=90)
    plt.show()
    fig, ax = plt.subplots()
    fig.set_figheight(2)
    fig.set_figwidth(15)
    points=[]
    for i in dn['ivl']:
        if 'Control' in i:
            points=points+[0]
        else:
            points=points+[1]
    cmap = mpl.colors.ListedColormap(['#FFAEB9','#B4EEB4'])
    bounds = [-0.5,0.5,1.5]
    points = [points for i in range(100)]
#     print(points)
#     points = np.random.rand(10,10)
#     print(points.shape)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    ax.imshow(points, cmap=cmap, norm=norm)
#     ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
    plt.show()
#     cmap = mpl.colors.ListedColormap(['#FFAEB9','#B4EEB4'])
#     mat=ax.matshow(a, cmap=cmap)
#     f.colorbar(points, orientation="horizontal")
#     print(dn['ivl'])
    return(dn['ivl'])


# In[114]:


GBM_label=[0]*len(control_renamed.columns)+([1]*len(GBM_renamed.columns))


# In[122]:


total_table=control_renamed.join(GBM_renamed)


# In[123]:


print(total_table)


# In[330]:


GBM_avg=GBM_renamed.mean(axis=1)
control_avg=control_renamed.mean(axis=1)


# In[334]:


plt.figure(figsize=(15,4))
plt.scatter([i for i in range(0,len(GBM_avg))],GBM_avg,s=2,color='#B4EEB4')
plt.scatter([i for i in range(0,len(control_avg))],control_avg,s=2,color='#FFAEB9')
plt.xlabel('CpGs')
plt.ylabel('Beta value')
plt.show()


# In[244]:


GBM_culster=plot_dendrogram(total_table.T,GBM_label)


# In[131]:


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
    loading_matrix=pca.components_.T * np.sqrt(pca.explained_variance_)
    return(principalDf,loading_matrix,pca.explained_variance_ratio_)


# In[176]:


# This block adds the name of donors on each point
# Input result is the 'y variable' with donors as row labels
# If uncomment block, plot feature loadings on current x and y axis PCs
def plotPCA(principalDf,selected_feature,loading_matrix,explained_variance):
    for i in range(0,len(principalDf.columns.values)-2):
        for j in range(i+1, len(principalDf.columns.values)-1):
            fig = plt.figure(figsize = (8,8))
            ax = fig.add_subplot(1,1,1) 
            ax.set_xlabel('PC'+str(i+1)+'('+str(round(explained_variance[i]*100,2))+'%)', fontsize = 15)
            ax.set_ylabel('PC'+str(j+1)+'('+str(round(explained_variance[j]*100,2))+'%)', fontsize = 15)
            principalDf_sort=principalDf.sort_values('target',axis=0,ascending=True)
            print(principalDf_sort)
            targets = principalDf_sort['target'].values
            color_list={0:'#FFAEB9',1:'#B4EEB4'}
            colors = [color_list[c] for c in targets]
#             colors = color_list[targets]
#             for target, color in zip(targets,colors):
#                 indicesToKeep = [i for i, x in enumerate(targets) if x == target]
#                 if len(indicesToKeep)>0:
#                     ax.scatter(principalDf_sort.loc[indicesToKeep,'PC'+str(i+1)]
#                                , principalDf_sort.loc[indicesToKeep,'PC'+str(j+1)]
#                                , c = color
#                                , s = 50)
            sc = ax.scatter(principalDf_sort.loc[:,'PC'+str(i+1)], principalDf_sort.loc[:,'PC'+str(j+1)], c=colors)
#             plt.colorbar(sc)
#             for k, txt in enumerate(principalDf_sort.index):
#                 ax.annotate(txt, (principalDf_sort['PC'+str(i+1)][k], principalDf_sort['PC'+str(j+1)][k]))
#             print(sort(pd.DataFrame(loading_matrix).iloc[:,i]))
            plt.show()
#             plt.bar(selected_feature, pd.DataFrame(loading_matrix).iloc[:,i])
#             plt.xticks(rotation='vertical')
#             plt.title('Loading of features on PC'+str(i+1))
#             plt.show()
#             plt.bar(selected_feature, pd.DataFrame(loading_matrix).iloc[:,j])
#             plt.xticks(rotation='vertical')
#             plt.title('Loading of features on PC'+str(j+1))
#             plt.show()


# In[173]:


[principalDf2, loading_matrix,explained_variance]=PCAanalysis(total_table.T)


# In[290]:


[principalDf3, loading_matrix2,explained_variance2]=PCAanalysis(table_trans[selected_feature])


# In[292]:


# This is the master run script for visualizing PCA analysis
# Calls PCAanalysis and PlotPCA functions
# feature_matrix: a matrix of physical features to run PCA on, result: label values, name: name of the label used
def PCAmasterrun(principalDf2, loading_matrix,explained_variance,feature_matrix,result):
#     [principalDf2, loading_matrix,explained_variance]=PCAanalysis(feature_matrix)
    plt.figure(figsize=(8,4))
    plt.bar(['PC'+ str(i) for i in range(1,7)],explained_variance[0:6],color='#79CDCD')
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


# In[180]:


PCAmasterrun(principalDf2, loading_matrix,explained_variance,total_table.T,GBM_label)


# In[293]:


PCAmasterrun(principalDf3, loading_matrix2,explained_variance2,table_trans[selected_feature],GBM_label)


# In[327]:


plt.figure(figsize=(8,4))
plt.hist(loading_matrix[:,0],color='#79CDCD',bins=20)
plt.ylabel('Frequency')
plt.xlabel('Loadings')


# In[245]:


# Lasso regression to eliminate features
def ElasticnetElim(a, x, y):
    clf = linear_model.Lasso(alpha=a)
    clf.fit(x,y)
    selected_feature=[]
    for i in range(0,len(clf.coef_)):
        if clf.coef_[i]!=0:
            selected_feature.append(x.columns.values[i])
    return(selected_feature)


# In[249]:


# Example of eliminating all physical features related to flow
table_trans=total_table.T
selected_feature=ElasticnetElim(0.05,total_table.T,GBM_label)
sns.heatmap(table_trans[selected_feature].corr(),cmap="YlGnBu")


# In[250]:





# In[337]:


principalDf2['target']=GBM_label
print(principalDf2)
ax = sns.violinplot(x='target', y='PC2',
                    data=principalDf2, palette=['#FFAEB9','#B4EEB4'])
ax.set_xticklabels(['Control','GBM'])
ax.set_xlabel('')
ax.set_ylabel('Values on PC2')
res = stats.ttest_ind(principalDf2['PC2'][0:295], principalDf2['PC2'][295:801], 
                      equal_var=True)

display(res)


# In[308]:


ax = sns.violinplot(x='target', y='cg27409364',
                    data=table_trans, palette=['#FFAEB9','#B4EEB4'])
ax.set_xticklabels(['Control','GBM'])
ax.set_xlabel('')
ax.set_ylabel('Values on cg27409364')
res = stats.ttest_ind(table_trans['cg27409364'][0:295], table_trans['cg27409364'][295:801], 
                      equal_var=True)

display(res)


# In[274]:


loading_PC=pd.DataFrame(loading_matrix,index=table_trans.columns[:len(table_trans.columns)-1], columns=None)
print(loading_PC)


# In[279]:


loading_PC_reorg=loading_PC.sort_values(1)


# In[284]:


plt.figure(figsize=(15,6))
plt.bar(loading_PC_reorg.index.values[0:40],loading_PC_reorg.iloc[0:40,1],color='#79CDCD')
plt.xticks(rotation = 90)
plt.show()


# In[286]:


plt.figure(figsize=(15,6))
plt.bar(loading_PC_reorg.index.values[-40:],loading_PC_reorg.iloc[-40:,1],color='#79CDCD')
plt.xticks(rotation = 90)
plt.show()


# In[ ]:


print(loading_matrix2)


# In[300]:


loading_PC=pd.DataFrame(loading_matrix2,index=table_trans[selected_feature].columns[:len(table_trans[selected_feature].columns)], columns=None)
loading_PC.index.name="index"
loading_PC_reorg=loading_PC.sort_values(by=0)


# In[326]:


plt.figure(figsize=(8,4))
plt.bar(loading_PC_reorg.index.values,loading_PC_reorg.iloc[:,1],color='#79CDCD')
plt.xticks(rotation = 90)
plt.show()


# In[338]:


# Support vector machine to separate data
def linearsvm(x,y,fold):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=1/fold, random_state=0)
    model = svm.SVC(kernel="linear", probability=True)
    model.fit(X_train, y_train)
#     print(clf.predict(X_test))
    print(model.score(X_test,y_test))
    decision_scores = model.decision_function(X_test)
    fpr, tpr, thres = roc_curve(y_test, decision_scores)
    preds = model.predict(X_test)
    print('AUC: {:.3f}'.format(roc_auc_score(y_test, decision_scores)))
    print("accuracy: ", metrics.accuracy_score(y_test, preds))
    print("precision: ", metrics.precision_score(y_test, preds)) 
    print("recall: ", metrics.recall_score(y_test, preds))
    print("f1: ", metrics.f1_score(y_test, preds))
    # roc curve
    plt.plot(fpr, tpr, '#79CDCD', label='Linear SVM')
    plt.plot([0,1],[0,1], "k--", label='Random Guess')
    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.legend(loc="best")
    plt.title("ROC curve")
    plt.show()
    print(model.coef_)
    return()


# In[339]:


linearsvm(table_trans[selected_feature],table_trans['target'],5)


# In[340]:


print(selected_feature)

