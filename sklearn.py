import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.datasets import  load_boston
from sklearn.linear_model import LinearRegression
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler

def find_max():
    boston = load_boston()
    print(boston.data.shape)
    print(boston.feature_names)
    print(boston.keys())
    
    bos = pd.DataFrame(boston.data)
    bos.col = boston.feature_names
    bos['PRICE'] = boston.target
    X = bos.drop('PRICE',axis = 1)
    linereg = LinearRegression()
    linereg.fit(X,bos['PRICE'])
    
    print('Value of W:',linereg.coef_)

    coef = (np.fabs(linereg.coef_)).tolist()
    max_index = coef.index(max(coef))
    
    print(max_index)
    print('Value of B',linereg.intercept_)
    print(boston.feature_names[max_index])
    return boston.feature_names[max_index]

def kmean(clusters):

    li = load_iris()
    li_data = li['data']
    li_names = li['feature_names']
    li_target = li['target']
    scale = MinMaxScaler().fit(li_data)
    li_dataScale = scale.transform(li_data)
    kmeans = KMeans(n_clusters = clusters,random_state = 123).fit(li_dataScale)
    print('K-Means:',kmeans)

    tsne = TSNE(n_components = 2,init = 'random',random_state = 177).fit(li_data)
    df = pd.DataFrame(tsne.embedding_)
    df['labels'] = kmeans.labels_
    
    if clusters==3:
        df1 = df[df['labels']==0]
        df2 = df[df['labels']==1]
        df3 = df[df['labels']==2]
        plt.plot(df1[0],df1[1],'bo',df2[0],df2[1],'r*',df3[0],df3[1],'gD')
        plt.show()
        
    if clusters==4:
        df1 = df[df['labels']==0]
        df2 = df[df['labels']==1]
        df3 = df[df['labels']==2]
        df4 = df[df['labels']==3]
        fig = plt.figure(figsize=(9,6))
        plt.plot(df1[0],df1[1],'bo',df2[0],df2[1],'r*',df3[0],df3[1],'gD',df4[0],df4[1],'m<')
        plt.show()
        
    if clusters==5:
        df1 = df[df['labels']==0]
        df2 = df[df['labels']==1]
        df3 = df[df['labels']==2]
        df4 = df[df['labels']==3]
        df5 = df[df['labels']==4]
        fig = plt.figure(figsize=(9,6))
        plt.plot(df1[0],df1[1],'r*',df2[0],df2[1],'m<',df3[0],df3[1],'yo',df4[0],df4[1],'bo',df5[0],df5[1],'gD',)
        plt.show()

if __name__ == '__main__':
    print(find_max())
    kmean(5)