import pandas as pd
import seaborn as sns; sns.set()
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.cluster import KMeans
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler


beer = pd.read_excel('/users/qinqi/PycharmProjects/projectgds/js_978.xlsx')
#beer=beer.drop(['sf','xueli'],axis=1)
beer.dropna(axis=0,how='all')
beer.dropna(inplace=True)
beer.reset_index(drop=True, inplace=True)

X=beer[["fjmxzzr","cxyw","fjmzmyh","fpgyyh","ncjmsh","ncscyh"]].fillna(0)

#计算均值
a=X.groupby('cluster').mean()



#相关性分析
#sns.heatmap(X.corr(),annot=True,cmap='Blues')

#聚类评估：轮廓系数 越大越好
scores = []
for k in range(2,20):
    labels = KMeans(n_clusters=k).fit(X).labels_
    score = metrics.silhouette_score(X, labels)
    scores.append(score)
plt.plot(list(range(2,20)),scores)
plt.xlabel("Number")
plt.ylabel("score")#看什么时候是拐点


km = KMeans(n_clusters=11).fit(X)
#展示分类
beer['cluster']=km.labels_
beer.sort_values('cluster')
colors = np.array(['red','green','blue','yellow','pink','black','brown','purple','cyan','magenta','white','gray'])
#多指标直观查看
scatter_matrix(beer[["fjmxzzr","cxyw","fjmzmyh","fpgyyh","ncjmsh","ncscyh"]],s=100,alpha=1,c=colors[beer["cluster"]],figsize=(10,8))
plt.suptitle("picture with k")