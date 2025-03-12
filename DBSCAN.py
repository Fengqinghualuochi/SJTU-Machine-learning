import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# 读取数据
df = pd.read_csv('data\L1-train.csv')
# 提取负荷和温度数据
X = df[['LOAD', 'w1']].values
# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 调整 DBSCAN 参数
db = DBSCAN(eps=0.13, min_samples=14)  # 调整 eps 和 min_samples 的值
labels = db.fit_predict(X_scaled)

# 统计簇的数量（排除噪声点）
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)

# 绘制原始散点图
plt.figure(figsize=(10, 6))
plt.scatter(df['w1'], df['LOAD'], c='gray', alpha=0.5, label='Original Data')
plt.title('Original Scatter Plot of Load and Temperature')
plt.xlabel('Temperature')
plt.ylabel('LOAD')
plt.legend()
plt.show()

# 绘制聚类结果散点图
plt.figure(figsize=(10, 6))
unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # 黑色用于噪声点
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)
    xy = X[class_member_mask]
    plt.scatter(xy[:, 1], xy[:, 0], c=[col], edgecolor='k', label=f'Cluster {k}' if k != -1 else 'Noise')

plt.title('Scatter Plot of Load and Temperature with DBSCAN Clustering')
plt.xlabel('Temperature')
plt.ylabel('LOAD')
plt.legend()
plt.show()