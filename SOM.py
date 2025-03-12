import pandas as pd
from minisom import MiniSom
import matplotlib.pyplot as plt

# 1. 读取 CSV 文件
file_path = 'data\SOM_Wind.csv'
data_df = pd.read_csv(file_path)
# 2. 数据预处理
# 提取特征列和目标列
features = data_df[['U10', 'V10', 'U100', 'V100']].values
target = data_df['TARGETVAR'].values
# 对特征数据进行标准化处理
features = (features - features.mean(axis=0)) / features.std(axis=0)

# 3. 初始化 SOM 模型
som_shape = (10, 10)  # SOM 网络的形状
input_len = features.shape[1]  # 输入数据的维度
sigma = 1.0  # 邻域半径
learning_rate = 0.5  # 学习率
som = MiniSom(som_shape[0], som_shape[1], input_len, sigma=sigma, learning_rate=learning_rate)

# 4. 训练 SOM 模型
num_iterations = 1000
som.train_random(features, num_iterations)

# 5. 可视化分析

# 5.1 统一距离矩阵（U - Matrix）
plt.figure(figsize=(10, 10))
plt.pcolor(som.distance_map().T, cmap='bone_r')
plt.colorbar()
plt.title('U - Matrix')
plt.show()

# 5.2 分量平面可视化
for i, var in enumerate(['U10', 'V10', 'U100', 'V100']):
    plt.figure(figsize=(8, 8))
    plt.pcolor(som.get_weights()[:, :, i].T, cmap='viridis')
    plt.colorbar()
    plt.title(f'Component Plane - {var}')
    plt.show()

# 5.3 样本映射到 SOM 节点并可视化功率输出
target_normalized = (target - target.min()) / (target.max() - target.min())
plt.figure(figsize=(10, 10))
for cnt, xx in enumerate(features):
    w = som.winner(xx)  # 获取获胜神经元
    plt.plot(w[0] + 0.5, w[1] + 0.5, 'o', markerfacecolor=plt.cm.viridis(target_normalized[cnt]),
             markeredgecolor='k', markersize=15, markeredgewidth=1)
plt.title('Mapping of TARGETVAR on SOM')
# 创建 ScalarMappable 对象并指定规范和颜色映射
sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=target_normalized.min(), vmax=target_normalized.max()))
sm.set_array([])
# 获取当前坐标轴
ax = plt.gca()
plt.colorbar(sm, ax=ax)
plt.show()