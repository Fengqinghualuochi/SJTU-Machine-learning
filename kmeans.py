import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 设置图片清晰度
plt.rcParams['figure.dpi'] = 300
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 可以根据系统情况替换为其他支持中文的字体
# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False

# 步骤1：读取数据
def read_data(file_path):
    """
    读取数据文件
    参数:
    file_path (str): 数据文件路径
    返回:
    df (DataFrame): 读取的数据
    """
    df = pd.read_csv(file_path)
    return df

# 步骤2：蒙特卡洛场景生成
def monte_carlo_scene_generation(data, num_days):
    """
    基于蒙特卡洛方法生成场景

    参数:
    data (DataFrame): 原始数据
    num_days (int): 要生成的天数

    返回:
    scenarios (ndarray): 生成的场景数据，形状为 (num_days, 24)
    """
    # 假设数据是按天排列，每天 24 个点，提取功率列
    power_data = data['POWER'].values
    # 按天分割数据
    daily_data = [power_data[i:i + 24] for i in range(0, len(power_data), 24)]
    daily_data = np.array(daily_data)

    # 获取数据的统计特征（均值和标准差）
    means = np.mean(daily_data, axis=0)
    stds = np.std(daily_data, axis=0)

    # 生成随机场景
    scenarios = np.random.normal(loc=means, scale=stds, size=(num_days, 24))
    return scenarios

# 自定义的 K-means 算法实现
def custom_kmeans(scenarios, num_clusters, max_iterations=100, tolerance=1e-4):
    """
    自定义的 K-means 聚类算法

    参数:
    scenarios (ndarray): 输入的场景数据，形状为 (num_scenarios, num_features)
    num_clusters (int): 聚类的数量
    max_iterations (int): 最大迭代次数
    tolerance (float): 聚类中心变化的阈值，用于判断收敛

    返回:
    centroids (ndarray): 最终的聚类中心，形状为 (num_clusters, num_features)
    labels (ndarray): 每个数据点所属的聚类标签，形状为 (num_scenarios,)
    """
    num_scenarios, num_features = scenarios.shape

    # 随机初始化聚类中心
    np.random.seed(42)
    centroids = scenarios[np.random.choice(num_scenarios, num_clusters, replace=False)]

    for _ in range(max_iterations):
        # 计算每个数据点到各个聚类中心的距离
        distances = np.sqrt(((scenarios[:, np.newaxis] - centroids) ** 2).sum(axis=2))

        # 将每个数据点分配到最近的聚类中心
        labels = np.argmin(distances, axis=1)

        old_centroids = centroids.copy()

        # 更新聚类中心
        for i in range(num_clusters):
            cluster_points = scenarios[labels == i]
            if len(cluster_points) > 0:
                centroids[i] = cluster_points.mean(axis=0)

        # 检查聚类中心是否收敛
        center_shift = np.sqrt(((centroids - old_centroids) ** 2).sum())
        if center_shift < tolerance:
            break

    return centroids, labels

# 步骤3：K-means 聚类削减
def kmeans_scenario_reduction(scenarios, num_reduced_scenarios):
    """
    使用自定义 K-means 聚类对生成的场景进行削减

    参数:
    scenarios (ndarray): 生成的场景数据，形状为 (num_scenarios, num_features)
    num_reduced_scenarios (int): 削减后要保留的场景数量

    返回:
    reduced_scenarios (ndarray): 削减后的场景数据，形状为 (num_reduced_scenarios, num_features)
    """
    centroids, _ = custom_kmeans(scenarios, num_reduced_scenarios)
    return centroids

# 计算每个场景的概率
def calculate_scenario_probabilities(labels, num_reduced_scenarios):
    """
    计算每个场景的概率

    参数:
    labels (ndarray): 每个数据点所属的聚类标签
    num_reduced_scenarios (int): 削减后要保留的场景数量

    返回:
    probabilities (ndarray): 每个场景的概率
    """
    num_samples = len(labels)
    probabilities = np.zeros(num_reduced_scenarios)
    for i in range(num_reduced_scenarios):
        # 统计每个簇中的样本数量
        cluster_count = np.sum(labels == i)
        # 计算每个场景的概率
        probabilities[i] = cluster_count / num_samples
    return probabilities

# 绘制三维图
def plot_3d(scenarios, title):
    """
    绘制三维图

    参数:
    scenarios (ndarray): 场景数据，形状为 (num_days, 24)
    title (str): 图的标题
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    num_days = scenarios.shape[0]
    for day in range(num_days):
        x = np.arange(24)  # 时间（24 个点）
        y = np.full(24, day)  # 天数
        z = scenarios[day]  # 功率
        ax.plot(x, y, z)

    ax.set_xlabel('时间')
    ax.set_ylabel('天数')
    ax.set_zlabel('功率')
    ax.set_title(title)
    plt.show()

def main():
    file_path = 'data\solar2013.csv'
    num_days = 500  # 蒙特卡洛生成的天数
    num_reduced_scenarios = 5  # 削减后保留的场景数量
    # 读取数据
    data = read_data(file_path)
    # 蒙特卡洛场景生成
    scenarios = monte_carlo_scene_generation(data, num_days)
    # 绘制生成的场景的三维图
    plot_3d(scenarios, '蒙特卡洛生成的 500 天场景')
    # K-means 聚类削减
    reduced_scenarios = kmeans_scenario_reduction(scenarios, num_reduced_scenarios)
    # 绘制削减后的场景的三维图
    plot_3d(reduced_scenarios, '聚类削减后的 5 类典型场景')
if __name__ == "__main__":
    main()