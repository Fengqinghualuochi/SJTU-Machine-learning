import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# 1. 数据处理模块
class LoadProfileDataset(Dataset):
    """电力负荷数据集类"""

    def __init__(self, csv_file, sequence_length=24):
        """
        初始化函数

        参数:
            csv_file (str): CSV文件路径
            sequence_length (int): 序列长度，默认为24小时
        """
        # 读取CSV文件
        self.data = pd.read_csv(csv_file)

        # 归一化功率数据
        self.scaler = MinMaxScaler()
        self.data['POWER_NORMALIZED'] = self.scaler.fit_transform(
            self.data['POWER'].values.reshape(-1, 1)
        )

        self.sequence_length = sequence_length

        # 将数据分成日序列
        self._prepare_sequences()

    def _prepare_sequences(self):
        """将数据准备为日序列"""
        self.sequences = []

        # 计算序列总数（假设数据是连续的）
        num_sequences = len(self.data) // self.sequence_length

        for i in range(num_sequences):
            start_idx = i * self.sequence_length
            end_idx = start_idx + self.sequence_length

            if end_idx <= len(self.data):
                # 获取一天的功率数据
                seq = self.data['POWER_NORMALIZED'].values[start_idx:end_idx]
                self.sequences.append(seq)

        # 转换为numpy数组
        self.sequences = np.array(self.sequences)

    def __len__(self):
        """返回数据集大小"""
        return len(self.sequences)

    def __getitem__(self, idx):
        """获取指定索引的样本"""
        sequence = self.sequences[idx]

        # 转换为PyTorch张量
        sequence_tensor = torch.FloatTensor(sequence)

        return sequence_tensor

    def denormalize(self, normalized_data):
        """反归一化数据"""
        return self.scaler.inverse_transform(normalized_data.reshape(-1, 1)).reshape(-1)


# 2. DDPM模型模块
class SimpleUNet(nn.Module):
    """简化的UNet网络，适用于负荷曲线生成"""

    def __init__(self, sequence_length=24, time_emb_dim=32):
        super(SimpleUNet, self).__init__()

        # 时间嵌入
        self.time_embedding = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # 下采样路径
        self.down1 = nn.Sequential(
            nn.Linear(sequence_length + time_emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

        self.down2 = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )

        # 上采样路径
        self.up1 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

        self.up2 = nn.Sequential(
            nn.Linear(128 + 128, sequence_length),  # 跳跃连接
            nn.ReLU(),
            nn.Linear(sequence_length, sequence_length)
        )

    def forward(self, x, t):
        # 时间嵌入
        t_emb = self.time_embedding(t.unsqueeze(-1).float())

        # 将时间信息与输入连接
        x_t = torch.cat([x, t_emb], dim=1)

        # 下采样
        down1 = self.down1(x_t)
        down2 = self.down2(down1)

        # 上采样（带跳跃连接）
        up1 = self.up1(down2)
        up2 = self.up2(torch.cat([up1, down1], dim=1))

        return up2


class DDPM(nn.Module):
    """去噪扩散概率模型 (DDPM)"""

    def __init__(self, model, beta_start=1e-4, beta_end=0.02, timesteps=1000):
        """
        初始化函数
        参数:
            model: 去噪网络
            beta_start: beta调度的起始值
            beta_end: beta调度的结束值
            timesteps: 扩散步数
        """
        super(DDPM, self).__init__()

        self.model = model
        self.timesteps = timesteps

        # 定义线性beta调度
        self.betas = torch.linspace(beta_start, beta_end, timesteps)

        # 预计算扩散过程中的其他参数
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # 用于计算噪声采样的参数
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        # 用于计算均值参数
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def forward_diffusion(self, x, t):
        """
        前向扩散过程: q(x_t | x_0)
        参数:
            x: 原始数据 x_0
            t: 时间步
        返回:
            添加噪声后的数据 x_t
            噪声 epsilon
        """
        noise = torch.randn_like(x)

        # 提取对应时间步的参数
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod.to(x.device)[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod.to(x.device)[t]

        # 在维度上调整
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.view(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.view(-1, 1)

        # 计算带噪声的样本
        x_t = sqrt_alphas_cumprod_t * x + sqrt_one_minus_alphas_cumprod_t * noise

        return x_t, noise

    def sample_timestep(self, x, t):
        """
        预测去噪过程的一个时间步
        参数:
            x: 当前带噪声的样本 x_t
            t: 当前时间步
        返回:
            预测的去噪后的样本 x_{t-1}
        """
        # 获取当前时间的beta和相关参数
        betas_t = self.betas.to(x.device)[t]
        sqrt_recip_alphas_t = self.sqrt_recip_alphas.to(x.device)[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod.to(x.device)[t]

        # 在维度上调整
        betas_t = betas_t.view(-1, 1)
        sqrt_recip_alphas_t = sqrt_recip_alphas_t.view(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.view(-1, 1)

        # 预测噪声
        predicted_noise = self.model(x, t)

        # 计算均值
        model_mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)

        # 如果是最后一步，则直接返回均值
        if t[0] == 0:
            return model_mean

        # 否则，添加噪声
        posterior_variance_t = self.posterior_variance.to(x.device)[t].view(-1, 1)
        noise = torch.randn_like(x)

        return model_mean + torch.sqrt(posterior_variance_t) * noise

    def sample(self, n_samples, sequence_length, device):
        """
        从纯噪声生成样本
        参数:
            n_samples: 样本数量
            sequence_length: 序列长度
            device: 设备
        返回:
            生成的样本
        """
        self.eval()
        with torch.no_grad():
            # 从纯噪声开始
            x = torch.randn(n_samples, sequence_length).to(device)

            # 逐步去噪
            for t in tqdm(reversed(range(self.timesteps)), desc="采样过程"):
                t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)
                x = self.sample_timestep(x, t_batch)

            # 将生成结果限制在[0,1]范围内
            x = torch.clamp(x, 0., 1.)

        return x


# 3. 训练模块
def train_ddpm(ddpm, dataloader, num_epochs=100, lr=1e-4, device="cpu", save_dir="ddpm_models"):
    """
    训练DDPM模型

    参数:
        ddpm: DDPM模型
        dataloader: 数据加载器
        num_epochs: 训练轮数
        lr: 学习率
        device: 训练设备
        save_dir: 模型保存目录
    """
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    # 定义优化器
    optimizer = torch.optim.Adam(ddpm.parameters(), lr=lr)
    # 损失函数 (MSE)
    mse = nn.MSELoss()
    # 将模型移至指定设备
    ddpm = ddpm.to(device)
    # 记录损失
    losses = []
    # 训练循环
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for step, batch in enumerate(progress_bar):
            # 将数据移至指定设备
            batch = batch.to(device)

            # 清零梯度
            optimizer.zero_grad()

            # 随机选择时间步
            t = torch.randint(0, ddpm.timesteps, (batch.shape[0],), device=device).long()

            # 应用前向扩散
            x_t, noise = ddpm.forward_diffusion(batch, t)

            # 预测噪声
            noise_pred = ddpm.model(x_t, t)

            # 计算损失
            loss = mse(noise_pred, noise)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            # 更新进度条
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

        # 计算平均损失
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)

        print(f"Epoch {epoch + 1}/{num_epochs}, 平均损失: {avg_loss:.6f}")

        # 每100个epoch保存一次模型
        if (epoch + 1) % 100 == 0:
            torch.save(ddpm.state_dict(), f"{save_dir}/ddpm_epoch{epoch + 1}.pt")

    # 保存最终模型
    torch.save(ddpm.state_dict(), f"{save_dir}/ddpm_final.pt")

    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('DDPM 训练损失')
    plt.xlabel('训练轮次')
    plt.ylabel('MSE 损失')
    plt.grid(True)
    plt.savefig(f"{save_dir}/ddpm_loss.png")
    plt.close()

    return ddpm, losses

# 4. 生成和可视化模块
def generate_and_visualize(ddpm, n_samples=9, sequence_length=24, device="cpu", dataset=None):
    """
    生成并可视化负荷曲线
    参数:
        ddpm: 训练好的DDPM模型
        n_samples: 生成样本数，默认为9个
        sequence_length: 序列长度
        device: 设备
        dataset: 数据集对象，用于反归一化
    """

    # 生成样本
    samples = ddpm.sample(n_samples, sequence_length, device).cpu().numpy()
    # 反归一化（如果提供了数据集）
    if dataset is not None:
        denormalized_samples = []
        for i in range(n_samples):
            denormalized_samples.append(dataset.denormalize(samples[i]))
        samples = np.array(denormalized_samples)
    # 创建3x3网格可视化
    plt.figure(figsize=(15, 12))
    # 设置全局样式
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['grid.linestyle'] = '--'
    # 创建颜色循环
    colors = plt.cm.tab10(np.linspace(0, 1, n_samples))
    for i in range(n_samples):
        ax = plt.subplot(3, 3, i + 1)

        # 绘制曲线
        plt.plot(samples[i], marker='o', color=colors[i], linewidth=2,
                 markersize=5, markeredgecolor='white', markeredgewidth=0.5)

        # 添加小时标签
        hours = np.arange(0, 24, 4)  # 每4小时标记一次
        plt.xticks(hours, [f"{h}:00" for h in hours])

        # 设置标题和标签
        plt.title(f"负荷曲线 #{i + 1}", fontsize=12, fontweight='bold')

        # 仅在左侧和底部子图添加标签
        if i % 3 == 0:  # 左侧子图
            plt.ylabel("负荷 (kW)" if dataset is not None else "归一化负荷", fontsize=10)
        if i >= 6:  # 底部子图
            plt.xlabel("时间", fontsize=10)
        # 设置y轴范围，使所有图一致
        if dataset is not None:
            plt.ylim(np.min(samples) * 0.9, np.max(samples) * 1.1)
        else:
            plt.ylim(-0.05, 1.05)
        # 添加次网格
        ax.grid(True, which='both', linestyle=':', alpha=0.2)
        # 去除顶部和右侧边框
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.suptitle("DDPM生成的日负荷曲线样本", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  # 为总标题留出空间
    plt.savefig("generated_load_profiles.png", dpi=300, bbox_inches='tight')
    plt.show()

    return samples

# 5. 主函数
def main():
    # 设置参数
    data_path = "L1-train.csv"  # 数据路径
    batch_size = 16
    sequence_length = 24
    timesteps = 30  # 扩散步数
    time_emb_dim = 32  # 时间嵌入维度
    learning_rate = 1e-4
    num_epochs = 3000
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = "ddpm_models"
    # 加载数据
    print("正在加载数据...")
    dataset = LoadProfileDataset(data_path, sequence_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"数据加载完成，共有 {len(dataset)} 个日负荷曲线")
    # 创建模型
    print("正在初始化DDPM模型...")
    denoise_model = SimpleUNet(sequence_length, time_emb_dim)
    ddpm = DDPM(denoise_model, timesteps=timesteps)
    # 训练模型
    print("开始训练DDPM模型...")
    ddpm, losses = train_ddpm(
        ddpm=ddpm,
        dataloader=dataloader,
        num_epochs=num_epochs,
        lr=learning_rate,
        device=device,
        save_dir=save_dir
    )
    # 生成并可视化样本
    print("生成并可视化负荷曲线...")
    generated_samples = generate_and_visualize(
        ddpm=ddpm,
        n_samples=9,  # 可视化9个样本
        sequence_length=sequence_length,
        device=device,
        dataset=dataset
    )
    print("DDPM负荷场景生成完成！")

if __name__ == "__main__":
    main()