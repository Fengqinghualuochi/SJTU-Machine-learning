import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# 1. 数据处理部分
class PhotovoltaicDataset(Dataset):

    def __init__(self, csv_file, sequence_length=24):
        """
        初始化函数

        参数:
            csv_file (str): CSV文件路径
            sequence_length (int): 序列长度，默认为24小时
        """
        # 读取CSV文件
        self.data = pd.read_csv(csv_file, delimiter=',')
        # 将时间戳转换为datetime类型
        self.data['TIMESTAMP'] = pd.to_datetime(self.data['TIMESTAMP'])
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

        # 按天分组数据
        grouped = self.data.groupby(self.data['TIMESTAMP'].dt.date)

        for date, group in grouped:
            if len(group) >= self.sequence_length:
                # 获取完整日的功率数据
                seq = group['POWER_NORMALIZED'].values[:self.sequence_length]
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

# 2. VAE模型部分
class VAE(nn.Module):
    """变分自编码器模型"""

    def __init__(self, sequence_length=24, hidden_dim=64, latent_dim=16):
        """
        初始化函数

        参数:
            sequence_length (int): 序列长度，默认为24小时
            hidden_dim (int): 隐藏层维度
            latent_dim (int): 潜在空间维度
        """
        super(VAE, self).__init__()

        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # 编码器网络
        self.encoder = nn.Sequential(
            nn.Linear(sequence_length, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU()
        )

        # 均值和对数方差
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # 解码器网络
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, sequence_length),
            nn.Sigmoid()  # 输出归一化到[0,1]的功率值
        )

    def encode(self, x):
        """编码器：将输入编码为潜在表示"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z):
        """解码器：将潜在表示解码为重构输出"""
        return self.decoder(z)

    def forward(self, x):
        """前向传播"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def generate(self, num_samples=1, device="cpu"):
        """生成新的光伏场景"""
        with torch.no_grad():
            # 从标准正态分布中采样
            z = torch.randn(num_samples, self.latent_dim).to(device)
            # 解码生成样本
            samples = self.decode(z)
        return samples

# 3. 损失函数部分
def vae_loss_function(recon_x, x, mu, logvar, beta=0.5):
    """
    VAE损失函数：重构损失 + KL散度
    参数:
        recon_x: 重构的输入
        x: 原始输入
        mu: 均值
        logvar: 对数方差
        beta: KL散度的权重
    返回:
        loss: 总损失
        recon_loss: 重构损失
        kl_loss: KL散度损失
    """
    # 使用MSE作为重构损失
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')

    # KL散度
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # 总损失
    loss = recon_loss + beta * kl_loss

    return loss, recon_loss, kl_loss

def plot_vae_losses(losses, save_path=None):
    """
    参数:
        losses: 包含total_losses, recon_losses, kl_losses的字典
        save_path: 保存图像的路径，如果为None则直接显示
    """
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(losses['total'], 'b-', label='总损失')
    plt.xlabel('周期')
    plt.ylabel('损失值')
    plt.title('VAE总损失')
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(losses['recon'], 'r-', label='重构损失')
    plt.xlabel('周期')
    plt.ylabel('损失值')
    plt.title('重构损失')
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(losses['kl'], 'g-', label='KL散度损失')
    plt.xlabel('周期')
    plt.ylabel('损失值')
    plt.title('KL散度损失')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

# 4. 训练和生成函数
def train_vae(vae, dataloader, num_epochs=100, learning_rate=1e-3, device="cpu", save_dir="models"):
    """
    训练VAE模型

    参数:
        vae: VAE模型
        dataloader: 数据加载器
        num_epochs: 训练轮数
        learning_rate: 学习率
        device: 训练设备
        save_dir: 模型保存目录
    """
    # 创建优化器
    optimizer = optim.Adam(vae.parameters(), lr=learning_rate)

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 将模型移至指定设备
    vae = vae.to(device)

    # 创建用于跟踪损失的列表
    total_losses = []
    recon_losses = []
    kl_losses = []

    # 训练循环
    for epoch in range(num_epochs):
        vae.train()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for batch_idx, data in enumerate(progress_bar):
            # 将数据移至指定设备
            data = data.to(device)
            # 重置梯度
            optimizer.zero_grad()
            # 前向传播
            recon_batch, mu, logvar = vae(data)
            # 计算损失
            loss, recon_loss, kl_loss = vae_loss_function(recon_batch, data, mu, logvar)
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
            # 记录损失
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            # 更新进度条
            progress_bar.set_postfix({"loss": loss.item() / len(data)})
        # 计算平均损失
        avg_loss = total_loss / len(dataloader.dataset)
        avg_recon_loss = total_recon_loss / len(dataloader.dataset)
        avg_kl_loss = total_kl_loss / len(dataloader.dataset)
        # 保存损失值
        total_losses.append(avg_loss)
        recon_losses.append(avg_recon_loss)
        kl_losses.append(avg_kl_loss)
        print(f"周期 {epoch + 1}: 损失 = {avg_loss:.4f}, 重构损失 = {avg_recon_loss:.4f}, KL损失 = {avg_kl_loss:.4f}")

    # 保存最终模型
    torch.save(vae.state_dict(), f"{save_dir}/vae_final.pt")

    # 绘制最终损失曲线
    plot_vae_losses({
        'total': total_losses,
        'recon': recon_losses,
        'kl': kl_losses
    }, f"{save_dir}/vae_losses_final.png")

    print("VAE模型训练完成！")

    return vae, {'total': total_losses, 'recon': recon_losses, 'kl': kl_losses}

def generate_and_visualize_samples(vae, num_samples=10, device="cpu", scaler=None):
    """
    生成并可视化光伏场景样本

    参数:
        vae: VAE模型
        num_samples: 生成样本数量
        device: 设备
        scaler: 用于反归一化的缩放器
    """
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    vae.eval()
    with torch.no_grad():
        # 生成样本
        z = torch.randn(num_samples, vae.latent_dim).to(device) * 2.0  # 乘以2.0增加采样范围
        samples = vae.decode(z).cpu().numpy()

        # 如果提供了缩放器，反归一化样本
        if scaler is not None:
            samples_reshaped = samples.reshape(-1, 1)
            samples_rescaled = scaler.inverse_transform(samples_reshaped)
            samples = samples_rescaled.reshape(samples.shape)

    # 可视化样本
    plt.figure(figsize=(15, 10))
    for i in range(num_samples):
        plt.subplot(3, 4, i + 1)
        plt.plot(samples[i])
        plt.title(f"样本 {i + 1}")
        plt.xlabel("小时")
        plt.ylabel("功率")
        plt.grid(True)

    plt.suptitle("VAE生成的光伏场景", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    return samples

# 5. 主函数
def main():
    # 设置参数
    data_path = "solar2013.csv"  # 数据路径
    batch_size = 16
    sequence_length = 24
    hidden_dim = 128
    latent_dim = 32
    learning_rate = 1e-3
    num_epochs = 100
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = "vae_models"
    # 加载数据
    print("正在加载数据...")
    dataset = PhotovoltaicDataset(data_path, sequence_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    print(f"数据加载完成，共有 {len(dataset)} 个日光伏曲线")
    # 创建VAE模型
    vae = VAE(sequence_length, hidden_dim, latent_dim)
    # 训练模型
    print("开始训练VAE模型...")
    vae_model, losses = train_vae(
        vae=vae,
        dataloader=dataloader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        device=device,
        save_dir=save_dir
    )

    # 生成并可视化样本
    print("使用VAE生成光伏场景...")
    samples = generate_and_visualize_samples(
        vae=vae_model,
        num_samples=12,
        device=device,
        scaler=dataset.scaler
    )
    plot_vae_losses(losses)
    print("VAE光伏场景生成完成！")
if __name__ == "__main__":
    main()