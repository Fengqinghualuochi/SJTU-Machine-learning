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
import random

# 1. 数据处理模块
class PhotovoltaicCGANDataset(Dataset):
    """带有月份条件的光伏数据集"""

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

        # 提取月份信息 (1-12)
        self.data['MONTH'] = self.data['TIMESTAMP'].dt.month

        # 归一化功率数据
        self.scaler = MinMaxScaler()
        self.data['POWER_NORMALIZED'] = self.scaler.fit_transform(
            self.data['POWER'].values.reshape(-1, 1)
        )

        self.sequence_length = sequence_length
        self.num_months = 12  # 总共12个月

        # 将数据分成日序列
        self._prepare_sequences()

    def _prepare_sequences(self):
        """将数据准备为日序列，并记录对应月份"""
        self.sequences = []
        self.months = []

        # 按天分组数据
        grouped = self.data.groupby(self.data['TIMESTAMP'].dt.date)

        for date, group in grouped:
            if len(group) >= self.sequence_length:
                # 获取完整日的功率数据
                seq = group['POWER_NORMALIZED'].values[:self.sequence_length]
                self.sequences.append(seq)

                # 获取该天的月份（取第一个小时的月份即可）
                month = group['MONTH'].iloc[0]
                self.months.append(month)

        # 转换为numpy数组
        self.sequences = np.array(self.sequences)
        self.months = np.array(self.months)

    def __len__(self):
        """返回数据集大小"""
        return len(self.sequences)

    def __getitem__(self, idx):
        """获取指定索引的样本和对应月份"""
        sequence = self.sequences[idx]
        month = self.months[idx]

        # 转换为PyTorch张量
        sequence_tensor = torch.FloatTensor(sequence)

        # 将月份转为one-hot编码
        month_onehot = torch.zeros(self.num_months)
        month_onehot[month - 1] = 1  # 月份从1开始，索引从0开始

        return sequence_tensor, month_onehot, month


# 2. CGAN模型模块
class Generator(nn.Module):
    """条件生成器网络"""

    def __init__(self, latent_dim=100, condition_dim=12, sequence_length=24, hidden_dim=128):
        """
        初始化生成器

        参数:
            latent_dim: 噪声向量维度
            condition_dim: 条件向量维度（月份one-hot编码，12维）
            sequence_length: 生成序列长度
            hidden_dim: 隐藏层维度
        """
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.sequence_length = sequence_length

        # 输入层 - 噪声向量和条件向量的连接
        self.input_layer = nn.Linear(latent_dim + condition_dim, hidden_dim)

        # 多层网络
        self.hidden1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.hidden2 = nn.Linear(hidden_dim * 2, hidden_dim * 4)
        self.output_layer = nn.Linear(hidden_dim * 4, sequence_length)

        # 激活函数
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()  # 输出归一化到[0,1]范围

    def forward(self, noise, condition):
        """
        前向传播

        参数:
            noise: 随机噪声向量
            condition: 条件向量（月份one-hot编码）
        """
        # 连接噪声和条件
        x = torch.cat([noise, condition], dim=1)

        # 前向传播
        x = self.leaky_relu(self.input_layer(x))
        x = self.leaky_relu(self.hidden1(x))
        x = self.leaky_relu(self.hidden2(x))
        x = self.sigmoid(self.output_layer(x))

        return x


class Discriminator(nn.Module):
    """条件判别器网络"""

    def __init__(self, sequence_length=24, condition_dim=12, hidden_dim=128):
        """
        初始化判别器

        参数:
            sequence_length: 输入序列长度
            condition_dim: 条件向量维度（月份one-hot编码，12维）
            hidden_dim: 隐藏层维度
        """
        super(Discriminator, self).__init__()

        # 输入层 - 序列和条件向量的连接
        self.input_layer = nn.Linear(sequence_length + condition_dim, hidden_dim * 2)

        # 多层网络
        self.hidden1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.output_layer = nn.Linear(hidden_dim // 2, 1)

        # 激活函数
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()  # 输出真假概率

    def forward(self, sequence, condition):
        """
        前向传播

        参数:
            sequence: 输入序列（真实或生成）
            condition: 条件向量（月份one-hot编码）
        """
        # 连接序列和条件
        x = torch.cat([sequence, condition], dim=1)

        # 前向传播
        x = self.leaky_relu(self.input_layer(x))
        x = self.leaky_relu(self.hidden1(x))
        x = self.leaky_relu(self.hidden2(x))
        x = self.sigmoid(self.output_layer(x))

        return x


# 3. 损失函数和训练模块
def train_cgan(generator, discriminator, dataloader, num_epochs=100,
               latent_dim=100, learning_rate=0.0002, beta1=0.5,
               device="cpu", save_dir="models"):
    """
    训练CGAN模型

    参数:
        generator: 生成器模型
        discriminator: 判别器模型
        dataloader: 数据加载器
        num_epochs: 训练轮数
        latent_dim: 噪声向量维度
        learning_rate: 学习率
        beta1: Adam优化器的beta1参数
        device: 训练设备
        save_dir: 模型保存目录
    """
    # 创建优化器
    optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(beta1, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(beta1, 0.999))

    # 创建损失函数
    criterion = nn.BCELoss()

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 将模型移至指定设备
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    # 记录损失
    G_losses = []
    D_losses = []

    # 定义真假标签
    real_label = 1.0
    fake_label = 0.0

    # 训练循环
    for epoch in range(num_epochs):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        epoch_G_loss = 0.0
        epoch_D_loss = 0.0

        for batch_idx, (real_sequences, conditions, _) in enumerate(progress_bar):
            batch_size = real_sequences.size(0)

            # 将数据移至指定设备
            real_sequences = real_sequences.to(device)
            conditions = conditions.to(device)

            # -----------------
            # 训练判别器
            # -----------------
            optimizer_D.zero_grad()

            # 真实样本的标签
            labels_real = torch.full((batch_size, 1), real_label, dtype=torch.float, device=device)

            # 判别器对真实样本的输出
            outputs_real = discriminator(real_sequences, conditions)
            d_loss_real = criterion(outputs_real, labels_real)
            d_loss_real.backward()

            # 生成假样本
            noise = torch.randn(batch_size, latent_dim, device=device)
            fake_sequences = generator(noise, conditions)

            # 假样本的标签
            labels_fake = torch.full((batch_size, 1), fake_label, dtype=torch.float, device=device)

            # 判别器对假样本的输出
            outputs_fake = discriminator(fake_sequences.detach(), conditions)
            d_loss_fake = criterion(outputs_fake, labels_fake)
            d_loss_fake.backward()

            # 计算判别器总损失
            d_loss = d_loss_real + d_loss_fake
            optimizer_D.step()

            # -----------------
            # 训练生成器
            # -----------------
            optimizer_G.zero_grad()

            # 生成器希望判别器将假样本识别为真样本
            outputs_fake = discriminator(fake_sequences, conditions)
            g_loss = criterion(outputs_fake, labels_real)
            g_loss.backward()
            optimizer_G.step()

            # 记录损失
            epoch_D_loss += d_loss.item()
            epoch_G_loss += g_loss.item()

            # 更新进度条
            progress_bar.set_postfix({
                "D Loss": d_loss.item(),
                "G Loss": g_loss.item()
            })

        # 计算平均损失
        avg_D_loss = epoch_D_loss / len(dataloader)
        avg_G_loss = epoch_G_loss / len(dataloader)

        # 保存损失
        D_losses.append(avg_D_loss)
        G_losses.append(avg_G_loss)

        print(f"周期 {epoch + 1}: D损失 = {avg_D_loss:.4f}, G损失 = {avg_G_loss:.4f}")

        # 每10个epoch保存一次模型
        if (epoch + 1) % 100 == 0:
            torch.save(generator.state_dict(), f"{save_dir}/generator_epoch{epoch + 1}.pt")
            torch.save(discriminator.state_dict(), f"{save_dir}/discriminator_epoch{epoch + 1}.pt")

    # 保存最终模型
    torch.save(generator.state_dict(), f"{save_dir}/generator_final.pt")
    torch.save(discriminator.state_dict(), f"{save_dir}/discriminator_final.pt")

    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{save_dir}/cgan_loss.png")
    plt.close()

    print("CGAN模型训练完成！")

    return generator, discriminator, {"G_losses": G_losses, "D_losses": D_losses}


# 4. 生成和可视化模块
def generate_samples_by_month(generator, month, num_samples=10, latent_dim=100, device="cpu", scaler=None):
    """
    根据指定月份生成光伏场景样本

    参数:
        generator: 生成器模型
        month: 月份 (1-12)
        num_samples: 生成样本数量
        latent_dim: 噪声向量维度
        device: 设备
        scaler: 用于反归一化的缩放器
    """
    generator.eval()
    with torch.no_grad():
        # 创建条件向量 (one-hot编码)
        condition = torch.zeros(num_samples, 12, device=device)
        condition[:, month - 1] = 1  # 月份从1开始，索引从0开始

        # 生成随机噪声
        noise = torch.randn(num_samples, latent_dim, device=device)

        # 生成样本
        fake_samples = generator(noise, condition).cpu().numpy()

        # 如果提供了缩放器，反归一化样本
        if scaler is not None:
            fake_samples_reshaped = fake_samples.reshape(-1, 1)
            fake_samples_rescaled = scaler.inverse_transform(fake_samples_reshaped)
            fake_samples = fake_samples_rescaled.reshape(fake_samples.shape)

    return fake_samples


def visualize_generated_samples(samples, month, title=None):
    """
    可视化生成的样本

    参数:
        samples: 生成的样本
        month: 月份
        title: 图表标题
    """
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    month_names = ["一月", "二月", "三月", "四月", "五月", "六月",
                   "七月", "八月", "九月", "十月", "十一月", "十二月"]

    plt.figure(figsize=(12, 8))

    for i in range(min(len(samples), 9)):
        plt.subplot(3, 3, i + 1)
        plt.plot(samples[i])
        plt.title(f"样本 {i + 1}")
        plt.xlabel("小时")
        plt.ylabel("功率")
        plt.grid(True)

    if title:
        plt.suptitle(title, fontsize=16)
    else:
        plt.suptitle(f"CGAN生成的{month_names[month - 1]}光伏场景", fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def compare_months(generator, months_to_compare, num_samples=3, latent_dim=100, device="cpu", scaler=None):
    """
    比较不同月份生成的光伏场景

    参数:
        generator: 生成器模型
        months_to_compare: 要比较的月份列表
        num_samples: 每个月份生成的样本数量
        latent_dim: 噪声向量维度
        device: 设备
        scaler: 用于反归一化的缩放器
    """
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    month_names = ["一月", "二月", "三月", "四月", "五月", "六月",
                   "七月", "八月", "九月", "十月", "十一月", "十二月"]

    # 使用相同的噪声，只改变月份条件
    fixed_noise = torch.randn(num_samples, latent_dim, device=device)

    plt.figure(figsize=(15, 10))

    for i, month in enumerate(months_to_compare):
        # 创建条件向量
        condition = torch.zeros(num_samples, 12, device=device)
        condition[:, month - 1] = 1

        # 生成样本
        with torch.no_grad():
            fake_samples = generator(fixed_noise, condition).cpu().numpy()

            # 如果提供了缩放器，反归一化样本
            if scaler is not None:
                fake_samples_reshaped = fake_samples.reshape(-1, 1)
                fake_samples_rescaled = scaler.inverse_transform(fake_samples_reshaped)
                fake_samples = fake_samples_rescaled.reshape(fake_samples.shape)

        # 绘制样本
        for j in range(num_samples):
            plt.subplot(len(months_to_compare), num_samples, i * num_samples + j + 1)
            plt.plot(fake_samples[j])
            plt.title(f"{month_names[month - 1]} 样本 {j + 1}")
            plt.xlabel("小时")
            plt.ylabel("功率")
            plt.grid(True)

    plt.suptitle("不同月份CGAN生成的光伏场景比较", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


# 5. 主函数模块
def main():
    # 设置参数
    data_path = "solar2013.csv"  # 数据路径
    batch_size = 16
    sequence_length = 24
    latent_dim = 100
    condition_dim = 12  # 12个月
    hidden_dim = 128
    learning_rate = 0.0002
    beta1 = 0.5
    num_epochs = 1500
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = "cgan_models"

    # 加载数据
    print("正在加载数据...")
    dataset = PhotovoltaicCGANDataset(data_path, sequence_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    print(f"数据加载完成，共有 {len(dataset)} 个日光伏曲线")

    # 创建CGAN模型
    generator = Generator(latent_dim, condition_dim, sequence_length, hidden_dim)
    discriminator = Discriminator(sequence_length, condition_dim, hidden_dim)

    # 训练模型
    print("开始训练CGAN模型...")
    generator, discriminator, losses = train_cgan(
        generator=generator,
        discriminator=discriminator,
        dataloader=dataloader,
        num_epochs=num_epochs,
        latent_dim=latent_dim,
        learning_rate=learning_rate,
        beta1=beta1,
        device=device,
        save_dir=save_dir
    )

    # 生成并可视化不同月份的样本
    print("使用CGAN生成不同月份的光伏场景...")
    for month in [1, 4, 7, 10]:  # 选择春夏秋冬四个季节代表月份
        samples = generate_samples_by_month(
            generator=generator,
            month=month,
            num_samples=9,
            latent_dim=latent_dim,
            device=device,
            scaler=dataset.scaler
        )
        visualize_generated_samples(samples, month)

    # 比较不同月份
    compare_months(
        generator=generator,
        months_to_compare=[1, 4, 7, 10],
        num_samples=3,
        latent_dim=latent_dim,
        device=device,
        scaler=dataset.scaler
    )

    print("CGAN光伏场景生成完成！")


if __name__ == "__main__":
    main()