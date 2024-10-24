import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from numpy.linalg import svd
from sklearn.metrics import mean_squared_error
import os


def load_ecg5000():
    """
    从UCI机器学习库加载ECG5000数据集。

    返回:
    X: 信号数据，形状为 (5000, 140)
    y: 标签数据，形状为 (5000,)
    """
    try:
        # 使用fetch_openml加载数据
        ecg = fetch_openml(name='ECG5000', version=1, as_frame=True)
        data = ecg.frame
        # 检查数据集的列数
        if data.shape[1] < 141:
            print(f"警告：数据集列数为{data.shape[1]}，预期为141。可能存在数据缺失。")
        # 数据集的前140列是信号，最后一列是标签
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
        return X, y
    except Exception as e:
        print(f"加载ECG5000数据集时出错: {e}")
        return None, None


def add_realistic_noise(X, noise_level=0.5):
    """
    模拟真实噪声，通过在信号中加入随机噪声。

    参数:
    X: 原始信号数据，形状为 (n_signals, n_samples)
    noise_level: 噪声标准差

    返回:
    noisy_signals: 添加噪声后的信号，形状为 (n_signals, n_samples)
    """
    noise = np.random.normal(0, noise_level, X.shape)
    noisy_signals = X + noise
    return noisy_signals


def svd_denoise_matrix(data_matrix, num_components):
    """
    使用SVD对数据矩阵进行去噪。

    参数:
    data_matrix: 形状为 (n_signals, n_samples) 的数据矩阵
    num_components: 保留的奇异值数量

    返回:
    denoised_matrix: 去噪后的数据矩阵
    """
    # 进行奇异值分解
    U, S, VT = svd(data_matrix, full_matrices=False)

    # 保留前num_components个奇异值
    S_denoised = np.copy(S)
    S_denoised[num_components:] = 0  # 将较小的奇异值设为0

    # 构建对角矩阵
    S_denoised_matrix = np.diag(S_denoised)

    # 重构去噪后的数据矩阵
    denoised_matrix = np.dot(U, np.dot(S_denoised_matrix, VT))
    return denoised_matrix


def select_num_components(S, threshold=0.50):
    """
    自动选择保留的奇异值数量，使得累计能量达到阈值。

    参数:
    S: 奇异值数组
    threshold: 累计能量阈值（默认50%）

    返回:
    num_components: 需要保留的奇异值数量
    """
    cumulative_energy = np.cumsum(S) / np.sum(S)
    num_components = np.searchsorted(cumulative_energy, threshold) + 1
    return num_components


def calculate_snr(clean_signal, noisy_signal):
    """
    计算信噪比（SNR）。

    参数:
    clean_signal: 原始干净信号
    noisy_signal: 带噪声的信号

    返回:
    SNR值（分贝）
    """
    power_signal = np.mean(clean_signal ** 2)
    power_noise = np.mean((noisy_signal - clean_signal) ** 2)
    snr = 10 * np.log10(power_signal / power_noise)
    return snr


def main():
    # 设置随机种子以保证结果可重复
    np.random.seed(42)

    # 加载数据
    X, y = load_ecg5000()
    if X is None or y is None:
        print("数据加载失败，程序终止。")
        return
    print(f"加载的ECG5000数据集形状: {X.shape}")

    # 检查数据集是否完整
    expected_signals = 5000
    actual_signals = X.shape[0]
    if actual_signals < expected_signals:
        print(f"警告：加载的信号数量为{actual_signals}，预期为{expected_signals}。请检查数据加载过程。")

    # 选择部分信号进行处理
    num_signals = 100  # 选择100个信号
    if actual_signals < num_signals:
        print(f"警告：数据集中信号数量少于{num_signals}，将选择所有信号。")
        num_signals = actual_signals
    selected_signals = X[:num_signals]

    # 添加噪声
    noise_level = 0.5  # 噪声标准差
    noisy_signals = add_realistic_noise(selected_signals, noise_level)

    # 数据预处理：中心化
    # 对每个信号进行中心化（减去均值）
    selected_signals_centered = selected_signals - np.mean(selected_signals, axis=1, keepdims=True)
    noisy_signals_centered = noisy_signals - np.mean(noisy_signals, axis=1, keepdims=True)

    # 数据预处理：标准化
    selected_signals_standardized = selected_signals_centered / np.std(selected_signals_centered, axis=1, keepdims=True)
    noisy_signals_standardized = noisy_signals_centered / np.std(noisy_signals_centered, axis=1, keepdims=True)

    # 计算去噪前的SNR和MSE
    snr_noisy_list = [calculate_snr(selected_signals_standardized[i], noisy_signals_standardized[i]) for i in
                      range(num_signals)]
    mse_noisy_list = [mean_squared_error(selected_signals_standardized[i], noisy_signals_standardized[i]) for i in
                      range(num_signals)]

    average_snr_noisy = np.mean(snr_noisy_list)
    average_mse_noisy = np.mean(mse_noisy_list)

    print(f"平均带噪声信号的SNR: {average_snr_noisy:.2f} dB")
    print(f"平均带噪声信号的MSE: {average_mse_noisy:.4f}")

    # 使用SVD进行去噪
    # 首先对数据矩阵进行SVD
    U, S, VT = svd(noisy_signals_standardized, full_matrices=False)

    # 绘制奇异值谱（Scree Plot）以帮助选择奇异值数量
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(S) + 1), S, 'o-', color='purple')
    plt.axhline(y=S[0] * 0.1, color='red', linestyle='--', label='10% of first singular value')
    plt.title('Scree Plot of Singular Values')
    plt.xlabel('Singular Value Index')
    plt.ylabel('Singular Value Magnitude')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 自动选择保留的奇异值数量
    threshold = 0.50  # 保留50%的能量
    num_components = select_num_components(S, threshold)
    print(f"保留的奇异值数量: {num_components}（累计能量达到{threshold * 100}%)")

    # 去噪
    denoised_matrix = svd_denoise_matrix(noisy_signals_standardized, num_components)

    # 计算去噪后的SNR和MSE
    snr_denoised_list = [calculate_snr(selected_signals_standardized[i], denoised_matrix[i]) for i in
                         range(num_signals)]
    mse_denoised_list = [mean_squared_error(selected_signals_standardized[i], denoised_matrix[i]) for i in
                         range(num_signals)]

    average_snr_denoised = np.mean(snr_denoised_list)
    average_mse_denoised = np.mean(mse_denoised_list)

    print(f"平均去噪后信号的SNR: {average_snr_denoised:.2f} dB")
    print(f"平均去噪后信号的MSE: {average_mse_denoised:.4f}")

    # 可视化结果
    plt.figure(figsize=(18, 12))

    # 选择几个示例信号进行展示
    num_examples = 5
    example_indices = np.random.choice(num_signals, num_examples, replace=False)

    for idx, signal_idx in enumerate(example_indices):
        clean = selected_signals_standardized[signal_idx]
        noisy = noisy_signals_standardized[signal_idx]
        denoised = denoised_matrix[signal_idx]

        plt.subplot(num_examples, 3, idx * 3 + 1)
        plt.plot(clean, color='blue')
        plt.title(f"Signal {signal_idx + 1} - Clean")
        plt.xlabel('Time Point')
        plt.ylabel('Amplitude')
        plt.grid(True)

        plt.subplot(num_examples, 3, idx * 3 + 2)
        plt.plot(noisy, color='orange')
        plt.title(f"Signal {signal_idx + 1} - Noisy")
        plt.xlabel('Time Point')
        plt.ylabel('Amplitude')
        plt.grid(True)

        plt.subplot(num_examples, 3, idx * 3 + 3)
        plt.plot(denoised, color='green')
        plt.title(f"Signal {signal_idx + 1} - Denoised")
        plt.xlabel('Time Point')
        plt.ylabel('Amplitude')
        plt.grid(True)

    plt.tight_layout()
    plt.show()

    # 绘制Scree Plot已在前面

    # 绘制SNR提升直方图
    plt.figure(figsize=(10, 6))
    plt.hist(snr_denoised_list, bins=20, alpha=0.7, label='Denoised SNR', color='green')
    plt.hist(snr_noisy_list, bins=20, alpha=0.7, label='Noisy SNR', color='orange')
    plt.title('SNR Improvement')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Number of Signals')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 绘制MSE降低直方图
    plt.figure(figsize=(10, 6))
    plt.hist(mse_denoised_list, bins=20, alpha=0.7, label='Denoised MSE', color='green')
    plt.hist(mse_noisy_list, bins=20, alpha=0.7, label='Noisy MSE', color='orange')
    plt.title('MSE Reduction')
    plt.xlabel('MSE')
    plt.ylabel('Number of Signals')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
