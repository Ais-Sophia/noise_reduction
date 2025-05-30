import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import time
import os
import sounddevice as sd
import pywt

class SpeechDenoiser:
    def __init__(self, sr=16000, frame_length=512, hop_length=256, n_fft=1024):
        """
        语音去噪系统初始化
        参数:
            sr: 采样率
            frame_length: 帧长度
            hop_length: 帧移
            n_fft: FFT点数
        """
        self.sr = sr
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.window = np.hanning(frame_length)
        
    def load_audio(self, file_path):
        """加载音频文件"""
        y, sr = librosa.load(file_path, sr=self.sr)
        return y, sr
    
    def add_noise(self, clean_signal, snr_db):
        """
        添加高斯白噪声
        参数:
            clean_signal: 干净语音信号
            snr_db: 信噪比(dB)
        """
        # 计算信号功率
        signal_power = np.mean(clean_signal ** 2)
        
        # 计算噪声功率
        noise_power = signal_power / (10 ** (snr_db / 10))
        
        # 生成高斯白噪声
        noise = np.random.normal(0, np.sqrt(noise_power), len(clean_signal))
        
        # 添加噪声
        noisy_signal = clean_signal + noise
        
        return noisy_signal, noise
    
    def spectral_subtraction(self, noisy_signal, noise_profile=None, alpha=3.2, beta=0.0001):
        """
        频谱减法去噪
        参数:
            noisy_signal: 带噪语音信号
            noise_profile: 噪声模板
            alpha: 过减因子
            beta: 谱下限因子
        """
        # 确保信号长度是帧长的整数倍
        padded_length = ((len(noisy_signal) - self.frame_length) // self.hop_length + 1) * self.hop_length + self.frame_length
        padded_signal = np.pad(noisy_signal, (0, padded_length - len(noisy_signal)), mode='constant')
        
        # 分帧
        frames = librosa.util.frame(padded_signal, frame_length=self.frame_length, hop_length=self.hop_length)
        frames = frames * self.window[:, None]
        
        # 如果没有提供噪声模板，使用前5帧作为噪声估计
        if noise_profile is None:
            noise_frames = frames[:, :5]
            noise_mag = np.abs(np.fft.rfft(noise_frames, n=self.n_fft, axis=0))
            noise_profile = np.mean(noise_mag, axis=1, keepdims=True)
        
        # 处理各帧
        enhanced_frames = []
        for i in range(frames.shape[1]):
            frame = frames[:, i]
            
            # 计算频谱
            spec = np.fft.rfft(frame, n=self.n_fft)
            mag = np.abs(spec)
            phase = np.angle(spec)
            
            # 频谱减法
            enhanced_mag = np.maximum(mag - alpha * noise_profile.squeeze(), beta * noise_profile.squeeze())
            
            # 重建复数频谱
            enhanced_spec = enhanced_mag * np.exp(1j * phase)
            
            # 逆变换并截断到原始帧长度
            enhanced_frame = np.fft.irfft(enhanced_spec)[:self.frame_length]
            enhanced_frames.append(enhanced_frame)
        
        # 重构信号：重叠相加
        output_length = (frames.shape[1] - 1) * self.hop_length + self.frame_length
        output_signal = np.zeros(output_length)
        window_sumsq = np.zeros(output_length)
        window_sq = self.window ** 2
        
        for i, enhanced_frame in enumerate(enhanced_frames):
            start = i * self.hop_length
            end = start + self.frame_length
            
            # 确保帧长度匹配
            if len(enhanced_frame) < self.frame_length:
                # 如果帧太短，进行填充
                padded_frame = np.pad(enhanced_frame, (0, self.frame_length - len(enhanced_frame)))
            else:
                padded_frame = enhanced_frame[:self.frame_length]
                
            output_signal[start:end] += padded_frame * self.window
            window_sumsq[start:end] += window_sq
        
        # 归一化
        window_sumsq[window_sumsq < 1e-10] = 1.0
        enhanced = output_signal / window_sumsq
        
        # 裁剪到原始长度
        enhanced = enhanced[:len(noisy_signal)]
        
        return enhanced
    
    def wiener_filter(self, noisy_signal, noise_profile=None, iterations=5):
        """
        维纳滤波去噪
        参数:
            noisy_signal: 带噪语音信号
            noise_profile: 噪声模板
            iterations: 迭代次数
        """
        # 确保信号长度是帧长的整数倍
        padded_length = ((len(noisy_signal) - self.frame_length) // self.hop_length + 1) * self.hop_length + self.frame_length
        padded_signal = np.pad(noisy_signal, (0, padded_length - len(noisy_signal)), mode='constant')
        
        # 分帧
        frames = librosa.util.frame(padded_signal, frame_length=self.frame_length, hop_length=self.hop_length)
        frames = frames * self.window[:, None]
        
        # 如果没有提供噪声模板，使用前5帧作为噪声估计
        if noise_profile is None:
            noise_frames = frames[:, :5]
            noise_mag = np.abs(np.fft.rfft(noise_frames, n=self.n_fft, axis=0))
            noise_profile = np.mean(noise_mag, axis=1, keepdims=True)
        
        # 初始化语音功率谱估计
        speech_profile = noise_profile.copy()
        
        # 处理各帧
        enhanced_frames = []
        for i in range(frames.shape[1]):
            frame = frames[:, i]
            
            # 计算频谱
            spec = np.fft.rfft(frame, n=self.n_fft)
            mag = np.abs(spec)
            phase = np.angle(spec)
            
            # 维纳滤波增益
            wiener_gain = speech_profile / (speech_profile + noise_profile)
            
            # 应用增益
            enhanced_mag = mag * wiener_gain.squeeze()
            
            # 更新语音功率谱估计
            speech_profile = 0.9 * speech_profile + 0.1 * (enhanced_mag ** 2)[:, None]
            
            # 重建复数频谱
            enhanced_spec = enhanced_mag * np.exp(1j * phase)
            
            # 逆变换并截断到原始帧长度
            enhanced_frame = np.fft.irfft(enhanced_spec)[:self.frame_length]
            enhanced_frames.append(enhanced_frame)
        
        # 重构信号：重叠相加
        output_length = (frames.shape[1] - 1) * self.hop_length + self.frame_length
        output_signal = np.zeros(output_length)
        window_sumsq = np.zeros(output_length)
        window_sq = self.window ** 2
        
        for i, enhanced_frame in enumerate(enhanced_frames):
            start = i * self.hop_length
            end = start + self.frame_length
            
            # 确保帧长度匹配
            if len(enhanced_frame) < self.frame_length:
                # 如果帧太短，进行填充
                padded_frame = np.pad(enhanced_frame, (0, self.frame_length - len(enhanced_frame)))
            else:
                padded_frame = enhanced_frame[:self.frame_length]
                
            output_signal[start:end] += padded_frame * self.window
            window_sumsq[start:end] += window_sq
        
        # 归一化
        window_sumsq[window_sumsq < 1e-10] = 1.0
        enhanced = output_signal / window_sumsq
        
        # 裁剪到原始长度
        enhanced = enhanced[:len(noisy_signal)]
        
        return enhanced
    
    def wavelet_denoising(self, noisy_signal, wavelet='db4', level=4, threshold=0.1):
        """
        小波阈值去噪
        参数:
            noisy_signal: 带噪语音信号
            wavelet: 小波基类型
            level: 分解层数
            threshold: 阈值
        """
        # 小波分解
        coeffs = pywt.wavedec(noisy_signal, wavelet, level=level)
        
        # 阈值处理
        coeffs_thresh = [coeffs[0]]  # 保留近似系数
        for i in range(1, len(coeffs)):
            # 软阈值处理
            coeffs_thresh.append(pywt.threshold(coeffs[i], threshold * np.max(np.abs(coeffs[i])), mode='soft'))
        
        # 小波重构
        enhanced = pywt.waverec(coeffs_thresh, wavelet)
        
        # 确保长度一致
        min_length = min(len(noisy_signal), len(enhanced))
        return enhanced[:min_length]
    
    def calculate_snr(self, clean_signal, processed_signal):
        """计算信噪比(SNR)"""
        """       
        20dB:高质量语音
        10-20dB:可接受质量
        <10dB:严重噪声污染
        代码中默认使用5dB(高噪声环境)
        """

        # 确保信号长度一致
        min_length = min(len(clean_signal), len(processed_signal))
        clean_signal = clean_signal[:min_length]
        processed_signal = processed_signal[:min_length]
        
        # 计算信号功率
        signal_power = np.mean(clean_signal ** 2)
        print(f"信号功率: {signal_power:.4f}")
        # 计算噪声功率
        noise = processed_signal - clean_signal
        noise_power = np.mean(noise ** 2)
        print(f"噪声功率: {noise_power:.4f}")
        # 避免除以零
        if noise_power == 0:
            return float('inf')
        
        # 计算SNR(dB)
        snr = 10 * np.log10(signal_power / noise_power)
        return snr
    
    def plot_signals(self, clean, noisy, enhanced, title="", save_path=None):
        """绘制信号波形图"""
        # 确保信号长度一致
        min_length = min(len(clean), len(noisy), len(enhanced))
        clean = clean[:min_length]
        noisy = noisy[:min_length]
        enhanced = enhanced[:min_length]
        
        plt.figure(figsize=(15, 10))
        
        # 干净信号
        plt.subplot(3, 1, 1)
        plt.plot(clean)
        plt.title("干净信号")
        plt.xlabel("样本")
        plt.ylabel("幅度")
        
        # 带噪信号
        plt.subplot(3, 1, 2)
        plt.plot(noisy)
        plt.title(f"带噪信号 (SNR: {self.calculate_snr(clean, noisy):.2f} dB)")
        plt.xlabel("样本")
        plt.ylabel("幅度")
        
        # 去噪信号
        plt.subplot(3, 1, 3)
        plt.plot(enhanced)
        plt.title(f"{title} (SNR Improvement: {self.calculate_snr(clean, enhanced) - self.calculate_snr(clean, noisy):.2f} dB)")
        plt.xlabel("样本")
        plt.ylabel("幅度")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def play_audio(self, signal, sr=None):
        """播放音频"""
        if sr is None:
            sr = self.sr
        sd.play(signal, sr)
        sd.wait()  # 等待播放完成
    
    def save_audio(self, signal, path, sr=None):
        """保存音频文件"""
        if sr is None:
            sr = self.sr
        sf.write(path, signal, sr)
    
    def compare_methods(self, clean_signal, noisy_signal, snr_db):
        """比较不同去噪方法性能"""
        # 计算原始SNR
        original_snr = self.calculate_snr(clean_signal, noisy_signal)
        
        # 应用不同去噪方法
        print("正在使用频谱减法处理...")
        start_time = time.time()
        ss_enhanced = self.spectral_subtraction(noisy_signal)
        ss_time = time.time() - start_time
        ss_snr = self.calculate_snr(clean_signal, ss_enhanced)
        
        print("正在使用维纳滤波处理...")
        start_time = time.time()
        wf_enhanced = self.wiener_filter(noisy_signal)
        wf_time = time.time() - start_time
        wf_snr = self.calculate_snr(clean_signal, wf_enhanced)
        
        print("正在使用小波去噪处理...")
        start_time = time.time()
        wt_enhanced = self.wavelet_denoising(noisy_signal)
        wt_time = time.time() - start_time
        wt_snr = self.calculate_snr(clean_signal, wt_enhanced)
        
        # 打印结果
        print("\n" + "="*60)
        print(f"去噪方法性能比较 (输入SNR: {original_snr:.2f} dB)")
        print("="*60)
        print(f"{'方法':<20} {'输出SNR(dB)':<15} {'SNR提升(dB)':<15} {'处理时间(s)':<15}")
        print("-"*60)
        print(f"{'原始带噪信号':<20} {original_snr:<15.2f} {'-':<15} {'-':<15}")
        print(f"{'频谱减法':<20} {ss_snr:<15.2f} {ss_snr - original_snr:<15.2f} {ss_time:<15.4f}")
        print(f"{'维纳滤波':<20} {wf_snr:<15.2f} {wf_snr - original_snr:<15.2f} {wf_time:<15.4f}")
        print(f"{'小波去噪':<20} {wt_snr:<15.2f} {wt_snr - original_snr:<15.2f} {wt_time:<15.4f}")
        print("="*60)
        
        # 绘制波形图
        self.plot_signals(clean_signal, noisy_signal, ss_enhanced, "频谱相减增强")
        self.plot_signals(clean_signal, noisy_signal, wf_enhanced, "维纳滤波器增强")
        self.plot_signals(clean_signal, noisy_signal, wt_enhanced, "小波去噪增强")

        return {
            "spectral_subtraction": (ss_enhanced, ss_snr, ss_time),
            "wiener_filter": (wf_enhanced, wf_snr, wf_time),
            "wavelet_denoising": (wt_enhanced, wt_snr, wt_time)
        }

# 使用示例
if __name__ == "__main__":
    # 初始化去噪系统
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

    denoiser = SpeechDenoiser(sr=16000, frame_length=512, hop_length=256, n_fft=1024)
    
    # 加载干净语音
    print("正在加载音频文件...")
    clean_signal, sr = denoiser.load_audio("IKUN.wav")
    
    # 添加噪声 (SNR=5dB)
    snr_db = 5
    print(f"正在添加高斯白噪声 (SNR={snr_db}dB)...")
    noisy_signal, noise = denoiser.add_noise(clean_signal, snr_db)
    
    # 保存带噪语音
    denoiser.save_audio(noisy_signal, "noisy_audio.wav")
    
    # 比较不同去噪方法
    print("开始比较不同去噪方法...")
    results = denoiser.compare_methods(clean_signal, noisy_signal, snr_db)
    
    # 保存去噪结果
    denoiser.save_audio(results["spectral_subtraction"][0], "ss_enhanced.wav")
    denoiser.save_audio(results["wiener_filter"][0], "wf_enhanced.wav")
    denoiser.save_audio(results["wavelet_denoising"][0], "wt_enhanced.wav")
    
    # 播放音频
    print("\n播放原始干净语音...")
    denoiser.play_audio(clean_signal)
    
    print("\n播放带噪语音...")
    denoiser.play_audio(noisy_signal)
    
    print("\n播放频谱减法去噪结果...")
    denoiser.play_audio(results["spectral_subtraction"][0])
    
    print("\n播放维纳滤波去噪结果...")
    denoiser.play_audio(results["wiener_filter"][0])
    
    print("\n播放小波去噪结果...")
    denoiser.play_audio(results["wavelet_denoising"][0])