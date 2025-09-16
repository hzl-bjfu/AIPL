#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
混合音频数据集生成器
生成包含2-5种鸟的10秒混合音频，支持背景音
"""

import os
import csv
import random
import numpy as np
import pandas as pd
from pathlib import Path
import soundfile as sf
import sys
from datetime import datetime

# 添加evascape目录到Python路径
evascape_dir = Path(__file__).parent / 'evascape' / 'Evascape'
sys.path.insert(0, str(evascape_dir))

# 导入evascape的工具函数
from evascape.Evascape.toolbox import waveread, addin, flatsound, bracket_ramp

class BirdMixDatasetGenerator:
    def __init__(self, bird_data_dir, output_dir, sample_rate=16000, background_dir=None):
        """
        初始化混合音频数据集生成器
        
        Parameters:
        -----------
        bird_data_dir : str
            鸟鸣声数据目录路径
        output_dir : str
            输出目录路径
        sample_rate : int
            采样率，默认16000Hz
        """
        self.bird_data_dir = Path(bird_data_dir)
        self.output_dir = Path(output_dir)
        self.sample_rate = sample_rate
        self.background_dir = Path(background_dir) if background_dir else None
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取所有鸟种
        self.bird_species = [d.name for d in self.bird_data_dir.iterdir() if d.is_dir()]
        print(f"发现 {len(self.bird_species)} 种鸟类: {self.bird_species}")
        
        # 构建鸟鸣声数据库
        self.bird_db = self._build_bird_database()
        
        # 背景音选项（若提供真实背景目录，则加入'real'）
        self.background_options = ['no_background', 'ambient_sound', 'rain', 'wind']
        self.real_background_files = []
        if self.background_dir and self.background_dir.exists():
            self.real_background_files = sorted(list(self.background_dir.rglob('*.wav')))
            if len(self.real_background_files) > 0:
                self.background_options.append('real')
        
    def _build_bird_database(self):
        """构建鸟鸣声数据库"""
        bird_data = []
        
        for species in self.bird_species:
            species_dir = self.bird_data_dir / species
            if species_dir.exists():
                audio_files = list(species_dir.glob("*.wav")) + list(species_dir.glob("*.mp3"))
                
                for audio_file in audio_files:
                    try:
                        # 读取音频文件获取时长
                        audio, sr = sf.read(str(audio_file))
                        duration = len(audio) / sr
                        
                        bird_data.append({
                            'species': species,
                            'filename': audio_file.name,
                            'fullfilename': str(audio_file),
                            'duration': duration,
                            'sample_rate': sr
                        })
                    except Exception as e:
                        print(f"警告: 无法读取文件 {audio_file}: {e}")
        
        return pd.DataFrame(bird_data)
    
    def _create_background_sound(self, background_type, duration):
        """创建背景音"""
        if background_type == 'no_background':
            return flatsound(val=0, d=duration, sr=self.sample_rate)
        elif background_type == 'real' and len(self.real_background_files) > 0:
            # 从真实背景列表中随机挑选一个背景文件并裁剪/填充到目标长度
            bg_path = random.choice(self.real_background_files)
            try:
                bg_audio, bg_sr = sf.read(str(bg_path))
                # 若是立体声，转单声道
                if isinstance(bg_audio[0], (list, tuple, np.ndarray)):
                    bg_audio = np.mean(np.array(bg_audio), axis=1)
                target_length = int(self.sample_rate * duration)
                # 简单重采样
                bg_audio = self._resample_audio(bg_audio, bg_sr, self.sample_rate, max(target_length, len(bg_audio)))
                # 随机起点裁剪/循环填充
                if len(bg_audio) < target_length:
                    reps = int(np.ceil(target_length / len(bg_audio)))
                    bg_audio = np.tile(bg_audio, reps)
                start_i = 0 if len(bg_audio) == target_length else random.randint(0, len(bg_audio) - target_length)
                bg_audio = bg_audio[start_i:start_i + target_length]
                # 背景音量降低
                return 0.2 * bg_audio
            except Exception:
                # 回退到合成环境音
                pass
        elif background_type == 'ambient_sound':
            # 创建环境音（低频噪声）
            noise = np.random.normal(0, 0.01, int(self.sample_rate * duration))
            # 添加低频成分
            t = np.linspace(0, duration, int(self.sample_rate * duration))
            low_freq = 0.1 * np.sin(2 * np.pi * 0.5 * t)  # 0.5Hz低频
            return noise + low_freq
        elif background_type == 'rain':
            # 创建雨声（高频噪声）
            noise = np.random.normal(0, 0.02, int(self.sample_rate * duration))
            # 添加高频成分
            t = np.linspace(0, duration, int(self.sample_rate * duration))
            high_freq = 0.05 * np.sin(2 * np.pi * 100 * t)  # 100Hz高频
            return noise + high_freq
        elif background_type == 'wind':
            # 创建风声（中频噪声）
            noise = np.random.normal(0, 0.015, int(self.sample_rate * duration))
            # 添加中频成分
            t = np.linspace(0, duration, int(self.sample_rate * duration))
            mid_freq = 0.08 * np.sin(2 * np.pi * 20 * t)  # 20Hz中频
            return noise + mid_freq
        else:
            return flatsound(val=0, d=duration, sr=self.sample_rate)
    
    def _resample_audio(self, audio, original_sr, target_sr, target_length):
        """重采样音频"""
        if original_sr != target_sr:
            # 简单的重采样实现
            if original_sr < target_sr:
                # 上采样：重复采样点
                ratio = target_sr // original_sr
                audio = np.repeat(audio, ratio)
            else:
                # 下采样：取每ratio个点中的一个
                ratio = original_sr // target_sr
                audio = audio[::ratio]
        
        # 调整长度
        if len(audio) > target_length:
            audio = audio[:target_length]
        elif len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        
        return audio
    
    def create_mixed_audio(self, n_species, duration=10, background_type='no_background', 
                          filename="mixed_audio.wav"):
        """
        创建混合音频
        
        Parameters:
        -----------
        n_species : int
            混合的鸟种数量
        duration : int
            音频时长（秒）
        background_type : str
            背景音类型
        filename : str
            输出文件名
            
        Returns:
        --------
        dict : 包含音频信息和元数据的字典
        """
        # 随机选择鸟种
        selected_species = random.sample(self.bird_species, min(n_species, len(self.bird_species)))
        
        # 创建基础音频向量（包含背景音）
        base_audio = self._create_background_sound(background_type, duration)
        
        # 为每种鸟选择音频文件
        selected_audio_files = []
        for species in selected_species:
            species_audio = self.bird_db[self.bird_db['species'] == species]
            if len(species_audio) > 0:
                # 随机选择该物种的一个音频文件
                selected_file = species_audio.sample(n=1).iloc[0]
                selected_audio_files.append(selected_file)
        
        # 混合音频（允许完全重叠）
        mixed_audio = base_audio.copy()
        bird_segments = []  # 记录每只鸟的时间段
        
        for i, audio_file in enumerate(selected_audio_files):
            try:
                # 读取音频文件
                audio, sr = sf.read(audio_file['fullfilename'])
                # 若是立体声，转单声道
                if isinstance(audio[0], (list, tuple, np.ndarray)):
                    audio = np.mean(np.array(audio), axis=1)
                
                # 随机截取源音频一段（0.5s 到 min(源长, duration)），增强多样性
                orig_len_sec = max(1e-6, len(audio) / float(sr))
                max_clip = min(orig_len_sec, duration)
                clip_dur = max(0.5, random.uniform(0.5, max_clip))
                if clip_dur > orig_len_sec:
                    clip_dur = orig_len_sec
                start_src = 0.0 if orig_len_sec == clip_dur else random.uniform(0, orig_len_sec - clip_dur)
                start_idx = int(start_src * sr)
                end_idx = int(start_idx + clip_dur * sr)
                sub_audio = audio[start_idx:end_idx]

                # 重采样到目标采样率，长度先保持子段长度对应的样本数上限
                target_len_any = int(self.sample_rate * clip_dur)
                sub_audio = self._resample_audio(sub_audio, sr, self.sample_rate, target_len_any)
                
                # 随机选择加入时间，允许完全重叠
                start_time = random.uniform(0, duration)
                
                # 若越界，进行裁剪以适配10s画布
                canvas_len = int(self.sample_rate * duration)
                start_sample = int(start_time * self.sample_rate)
                remain = max(0, canvas_len - start_sample)
                if remain <= 0:
                    # 起点在末尾之后，忽略该片段
                    continue
                if len(sub_audio) > remain:
                    sub_audio = sub_audio[:remain]
                end_time = start_time + len(sub_audio) / float(self.sample_rate)
                
                # 记录鸟的时间段
                bird_segments.append({
                    'species': audio_file['species'],
                    'filename': audio_file['filename'],
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': end_time - start_time
                })
                
                # 添加音频到混合中
                mixed_audio = addin(
                    base_sound=mixed_audio,
                    added_sound=sub_audio,
                    time_code=start_time,
                    ramp_duration=0.1,
                    sr=self.sample_rate
                )
                
            except Exception as e:
                print(f"  处理音频文件 {audio_file['filename']} 时出错: {e}")
        
        # 保存音频文件
        output_path = self.output_dir / filename
        sf.write(str(output_path), mixed_audio, self.sample_rate)
        
        # 收集元数据
        metadata = {
            'filename': filename,
            'species_list': selected_species,
            'n_species': len(selected_species),
            'duration': duration,
            'sample_rate': self.sample_rate,
            'background_type': background_type,
            'bird_segments': bird_segments
        }
        
        return metadata
    
    def generate_dataset(self, n_mix=100, min_species=2, max_species=5, duration=10, species_choices=None):
        """
        生成混合音频数据集
        
        Parameters:
        -----------
        n_mix : int
            要生成的混合音频数量
        min_species : int
            最少鸟种数量
        max_species : int
            最多鸟种数量
        duration : int
            音频时长（秒）
        """
        metadata_csv = self.output_dir / 'metadata.csv'
        
        # 打开CSV写入器
        with open(metadata_csv, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['filename', 'n_species', 'species_list', 'background_type', 
                         'duration', 'sample_rate', 'bird_segments']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            print(f"🔄 正在生成 {n_mix} 条混合音频...")
            
            for i in range(1, n_mix + 1):
                # 选择鸟种数量（支持指定集合）
                if species_choices:
                    n_species = random.choice(species_choices)
                else:
                    n_species = random.randint(min_species, max_species)
                
                # 随机选择背景音类型
                background_type = random.choice(self.background_options)
                
                filename = f'mix_{i:04d}.wav'
                
                # 创建混合音频
                metadata = self.create_mixed_audio(
                    n_species=n_species,
                    duration=duration,
                    background_type=background_type,
                    filename=filename
                )
                
                if metadata:
                    # 格式化鸟段信息
                    bird_segments_str = ';'.join([
                        f"{seg['species']}:{seg['start_time']:.2f}-{seg['end_time']:.2f}"
                        for seg in metadata['bird_segments']
                    ])
                    
                    # 写入CSV行
                    writer.writerow({
                        'filename': metadata['filename'],
                        'n_species': metadata['n_species'],
                        'species_list': ';'.join(metadata['species_list']),
                        'background_type': metadata['background_type'],
                        'duration': metadata['duration'],
                        'sample_rate': metadata['sample_rate'],
                        'bird_segments': bird_segments_str
                    })
                    
                    print(f"✅ [{i}/{n_mix}] 生成: {filename} ({metadata['n_species']}种鸟, {background_type})")
                else:
                    print(f"❌ [{i}/{n_mix}] 生成失败: {filename}")
        
        print("🎉 数据集生成完成！")
        print(f"📁 输出目录: {self.output_dir}")
        print(f"📄 元数据文件: {metadata_csv}")
        
        # 生成统计信息
        self._generate_statistics()
    
    def _generate_statistics(self):
        """生成数据集统计信息"""
        metadata_file = self.output_dir / 'metadata.csv'
        if metadata_file.exists():
            df = pd.read_csv(metadata_file)
            
            print("\n📊 数据集统计信息:")
            print(f"   总音频数量: {len(df)}")
            print(f"   平均鸟种数量: {df['n_species'].mean():.2f}")
            print(f"   鸟种数量分布:")
            for n in sorted(df['n_species'].unique()):
                count = len(df[df['n_species'] == n])
                print(f"     {n}种鸟: {count}个音频")
            
            print(f"   背景音类型分布:")
            for bg in df['background_type'].unique():
                count = len(df[df['background_type'] == bg])
                print(f"     {bg}: {count}个音频")

def main():
    """主函数"""
    # 参数配置
    bird_data_dir = 'data/data_2s_19_16000'        # 输入音频目录
    output_dir = 'bird_mix'                        # 输出目录
    background_dir = None                           # 可选：真实背景音目录路径，如 'backgrounds/'
    
    n_mix = 3000               # 生成音频数量
    min_species = 2            # 最少鸟种数量（备用）
    max_species = 5            # 最多鸟种数量（备用）
    species_choices = [3, 5, 7]  # 指定物种数集合
    duration = 10              # 音频时长（秒）
    sample_rate = 16000        # 采样率
    
    print("🎵 开始生成混合音频数据集...")
    print(f"📁 输入目录: {bird_data_dir}")
    print(f"📁 输出目录: {output_dir}")
    print(f"🎯 目标: {n_mix}个音频，物种数∈{species_choices}，{duration}秒")
    
    # 创建生成器
    generator = BirdMixDatasetGenerator(
        bird_data_dir=bird_data_dir,
        output_dir=output_dir,
        sample_rate=sample_rate,
        background_dir=background_dir
    )
    
    # 生成数据集
    generator.generate_dataset(
        n_mix=n_mix,
        min_species=min_species,
        max_species=max_species,
        duration=duration,
        species_choices=species_choices
    )

if __name__ == "__main__":
    main()

