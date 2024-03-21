from pydub import AudioSegment
import os

# 指定数据集文件夹的路径
data_folder = '/home/hzl/code/data/data_2s_20_44100'

# 指定目标采样率
target_sample_rate = 16000

# 创建目标文件夹用于存储处理后的音频文件
target_data_folder = 'data1'
os.makedirs(target_data_folder, exist_ok=True)

# 遍历每个文件夹
for bird_folder in os.listdir(data_folder):
    bird_folder_path = os.path.join(data_folder, bird_folder)
    
    # 创建目标文件夹，用于存储降采样后的音频和改为单通道后的音频
    target_bird_folder_path = os.path.join(target_data_folder, bird_folder)
    os.makedirs(target_bird_folder_path, exist_ok=True)
    
    # 遍历每个.wav文件
    for audio_file in os.listdir(bird_folder_path):
        if audio_file.endswith('.wav'):
            audio_file_path = os.path.join(bird_folder_path, audio_file)
            
            # 读取原始音频文件
            audio = AudioSegment.from_file(audio_file_path)
            
            # 将音频改为单通道
            audio = audio.set_channels(1)
            
            # 降采样到目标采样率
            audio = audio.set_frame_rate(target_sample_rate)
            
            # 保存处理后的音频文件到目标文件夹
            target_audio_file_path = os.path.join(target_bird_folder_path, audio_file)
            audio.export(target_audio_file_path, format="wav")

print("音频数据处理完成。")
