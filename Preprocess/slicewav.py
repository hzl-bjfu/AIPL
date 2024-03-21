from pydub import AudioSegment
import os

def split_audio(input_path, output_path, split_length=1000):
    # 读取音频文件
    audio = AudioSegment.from_wav(input_path)

    # 确保输出路径存在
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # 获取文件名（不包括扩展名）
    file_name = os.path.splitext(os.path.basename(input_path))[0]

    # 分割音频
    for i, start_time in enumerate(range(0, len(audio), split_length)):
        end_time = start_time + split_length
        split_audio = audio[start_time:end_time]

        # 保存分割后的音频
        split_file_name = f"{file_name}_{i+1}.wav"
        split_file_path = os.path.join(output_path, split_file_name)
        split_audio.export(split_file_path, format="wav")

# 大文件夹路径
data3_folder = "/home/hzl/code/data3"  # 更改为你的data3文件夹路径

# 遍历每个小文件夹
for subfolder in os.listdir(data3_folder):
    subfolder_path = os.path.join(data3_folder, subfolder)

    # 遍历每个音频文件
    for audio_file in os.listdir(subfolder_path):
        if audio_file.endswith(".wav"):
            audio_file_path = os.path.join(subfolder_path, audio_file)

            # 创建新文件夹用于保存分割后的音频
            new_folder_path = os.path.join(subfolder_path, "new")
            if not os.path.exists(new_folder_path):
                os.makedirs(new_folder_path)

            # 分割并保存音频
            split_audio(audio_file_path, new_folder_path)
