import os

# 定义数据文件夹路径
data_folder = "/home/hzl/code/investigation-phase-master/scripts/data"

# 获取数据文件夹中的所有子文件夹
sub_folders = [folder for folder in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, folder))]

# 遍历每个子文件夹并统计音频文件数量
for folder in sub_folders:
    folder_path = os.path.join(data_folder, folder)
    audio_files = [file for file in os.listdir(folder_path) if file.endswith('.wav')]
    audio_count = len(audio_files)
    print(f"文件夹 {folder} 中有 {audio_count} 个音频文件。")
