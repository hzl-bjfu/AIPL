import os
import csv

# 设置文件夹路径
folder_path = '/home/hzl/code/investigation-phase-master/scripts/data'

# 设置CSV文件路径
csv_file_path = '/home/hzl/code/investigation-phase-master/scripts/csv_pkl/data.csv'

# 创建CSV文件并写入列头
with open(csv_file_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['primary_label', 'filename'])

    # 遍历data文件夹中的每个小文件夹
    for subfolder in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder)
        
        # 检查是否是文件夹
        if os.path.isdir(subfolder_path):
            # 遍历小文件夹中的每个音频文件
            for audio_file in os.listdir(subfolder_path):
                # 检查是否是.wav文件
                if audio_file.endswith('.wav'):
                    # 获取文件名
                    filename = os.path.join(subfolder, audio_file)
                    # 将数据写入CSV文件
                    writer.writerow([subfolder, filename])
