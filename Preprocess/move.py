import os
import shutil

# 源文件夹路径，包含音频文件的文件夹
#source_folder = '/home/hzl/code/resampledata/zhonghuazhegu/new'

# 目标文件夹路径，音频文件将被移动到这里
#destination_folder = '/home/hzl/code/data/zhonghuazhegu'

# 遍历源文件夹中的所有文件
#for filename in os.listdir(source_folder):
    #if filename.endswith('.wav'):  # 这里假设音频文件都是.wav格式的，根据实际情况修改
        #source_file = os.path.join(source_folder, filename)
        #destination_file = os.path.join(destination_folder, filename)

        # 使用shutil.move()将文件从源文件夹移动到目标文件夹
        #shutil.move(source_file, destination_file)

#print("音频文件移动完成。")
import os
import shutil

# 大文件夹路径
data_folder = "/home/hzl/code/data3_1s"  # 更改为你的data文件夹路径
data1_folder = "/home/hzl/code/data3"  # 更改为你的data1文件夹路径

# 获取data1文件夹下的所有小文件夹名字
data1_subfolders = [f for f in os.listdir(data1_folder) if os.path.isdir(os.path.join(data1_folder, f))]

# 遍历data1文件夹下的每个小文件夹
for subfolder in data1_subfolders:
    data1_new_folder = os.path.join(data1_folder, subfolder, 'new')
    data_folder_new = os.path.join(data_folder, subfolder)

    # 检查data文件夹对应的小文件夹是否存在，如果不存在则创建
    if not os.path.exists(data_folder_new):
        os.makedirs(data_folder_new)

    # 复制data1文件夹下的new文件夹内容到data文件夹下的对应小文件夹
    for root, dirs, files in os.walk(data1_new_folder):
        for file in files:
            src_path = os.path.join(root, file)
            dst_path = os.path.join(data_folder_new, file)
            shutil.copyfile(src_path, dst_path)

print("数据已成功复制到data文件夹下的对应小文件夹中。")
