import os

# 大文件夹路径
base_folder = "/home/hzl/code/data3_1s"  # 更改为你的大文件夹路径

# 你指定的文件夹名字列表
folder_names = ["0009", "0017", "0034", "0036", "0074", 
                "0077", "0114", "0121", "0180", "0202", 
                "0235", "0257", "0265", "0281", "0298", 
                "0300", "0364", "0368", "0370", "1331"]

# 创建文件夹
for folder_name in folder_names:
    folder_path = os.path.join(base_folder, folder_name)

    # 检查文件夹是否存在，如果不存在则创建
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

print("指定名字的20个小文件夹已创建完成。")
