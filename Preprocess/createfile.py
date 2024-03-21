import os

# 大文件夹路径
base_dir = '/home/hzl/code/data'

# 定义小文件夹名称
folder_names = ['anchun', 'anlvliuying', 'cangying', 'datiane', 'haiou', 'heiqinji', 'hongsun', 'huanjingzhi', 'huihe', 'lixingliaomei', 'queying', 'xiangsiniao', 'yanleiniao', 'zhangerxiao', 'zhongbiaoyv', 'zhonghuazhegu']

# 创建小文件夹
for folder_name in folder_names:
    folder_path = os.path.join(base_dir, folder_name)
    os.makedirs(folder_path)

print("16小文件夹已创建")
