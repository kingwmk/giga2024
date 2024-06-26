import os
import shutil
import random

def copy_random_files(src_dir, dest_dir, percentage=0.02):
    # 获取源目录中的所有文件
    all_files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]
    
    # 计算要复制的文件数量
    num_files_to_copy = int(len(all_files) * percentage)
    
    # 随机选择文件
    files_to_copy = random.sample(all_files, num_files_to_copy)
    
    # 确保目标目录存在
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    # 复制文件
    for file_name in files_to_copy:
        src_file = os.path.join(src_dir, file_name)
        dest_file = os.path.join(dest_dir, file_name)
        shutil.copy2(src_file, dest_file)
        print(f"Copied {src_file} to {dest_file}")

# 设置源目录和目标目录
src_directory = "/mnt/home/data/giga2024/Trajectory/train/transformer_preprocess_files/"
dest_directory = "/mnt/home/data/giga2024/Trajectory/train/transformer_preprocess_files_val/"

# 调用函数复制文件
copy_random_files(src_directory, dest_directory)
