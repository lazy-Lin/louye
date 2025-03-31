import os
from pathlib import Path
import shutil
from tqdm import tqdm

def convert_labels():
    # 创建备份目录
    backup_dir = Path('dataset/labels_backup')
    backup_dir.mkdir(exist_ok=True)
    
    # 处理所有数据集分割
    splits = ['train', 'val', 'test']
    total_files = 0
    converted_files = 0
    
    for split in splits:
        print(f"\n处理{split}集...")
        label_dir = Path(f"dataset/{split}/labels")
        
        if not label_dir.exists():
            print(f"警告: {split}集标签目录不存在: {label_dir}")
            continue
        
        # 创建对应的备份目录
        split_backup = backup_dir / split
        split_backup.mkdir(exist_ok=True)
        
        # 获取所有标签文件
        label_files = list(label_dir.glob('*.txt'))
        total_files += len(label_files)
        
        # 使用tqdm显示进度
        for label_file in tqdm(label_files, desc=f"转换{split}集标签"):
            # 备份原始文件
            backup_file = split_backup / label_file.name
            shutil.copy2(label_file, backup_file)
            
            # 读取并转换标签
            with open(label_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # 转换标签
            converted_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 1:
                    # 将类别ID转换为0
                    parts[0] = '0'
                    converted_lines.append(' '.join(parts) + '\n')
            
            # 写回转换后的标签
            with open(label_file, 'w', encoding='utf-8') as f:
                f.writelines(converted_lines)
            
            converted_files += 1
    
    print(f"\n转换完成:")
    print(f"总文件数: {total_files}")
    print(f"已转换文件数: {converted_files}")
    print(f"备份目录: {backup_dir}")
    print("\n注意: 原始标签文件已备份到 dataset/labels_backup 目录")

if __name__ == '__main__':
    # 询问用户是否继续
    response = input("此操作会将所有标签文件的类别ID转换为0，并创建备份。是否继续？(y/n): ")
    if response.lower() == 'y':
        convert_labels()
    else:
        print("操作已取消") 