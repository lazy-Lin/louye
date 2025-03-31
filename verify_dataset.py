import os
import glob
from pathlib import Path
import yaml

def verify_dataset():
    # 读取数据配置文件
    with open('dataset/data.yaml', 'r', encoding='utf-8') as f:
        data_yaml = yaml.safe_load(f)
    
    # 检查类别数量
    num_classes = len(data_yaml['names'])
    print(f"数据集类别数量: {num_classes}")
    print("类别列表:")
    for idx, name in data_yaml['names'].items():
        print(f"  {idx}: {name}")
    
    # 检查数据集目录结构
    splits = ['train', 'val', 'test']
    for split in splits:
        print(f"\n检查{split}集:")
        # 检查图片目录
        img_dir = Path(f"dataset/{split}/images")
        if not img_dir.exists():
            print(f"警告: {split}集图片目录不存在: {img_dir}")
            continue
        
        # 检查标签目录
        label_dir = Path(f"dataset/{split}/labels")
        if not label_dir.exists():
            print(f"警告: {split}集标签目录不存在: {label_dir}")
            continue
        
        # 统计文件数量
        img_files = list(img_dir.glob('*.*'))
        label_files = list(label_dir.glob('*.txt'))
        
        print(f"图片数量: {len(img_files)}")
        print(f"标签数量: {len(label_files)}")
        
        # 检查标签文件
        if len(label_files) > 0:
            # 检查第一个标签文件的内容
            with open(label_files[0], 'r', encoding='utf-8') as f:
                first_label = f.readline().strip()
                print(f"标签文件示例: {first_label}")
            
            # 检查所有标签文件中的类别编号
            invalid_labels = []
            for label_file in label_files:
                with open(label_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        class_id = int(line.split()[0])
                        if class_id >= num_classes:
                            invalid_labels.append(f"{label_file}: {line.strip()}")
            
            if invalid_labels:
                print("警告: 发现无效的类别编号:")
                for label in invalid_labels[:5]:  # 只显示前5个
                    print(f"  {label}")
                if len(invalid_labels) > 5:
                    print(f"  ... 还有 {len(invalid_labels)-5} 个")

if __name__ == '__main__':
    verify_dataset() 