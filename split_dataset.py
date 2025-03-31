import os
import random
import shutil
from pathlib import Path

def create_dirs(dir_list):
    """创建目录"""
    for dir_path in dir_list:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def split_dataset(image_dir, label_dir, train_ratio=0.8, val_ratio=0.2):
    """
    划分数据集为训练集和验证集
    
    Args:
        image_dir: 图片目录
        label_dir: 标签目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
    """
    # 获取所有图片文件名（不含扩展名）
    image_files = [f.stem for f in Path(image_dir).glob('*.*')]
    random.shuffle(image_files)
    
    # 计算每个集合的大小
    total = len(image_files)
    train_size = int(total * train_ratio)
    
    # 划分数据集
    train_files = image_files[:train_size]
    val_files = image_files[train_size:]
    
    # 创建目录结构
    splits = ['train', 'val']
    for split in splits:
        create_dirs([
            f'dataset/{split}/images',
            f'dataset/{split}/labels'
        ])
    
    # 移动文件
    for split, files in zip(splits, [train_files, val_files]):
        for file in files:
            # 移动图片
            src_img = f'{image_dir}/{file}.*'
            dst_img = f'dataset/{split}/images/'
            for img in Path(image_dir).glob(f'{file}.*'):
                shutil.copy2(img, dst_img)
            
            # 移动标签
            src_label = f'{label_dir}/{file}.txt'
            dst_label = f'dataset/{split}/labels/'
            if os.path.exists(src_label):
                shutil.copy2(src_label, dst_label)
    
#     # 更新data.yaml
#     yaml_content = f"""path: dataset  # dataset root dir
# train: train/images  # train images (relative to 'path')
# val: val/images  # val images (relative to 'path')

# # Classes
# names:
#   0: person
#   1: car
#   2: truck
#   3: bus
#   4: motorcycle
#   5: bicycle
#   6: other
# """
    
#     with open('dataset/data.yaml', 'w', encoding='utf-8') as f:
#         f.write(yaml_content)
    
    print(f"数据集划分完成！")
    print(f"训练集: {len(train_files)} 个文件")
    print(f"验证集: {len(val_files)} 个文件")

if __name__ == '__main__':
    # 设置随机种子以确保结果可复现
    random.seed(42)
    
    # 划分数据集
    split_dataset(
        image_dir='dataset/images',
        label_dir='dataset/labels',
        train_ratio=0.8,
        val_ratio=0.2
    ) 