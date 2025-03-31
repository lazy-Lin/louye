from ultralytics import YOLO
import torch
from datetime import datetime
import os
from pathlib import Path
# 在训练脚本中添加验证
from ultralytics.nn.modules.head import Segment
print(Segment)  # 应输出类定义信息

def get_next_exp_name(base_name="exp_small_objects"):
    """
    生成下一个实验名称，格式为 exp_small_objects_v1, exp_small_objects_v2 等
    
    Args:
        base_name (str): 基础实验名称
        
    Returns:
        str: 下一个可用的实验名称
    """
    # 获取runs/train目录下的所有文件夹
    train_dir = Path('runs/train')
    if not train_dir.exists():
        return f"{base_name}_v1"
        
    # 获取所有匹配的文件夹
    existing_dirs = [d for d in train_dir.iterdir() if d.is_dir() and d.name.startswith(base_name)]
    
    if not existing_dirs:
        return f"{base_name}_v1"
        
    # 提取版本号并找出最大值
    versions = []
    for d in existing_dirs:
        try:
            version = int(d.name.split('_v')[-1])
            versions.append(version)
        except (ValueError, IndexError):
            continue
            
    if not versions:
        return f"{base_name}_v1"
        
    next_version = max(versions) + 1
    return f"{base_name}_v{next_version}"

def train():
    # 硬件环境检查
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        device = 0  # 使用GPU
    else:
        print("No GPU available, using CPU")
        device = 'cpu'  # 使用CPU

    # 实验名称 - 自动递增版本号
    exp_name = get_next_exp_name()

    # 加载预训练模型
    model = YOLO('ultralytics/cfg/models/11/yolo11l-seg.yaml')  # 使用官方预训练模型

    # 针对小目标优化的训练配置
    results = model.train(
        # 数据配置
        data='./dataset/data.yaml',
        # data='dataset/data.yaml',
        epochs=100,                # 增加训练轮数
        multi_scale=True,         # 启用多尺度训练
        imgsz=800,                # 降低图像分辨率以减少内存使用
        batch=2,                  # 减小批次大小以减少内存使用
        workers=2,                # 减少工作进程数
        rect=False,               # 关闭矩形训练以增加随机性
        overlap_mask=True,
        mask_ratio=4,             # 降低掩码比例以减少内存使用
        single_cls=True,
        resume=False,
        project='runs/train',
        name=exp_name,

        # 优化器配置 - 针对小目标的学习率策略
        device=device,            # 根据硬件自动选择设备
        optimizer='AdamW',
        lr0=0.0002,               # 降低初始学习率
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,      # 增加权重衰减以防止过拟合
        warmup_epochs=15,         # 增加预热轮数
        
        # 损失权重调整 - 增强对小目标的关注
        box=7.5,                  # 增加边界框权重
        cls=0.7,                  # 保持分类权重
        dfl=1.5,
        pose=18.0,
        kobj=1.0,
        
        # 数据增强优化 - 针对小目标的数据增强策略
        hsv_h=0.015,
        hsv_s=0.7,                # 降低饱和度扰动
        hsv_v=0.4,                # 降低亮度变化
        degrees=5.0,              # 减小旋转角度
        translate=0.2,            # 减小平移范围
        scale=0.6,                # 减小缩放范围
        shear=3.0,                # 减小剪切角度
        perspective=0.0,          # 禁用透视变换以避免除零警告
        flipud=0.3,               # 减小上下翻转概率
        fliplr=0.5,
        mosaic=0.95,              # 增加马赛克概率
        mixup=0.3,                # 增加混合概率
        copy_paste=0.3,           # 降低复制粘贴概率
        auto_augment='randaugment',
        erasing=0.1,              # 降低随机擦除概率
        
        # 训练控制优化
        patience=50,              # 增加早停耐心值
        cos_lr=True,
        close_mosaic=40,          # 延迟关闭马赛克增强
        amp=True,                 # 启用混合精度训练
        fraction=1.0,
        profile=False,
        freeze=None,
        
        # IoU与NMS阈值调整 - 优化小目标检测
        iou=0.25,                 # 降低IoU阈值以更容易检测小目标
        conf=0.2,                 # 降低置信度阈值
        max_det=300,              # 减少最大检测数量
        nms=True,
        
        # 其他参数优化
        deterministic=True,
        seed=3407,
        save=True,
        save_period=5,
        plots=True,
        val=True,
        exist_ok=True,
        pretrained=False,
        verbose=True,
        split='val',
        save_json=False,
        save_hybrid=False,
        half=True,                # 启用半精度训练以减少内存使用
        dnn=False,
        source=None,
        vid_stride=1,
        stream_buffer=False,
        visualize=False,
        augment=False,
        agnostic_nms=True,
        classes=None,
        retina_masks=True,
        embed=None,
        show=False,
        save_frames=False,
        save_txt=False,
        save_conf=False,
        save_crop=False,
        show_labels=True,
        show_conf=True,
        show_boxes=True,
        line_width=None,
        format='torchscript',
        keras=False,
        optimize=False,
        int8=False,
        dynamic=False,
        simplify=True,
        opset=None,
        workspace=None,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        nbs=64,
        bgr=0.0,
        copy_paste_mode='flip',
        crop_fraction=1.0,
        cfg=None,
        tracker='botsort.yaml',
    )

if __name__ == '__main__':
    train()