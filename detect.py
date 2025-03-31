from ultralytics import YOLO
import cv2
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import pytest
from PIL import Image, ImageDraw, ImageFont
import torch

def get_device():
    """
    自动检测并返回可用的设备
    """
    if torch.cuda.is_available():
        return '0'  # 如果有GPU，使用第一个GPU
    return 'cpu'    # 如果没有GPU，使用CPU

def plot_results(image, results, save_path=None):
    """
    绘制预测结果
    """
    # 转换为PIL Image
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)
    
    # 加载中文字体
    try:
        font = ImageFont.truetype("/hy-tmp/fonts/simhei.ttf", 20)
    except:
        # 如果找不到中文字体，使用默认字体
        font = ImageFont.load_default()
    
    # 绘制分割结果
    if results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()
        img_np = np.array(img)
        for i, mask in enumerate(masks):
            # 调整掩码尺寸以匹配图像尺寸
            mask = cv2.resize(mask, (img_np.shape[1], img_np.shape[0]))
            
            # 为每个类别使用不同的颜色
            color = tuple(np.random.randint(0, 255, size=(3,)).tolist())
            colored_mask = np.zeros_like(img_np)
            colored_mask[mask > 0.5] = color
            
            # 添加半透明遮罩
            img_np = cv2.addWeighted(img_np, 1, colored_mask.astype(np.uint8), 0.5, 0)
        img = Image.fromarray(img_np)
        draw = ImageDraw.Draw(img)
    
    # 绘制边界框和标签
    for box, cls, conf in zip(results[0].boxes.xyxy.cpu().numpy(), 
                            results[0].boxes.cls.cpu().numpy(),
                            results[0].boxes.conf.cpu().numpy()):
        x1, y1, x2, y2 = map(int, box)
        class_name = results[0].names[int(cls)]
        
        # 绘制边界框
        draw.rectangle([(x1, y1), (x2, y2)], outline=(0, 255, 0), width=2)
        
        # 添加标签和置信度
        label = f'{class_name} {conf:.2f}'
        # 获取文本大小
        bbox = draw.textbbox((x1, y1), label, font=font)
        # 绘制标签背景
        draw.rectangle([bbox[0], bbox[1]-2, bbox[2], bbox[3]+2], fill=(0, 255, 0))
        # 绘制文本
        draw.text((x1, bbox[1]-2), label, fill=(0, 0, 0), font=font)
    
    # 保存或返回结果
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        img.save(save_path)
    
    return np.array(img)

def get_next_result_dir():
    """
    获取下一个可用的结果目录
    返回格式：result1, result2, result3, ...
    """
    base_dir = 'result'
    counter = 1
    while True:
        dir_name = f"{base_dir}{counter}"
        if not os.path.exists(dir_name):
            return dir_name
        counter += 1

def run_model_test(model_path, test_dir, output_dir=None):
    """
    运行模型测试
    :param model_path: 模型路径
    :param test_dir: 测试图片目录
    :param output_dir: 输出目录（可选）
    """
    # 创建输出目录
    if output_dir is None:
        output_dir = get_next_result_dir()
    
    # 创建detect和mask子目录
    detect_dir = os.path.join(output_dir, 'detect')
    mask_dir = os.path.join(output_dir, 'mask')
    os.makedirs(detect_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    
    # 加载模型
    model = YOLO(model_path)
    print(f"Loaded model from {model_path}")
    
    # 获取测试图片列表
    test_images = list(Path(test_dir).glob('*.jpg')) + list(Path(test_dir).glob('*.png'))
    print(f"Found {len(test_images)} test images")
    
    # 获取可用设备
    device = get_device()
    print(f"Using device: {device}")
    
    results = []
    # 处理每张图片
    for img_path in test_images:
        print(f"\nProcessing {img_path.name}")
        
        # 读取图片
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 进行预测
        pred_results = model.predict(
            source=img,
            conf=0.3,         # 置信度阈值
            iou=0.3,          # NMS IOU阈值
            save=False,       # 不保存原始预测结果
            device=device,    # 使用检测到的设备
            agnostic_nms=True # 使用类别无关的NMS
        )
        
        # 绘制结果并保存到detect目录
        detect_path = os.path.join(detect_dir, f"{img_path.stem}_result{img_path.suffix}")
        result_img = plot_results(img, pred_results, save_path=detect_path)
        
        # 收集预测结果
        image_results = {
            'image_name': img_path.name,
            'predictions': []
        }
        
        if pred_results[0].boxes is not None:
            for box, cls, conf in zip(pred_results[0].boxes.xyxy.cpu().numpy(), 
                                    pred_results[0].boxes.cls.cpu().numpy(),
                                    pred_results[0].boxes.conf.cpu().numpy()):
                class_name = pred_results[0].names[int(cls)]
                image_results['predictions'].append({
                    'class': class_name,
                    'confidence': float(conf),
                    'box': box.tolist()
                })
                print(f"- {class_name}: {conf:.2%} confidence")
        
        # 保存分割掩码到mask目录（如果有）
        if pred_results[0].masks is not None:
            image_results['masks'] = []
            for i, mask in enumerate(pred_results[0].masks.data.cpu().numpy()):
                # 调整掩码尺寸以匹配原始图像
                mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
                mask_path = os.path.join(mask_dir, f"{img_path.stem}_mask_{i}{img_path.suffix}")
                cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))
                image_results['masks'].append(mask_path)
        
        results.append(image_results)
    
    return results

def test_model(model_config):
    """
    测试模型预测功能
    """
    # 运行测试
    results = run_model_test(**model_config)
    
    # 验证结果
    assert len(results) > 0, "No results generated"
    
    for result in results:
        # 验证基本结构
        assert 'image_name' in result, "Missing image name in results"
        assert 'predictions' in result, "Missing predictions in results"
        
        # 验证预测结果
        if result['predictions']:
            for pred in result['predictions']:
                assert 'class' in pred, "Missing class in prediction"
                assert 'confidence' in pred, "Missing confidence in prediction"
                assert 'box' in pred, "Missing bounding box in prediction"
                assert len(pred['box']) == 4, "Invalid bounding box format"
                assert pred['confidence'] > 0 and pred['confidence'] <= 1, "Invalid confidence value"

if __name__ == '__main__':
    # 直接运行测试
    config = {
        'model_path': 'runs/exp/weights/best.pt',
        'test_dir': 'te'
    }
    run_model_test(**config)
    print(111)