# 验证模型

from ultralytics import YOLO
import torch
import argparse
import time
import cv2
import numpy as np
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='runs/train/exp_0328_opt/weights/best.pt', help='Model path')
    parser.add_argument('--img-size', type=int, default=640, help='Inference size')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=300, help='Maximum detections per image')
    parser.add_argument('--device', default='0', help='Device to use')
    parser.add_argument('--source', type=str, default=None, help='Source (file/folder/0 for webcam)')
    parser.add_argument('--save-dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--show', action='store_true', help='Show results')
    parser.add_argument('--profile', action='store_true', help='Profile model speed')
    parser.add_argument('--mode', default='balanced', 
                        choices=['high_recall', 'high_precision', 'balanced', 'best_f1', 'best_map', 'best_seg'],
                        help='Inference mode: high_recall, high_precision, balanced, best_f1, best_map, or best_seg')
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # 创建保存目录
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # 加载模型
    model = YOLO(args.model)
    print(f"Loaded model: {args.model}")
    
    # 根据模式设置参数 (根据参数评估结果设置)
    if args.mode == 'high_recall':
        # 高召回率模式 - 参数评估结果
        conf_thres = 0.15           # 低置信度阈值
        iou_thres = 0.3             # 较低的IoU阈值  
        max_det = 500               # 更多检测数量
        print("Running in HIGH RECALL mode")
        print(f"优化设置: 精确率=0.7186, 召回率=0.5418, F1=0.6178, mAP50=0.6820")
    elif args.mode == 'high_precision':
        # 高精确率模式 - 参数评估结果
        conf_thres = 0.4            # 较高的置信度阈值
        iou_thres = 0.3             # IoU阈值
        max_det = 200               # 较少检测数量
        print("Running in HIGH PRECISION mode")
        print(f"优化设置: 精确率=0.8589, 召回率=0.4808, F1=0.6165, mAP50=0.6933")
    elif args.mode == 'best_f1':
        # 最佳F1分数模式 - 参数评估结果
        conf_thres = 0.35           # 最佳F1的置信度阈值
        iou_thres = 0.3             # 最佳F1的IoU阈值
        max_det = 300               # 平衡检测数量
        print("Running in BEST F1 SCORE mode")
        print(f"优化设置: 精确率=0.8528, 召回率=0.4853, F1=0.6186, mAP50=0.6914")
    elif args.mode == 'best_map':
        # 最佳mAP模式 - 参数评估结果
        conf_thres = 0.4            # 最佳mAP的置信度阈值
        iou_thres = 0.3             # 最佳mAP的IoU阈值
        max_det = 300               # 平衡检测数量
        print("Running in BEST mAP mode")
        print(f"优化设置: 精确率=0.8589, 召回率=0.4808, F1=0.6165, mAP50=0.6933")
    elif args.mode == 'best_seg':
        # 最佳分割模式 - 参数评估结果
        conf_thres = 0.4            # 最佳分割的置信度阈值
        iou_thres = 0.4             # 最佳分割的IoU阈值
        max_det = 300               # 平衡检测数量
        print("Running in BEST SEGMENTATION mode")
        print(f"优化设置: seg_mAP50=0.6351, seg_mAP50-95=0.3792")
    else:
        # 平衡模式 - 设为最佳F1参数
        conf_thres = 0.35           # 平衡置信度阈值
        iou_thres = 0.3             # 平衡IoU阈值
        max_det = 300               # 平衡检测数量
        print("Running in BALANCED mode (using Best F1 parameters)")
        print(f"优化设置: 精确率=0.8528, 召回率=0.4853, F1=0.6186, mAP50=0.6914")
    
    # 覆盖命令行参数
    if args.conf_thres != 0.25:
        conf_thres = args.conf_thres
    if args.iou_thres != 0.45:
        iou_thres = args.iou_thres
    if args.max_det != 300:
        max_det = args.max_det
        
    print(f"Inference parameters: conf_thres={conf_thres}, iou_thres={iou_thres}, max_det={max_det}")
    
    # 性能测试模式
    if args.profile:
        print("Profiling model speed...")
        # 计算推理速度
        num_warmup = 5
        num_samples = 20
        
        # 预热
        for _ in range(num_warmup):
            _ = model.predict(
                source="https://ultralytics.com/images/bus.jpg",
                imgsz=args.img_size,
                conf=conf_thres,
                iou=iou_thres,
                max_det=max_det,
                device=args.device,
                verbose=False
            )
        
        # 计时
        total_inference = 0
        total_preprocess = 0
        total_postprocess = 0
        
        for _ in range(num_samples):
            results = model.predict(
                source="https://ultralytics.com/images/bus.jpg",
                imgsz=args.img_size,
                conf=conf_thres,
                iou=iou_thres,
                max_det=max_det,
                device=args.device,
                verbose=False
            )
            
            total_inference += results[0].speed['inference']
            total_preprocess += results[0].speed['preprocess']
            total_postprocess += results[0].speed['postprocess']
        
        # 计算平均
        avg_inference = total_inference / num_samples
        avg_preprocess = total_preprocess / num_samples
        avg_postprocess = total_postprocess / num_samples
        total_time = avg_preprocess + avg_inference + avg_postprocess
        
        print(f"Average times: {avg_preprocess:.1f}ms preprocess, {avg_inference:.1f}ms inference, "
              f"{avg_postprocess:.1f}ms postprocess, {total_time:.1f}ms total per image")
        print(f"FPS: {1000 / total_time:.1f}")
        return
    
    # 检查源是否有效
    if args.source is None:
        print("No source specified. Using sample image for demonstration.")
        source = "https://ultralytics.com/images/bus.jpg"
    else:
        source = args.source
    
    # 运行预测
    results = model.predict(
        source=source,
        imgsz=args.img_size,
        conf=conf_thres,
        iou=iou_thres,
        max_det=max_det,
        device=args.device,
        save=True,
        save_txt=True,
        save_conf=True,
        save_crop=False,
        project=args.save_dir,
        name=f"{args.mode}_conf{conf_thres}_iou{iou_thres}",
        show=args.show,
        stream=False,
        verbose=True
    )
    
    print(f"Results saved to {args.save_dir}")

if __name__ == "__main__":
    main() 