# 算出最佳测试参数

from ultralytics import YOLO
import torch
import numpy as np
import pandas as pd
import itertools
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # 加载模型 - 使用最优模型
    model_path = 'runs/train/exp_0328_opt/weights/best.pt'
    model = YOLO(model_path)
    print(f"Loaded model: {model_path}")
    
    # 参数网格
    conf_thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    iou_thresholds = [0.3, 0.4, 0.45, 0.5, 0.55, 0.6]
    
    # 记录结果的数据结构
    results = []
    
    # 结果目录
    os.makedirs('param_eval', exist_ok=True)
    
    # 测试参数组合
    for conf, iou in tqdm(list(itertools.product(conf_thresholds, iou_thresholds)), 
                          desc="Evaluating parameter combinations"):
        # 评估指定参数
        metrics = model.val(
            data='dataset/data.yaml',
            imgsz=640,
            conf=conf,
            iou=iou,
            max_det=300,
            device='0',
            verbose=False
        )
        
        # 提取关键指标
        precision = metrics.box.mp  # 精确率 - 修正：使用 mp 属性获取平均精确率
        recall = metrics.box.mr  # 召回率
        mAP50 = metrics.box.map50  # mAP@0.5
        mAP50_95 = metrics.box.map  # mAP@0.5:0.95
        
        seg_mAP50 = metrics.seg.map50  # 分割mAP@0.5
        seg_mAP50_95 = metrics.seg.map  # 分割mAP@0.5:0.95
        
        # 计算F1得分
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        # 保存结果
        results.append({
            'conf_threshold': conf,
            'iou_threshold': iou,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'mAP50': mAP50,
            'mAP50_95': mAP50_95,
            'seg_mAP50': seg_mAP50,
            'seg_mAP50_95': seg_mAP50_95
        })
    
    # 转换为pandas DataFrame
    results_df = pd.DataFrame(results)
    
    # 保存结果表格
    results_df.to_csv('param_eval/results.csv', index=False)
    print(f"Saved results to param_eval/results.csv")
    
    # 找出最佳组合
    best_f1 = results_df.loc[results_df['f1_score'].idxmax()]
    best_map = results_df.loc[results_df['mAP50'].idxmax()]
    best_prec = results_df.loc[results_df['precision'].idxmax()]
    best_recall = results_df.loc[results_df['recall'].idxmax()]
    best_seg = results_df.loc[results_df['seg_mAP50'].idxmax()]
    
    print("\n=== BEST PARAMETER COMBINATIONS ===")
    print(f"Best F1 Score: conf={best_f1['conf_threshold']}, iou={best_f1['iou_threshold']}, F1={best_f1['f1_score']:.4f}")
    print(f"Best mAP50: conf={best_map['conf_threshold']}, iou={best_map['iou_threshold']}, mAP50={best_map['mAP50']:.4f}")
    print(f"Best Precision: conf={best_prec['conf_threshold']}, iou={best_prec['iou_threshold']}, Precision={best_prec['precision']:.4f}")
    print(f"Best Recall: conf={best_recall['conf_threshold']}, iou={best_recall['iou_threshold']}, Recall={best_recall['recall']:.4f}")
    print(f"Best Segmentation: conf={best_seg['conf_threshold']}, iou={best_seg['iou_threshold']}, seg_mAP50={best_seg['seg_mAP50']:.4f}")
    
    # 可视化结果
    plt.figure(figsize=(15, 12))
    
    # F1得分热图
    plt.subplot(2, 2, 1)
    pivot_f1 = results_df.pivot_table(index='conf_threshold', columns='iou_threshold', values='f1_score')
    sns.heatmap(pivot_f1, annot=True, fmt='.3f', cmap='viridis')
    plt.title('F1 Score')
    
    # Precision热图
    plt.subplot(2, 2, 2)
    pivot_prec = results_df.pivot_table(index='conf_threshold', columns='iou_threshold', values='precision')
    sns.heatmap(pivot_prec, annot=True, fmt='.3f', cmap='viridis')
    plt.title('Precision')
    
    # Recall热图
    plt.subplot(2, 2, 3)
    pivot_recall = results_df.pivot_table(index='conf_threshold', columns='iou_threshold', values='recall')
    sns.heatmap(pivot_recall, annot=True, fmt='.3f', cmap='viridis')
    plt.title('Recall')
    
    # mAP50热图
    plt.subplot(2, 2, 4)
    pivot_map = results_df.pivot_table(index='conf_threshold', columns='iou_threshold', values='mAP50')
    sns.heatmap(pivot_map, annot=True, fmt='.3f', cmap='viridis')
    plt.title('mAP50')
    
    plt.tight_layout()
    plt.savefig('param_eval/heatmaps.png')
    print("Saved heatmap visualization to param_eval/heatmaps.png")
    
    # PR曲线
    plt.figure(figsize=(10, 6))
    plt.scatter(results_df['recall'], results_df['precision'], c=results_df['conf_threshold'], cmap='viridis', s=50)
    plt.colorbar(label='Confidence Threshold')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Trade-off')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 高亮最佳F1点
    plt.scatter([best_f1['recall']], [best_f1['precision']], c='red', s=100, marker='*', 
                label=f'Best F1: {best_f1["f1_score"]:.3f} (conf={best_f1["conf_threshold"]}, iou={best_f1["iou_threshold"]})')
    plt.legend()
    plt.savefig('param_eval/pr_curve.png')
    print("Saved PR curve to param_eval/pr_curve.png")
    
    # 创建推荐配置文件
    with open('param_eval/recommended_settings.txt', 'w') as f:
        f.write("# Recommended Parameter Settings\n\n")
        f.write("## Balanced (Best F1)\n")
        f.write(f"conf_threshold: {best_f1['conf_threshold']}\n")
        f.write(f"iou_threshold: {best_f1['iou_threshold']}\n")
        f.write(f"Metrics: Precision={best_f1['precision']:.4f}, Recall={best_f1['recall']:.4f}, F1={best_f1['f1_score']:.4f}, mAP50={best_f1['mAP50']:.4f}\n\n")
        
        f.write("## High Precision\n")
        f.write(f"conf_threshold: {best_prec['conf_threshold']}\n")
        f.write(f"iou_threshold: {best_prec['iou_threshold']}\n")
        f.write(f"Metrics: Precision={best_prec['precision']:.4f}, Recall={best_prec['recall']:.4f}, F1={best_prec['f1_score']:.4f}, mAP50={best_prec['mAP50']:.4f}\n\n")
        
        f.write("## High Recall\n")
        f.write(f"conf_threshold: {best_recall['conf_threshold']}\n")
        f.write(f"iou_threshold: {best_recall['iou_threshold']}\n")
        f.write(f"Metrics: Precision={best_recall['precision']:.4f}, Recall={best_recall['recall']:.4f}, F1={best_recall['f1_score']:.4f}, mAP50={best_recall['mAP50']:.4f}\n\n")
        
        f.write("## Best Overall Performance (mAP50)\n")
        f.write(f"conf_threshold: {best_map['conf_threshold']}\n")
        f.write(f"iou_threshold: {best_map['iou_threshold']}\n")
        f.write(f"Metrics: Precision={best_map['precision']:.4f}, Recall={best_map['recall']:.4f}, F1={best_map['f1_score']:.4f}, mAP50={best_map['mAP50']:.4f}\n\n")
        
        f.write("## Best Segmentation\n")
        f.write(f"conf_threshold: {best_seg['conf_threshold']}\n")
        f.write(f"iou_threshold: {best_seg['iou_threshold']}\n")
        f.write(f"Metrics: seg_mAP50={best_seg['seg_mAP50']:.4f}, seg_mAP50-95={best_seg['seg_mAP50_95']:.4f}\n")
    
    print("Saved recommended settings to param_eval/recommended_settings.txt")
    
    print("\nParameter evaluation complete!")

if __name__ == "__main__":
    main() 