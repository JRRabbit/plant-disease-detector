"""
农业病虫害监测 YOLO 模型训练
优化增强版 - 提升准确率 + 完整保存功能
"""

import os
import shutil
import random
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from collections import Counter
from tqdm import tqdm
import cv2

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def prepare_classification_dataset(source_dir, output_dir, train_ratio=0.8):
    """准备分类数据集"""
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    
    if output_dir.exists():
        shutil.rmtree(output_dir)
    
    img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    
    class_names = sorted([
        d.name for d in source_dir.iterdir() 
        if d.is_dir() and not d.name.startswith('.')
    ])
    
    print(f"发现 {len(class_names)} 个类别: {class_names}")
    
    stats = {}
    
    for class_name in class_names:
        class_dir = source_dir / class_name
        images = [f for f in class_dir.iterdir() if f.suffix.lower() in img_extensions]
        stats[class_name] = len(images)
        
        print(f"  {class_name}: {len(images)} 张图像")
        
        if len(images) < 2:
            train_imgs, val_imgs = images, []
        else:
            train_imgs, val_imgs = train_test_split(
                images, test_size=1-train_ratio, random_state=42
            )
        
        train_dir = output_dir / 'train' / class_name
        val_dir = output_dir / 'val' / class_name
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)
        
        for img in train_imgs:
            shutil.copy2(img, train_dir / img.name)
        for img in val_imgs:
            shutil.copy2(img, val_dir / img.name)
    
    print(f"\n数据集准备完成!")
    return class_names, stats


def visualize_dataset(source_dir, stats, save_path='dataset_distribution.png'):
    """可视化数据集分布"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    classes = list(stats.keys())
    counts = list(stats.values())
    colors = plt.cm.viridis(np.linspace(0, 1, len(classes)))
    
    ax1 = axes[0]
    bars = ax1.barh(classes, counts, color=colors)
    ax1.set_xlabel('图像数量')
    ax1.set_title('各类别图像数量')
    ax1.bar_label(bars, padding=3)
    
    ax2 = axes[1]
    ax2.pie(counts, labels=classes, autopct='%1.1f%%', colors=colors)
    ax2.set_title('各类别占比')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"已保存: {save_path}")
    plt.show()


def show_sample_images(source_dir, num_per_class=3, save_path='sample_images.png'):
    """展示样本图像"""
    source_dir = Path(source_dir)
    img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    
    class_dirs = sorted([d for d in source_dir.iterdir() 
                         if d.is_dir() and not d.name.startswith('.')])
    num_classes = len(class_dirs)
    
    fig, axes = plt.subplots(num_classes, num_per_class, 
                             figsize=(num_per_class * 3, num_classes * 2.5))
    
    if num_classes == 1:
        axes = axes.reshape(1, -1)
    
    for i, class_dir in enumerate(class_dirs):
        images = [f for f in class_dir.iterdir() if f.suffix.lower() in img_extensions]
        samples = random.sample(images, min(num_per_class, len(images)))
        
        for j in range(num_per_class):
            ax = axes[i, j] if num_classes > 1 else axes[j]
            if j < len(samples):
                img = plt.imread(str(samples[j]))
                ax.imshow(img)
                if j == 0:
                    ax.set_ylabel(class_dir.name, fontsize=10)
            ax.axis('off')
    
    plt.suptitle('各类别样本图像', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"已保存: {save_path}")
    plt.show()


def train_model_optimized(data_dir, epochs=100, imgsz=320, batch=32, model_size='s'):
    """
    优化版训练函数 - 提升准确率
    """
    model_name = f'yolov8{model_size}-cls.pt'
    print(f"\n加载模型: {model_name}")
    print("=" * 50)
    print("优化参数配置:")
    print(f"  - 训练轮数: {epochs}")
    print(f"  - 图像尺寸: {imgsz}")
    print(f"  - 批次大小: {batch}")
    print(f"  - 模型大小: {model_size}")
    print(f"  - 数据增强: 已启用")
    print(f"  - 标签平滑: 0.1")
    print(f"  - 余弦学习率: 已启用")
    print("=" * 50)
    
    model = YOLO(model_name)
    
    results = model.train(
        data=str(data_dir),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project='runs/classify',
        name='pest_disease_optimized',
        
        # 学习率设置
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=5,
        warmup_momentum=0.8,
        
        # 优化器
        optimizer='AdamW',
        weight_decay=0.0005,
        momentum=0.937,
        
        # 数据增强
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=15.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0001,
        flipud=0.1,
        fliplr=0.5,
        mosaic=0.5,
        mixup=0.1,
        erasing=0.2,
        
        # 训练策略
        patience=30,
        cos_lr=True,
        label_smoothing=0.1,
        
        # 其他
        save=True,
        plots=True,
        verbose=True,
        device='mps',
        workers=4,
        seed=42,
    )
    
    return model, results


def plot_training_curves(results_dir, save_path='training_curves.png'):
    """绘制训练曲线"""
    results_dir = Path(results_dir)
    csv_path = results_dir / 'results.csv'
    
    if not csv_path.exists():
        print(f"未找到: {csv_path}")
        return None
    
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 损失曲线
    ax1 = axes[0, 0]
    if 'train/loss' in df.columns:
        ax1.plot(df['epoch'], df['train/loss'], 'b-', label='训练损失', linewidth=2)
    if 'val/loss' in df.columns:
        ax1.plot(df['epoch'], df['val/loss'], 'r-', label='验证损失', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('损失曲线')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 准确率曲线
    ax2 = axes[0, 1]
    if 'metrics/accuracy_top1' in df.columns:
        ax2.plot(df['epoch'], df['metrics/accuracy_top1'], 'g-', 
                label='Top-1 准确率', linewidth=2)
    if 'metrics/accuracy_top5' in df.columns:
        ax2.plot(df['epoch'], df['metrics/accuracy_top5'], 'm-', 
                label='Top-5 准确率', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('准确率曲线')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    # 学习率曲线
    ax3 = axes[1, 0]
    lr_cols = [col for col in df.columns if 'lr' in col.lower()]
    for col in lr_cols:
        ax3.plot(df['epoch'], df[col], label=col, linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_title('学习率曲线')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 统计信息
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    stats_text = "训练统计\n" + "=" * 35 + "\n\n"
    stats_text += f"总训练轮数: {len(df)}\n\n"
    
    if 'metrics/accuracy_top1' in df.columns:
        best_acc = df['metrics/accuracy_top1'].max()
        best_epoch = df.loc[df['metrics/accuracy_top1'].idxmax(), 'epoch']
        final_acc = df['metrics/accuracy_top1'].iloc[-1]
        stats_text += f"最佳 Top-1: {best_acc:.2%} (Epoch {int(best_epoch)})\n"
        stats_text += f"最终 Top-1: {final_acc:.2%}\n\n"
    
    if 'metrics/accuracy_top5' in df.columns:
        stats_text += f"最佳 Top-5: {df['metrics/accuracy_top5'].max():.2%}\n"
        stats_text += f"最终 Top-5: {df['metrics/accuracy_top5'].iloc[-1]:.2%}\n\n"
    
    if 'train/loss' in df.columns:
        stats_text += f"最小训练损失: {df['train/loss'].min():.4f}\n"
    if 'val/loss' in df.columns:
        stats_text += f"最小验证损失: {df['val/loss'].min():.4f}\n"
    
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes,
            fontsize=12, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"已保存: {save_path}")
    plt.show()
    
    return df


def plot_confusion_matrix(model, data_dir, class_names, save_path='confusion_matrix.png'):
    """绘制混淆矩阵"""
    print("\n正在生成混淆矩阵...")
    
    data_dir = Path(data_dir)
    val_dir = data_dir / 'val'
    img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    
    y_true = []
    y_pred = []
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = val_dir / class_name
        if not class_dir.exists():
            continue
        
        images = [f for f in class_dir.iterdir() 
                  if f.is_file() and f.suffix.lower() in img_extensions]
        
        for img_path in tqdm(images, desc=f"预测 {class_name}"):
            results = model.predict(source=str(img_path), verbose=False)
            pred_class = results[0].probs.top1
            y_true.append(class_idx)
            y_pred.append(pred_class)
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # 数量混淆矩阵
    ax1 = axes[0]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax1)
    ax1.set_xlabel('预测类别')
    ax1.set_ylabel('真实类别')
    ax1.set_title('混淆矩阵 (数量)')
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 归一化混淆矩阵
    ax2 = axes[1]
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)
    sns.heatmap(cm_norm, annot=True, fmt='.1%', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax2)
    ax2.set_xlabel('预测类别')
    ax2.set_ylabel('真实类别')
    ax2.set_title('混淆矩阵 (归一化)')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"已保存: {save_path}")
    plt.show()
    
    # 保存分类报告
    report = classification_report(y_true, y_pred, target_names=class_names)
    print("\n分类报告:")
    print("=" * 60)
    print(report)
    
    report_path = save_path.replace('.png', '_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("分类报告\n" + "=" * 60 + "\n" + report)
    print(f"已保存: {report_path}")
    
    return cm, y_true, y_pred


def plot_per_class_accuracy(y_true, y_pred, class_names, save_path='per_class_accuracy.png'):
    """绘制每类准确率"""
    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    per_class_acc = np.nan_to_num(per_class_acc)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = plt.cm.RdYlGn(per_class_acc)
    bars = ax.barh(class_names, per_class_acc, color=colors)
    
    for bar, acc in zip(bars, per_class_acc):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{acc:.1%}', va='center', fontsize=10)
    
    ax.set_xlabel('准确率')
    ax.set_title('各类别准确率')
    ax.set_xlim([0, 1.15])
    ax.axvline(x=np.mean(per_class_acc), color='red', linestyle='--', 
               label=f'平均: {np.mean(per_class_acc):.1%}')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"已保存: {save_path}")
    plt.show()


def visualize_predictions(model, data_dir, class_names, num_samples=16, 
                          save_path='prediction_samples.png'):
    """可视化预测样本"""
    print("\n正在生成预测可视化...")
    
    data_dir = Path(data_dir)
    val_dir = data_dir / 'val'
    img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    
    all_images = []
    for class_name in class_names:
        class_dir = val_dir / class_name
        if class_dir.exists():
            images = [f for f in class_dir.iterdir() 
                      if f.is_file() and f.suffix.lower() in img_extensions]
            for img in images:
                all_images.append((img, class_name))
    
    samples = random.sample(all_images, min(num_samples, len(all_images)))
    
    cols = 4
    rows = (len(samples) + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axes = axes.flatten()
    
    for idx, (img_path, true_label) in enumerate(samples):
        ax = axes[idx]
        
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        results = model.predict(source=str(img_path), verbose=False)
        probs = results[0].probs
        pred_idx = probs.top1
        pred_conf = probs.top1conf.item()
        pred_label = class_names[pred_idx]
        
        ax.imshow(img)
        
        is_correct = (pred_label == true_label)
        title_color = 'green' if is_correct else 'red'
        title = f"真实: {true_label}\n预测: {pred_label} ({pred_conf:.1%})"
        ax.set_title(title, color=title_color, fontsize=10)
        ax.axis('off')
    
    for idx in range(len(samples), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('预测结果 (绿色=正确, 红色=错误)', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"已保存: {save_path}")
    plt.show()


def visualize_wrong_predictions(model, data_dir, class_names, num_samples=12,
                                 save_path='wrong_predictions.png'):
    """可视化错误预测"""
    print("\n正在查找错误预测...")
    
    data_dir = Path(data_dir)
    val_dir = data_dir / 'val'
    img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    
    wrong_predictions = []
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = val_dir / class_name
        if not class_dir.exists():
            continue
        
        images = [f for f in class_dir.iterdir() 
                  if f.is_file() and f.suffix.lower() in img_extensions]
        
        for img_path in images:
            results = model.predict(source=str(img_path), verbose=False)
            probs = results[0].probs
            pred_idx = probs.top1
            pred_conf = probs.top1conf.item()
            
            if pred_idx != class_idx:
                wrong_predictions.append({
                    'path': img_path,
                    'true_label': class_name,
                    'pred_label': class_names[pred_idx],
                    'confidence': pred_conf
                })
    
    if len(wrong_predictions) == 0:
        print("没有错误预测!")
        return
    
    print(f"找到 {len(wrong_predictions)} 个错误预测")
    
    samples = random.sample(wrong_predictions, min(num_samples, len(wrong_predictions)))
    
    cols = 4
    rows = (len(samples) + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axes = np.array(axes).flatten()
    
    for idx, sample in enumerate(samples):
        ax = axes[idx]
        
        img = cv2.imread(str(sample['path']))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        ax.imshow(img)
        title = f"真实: {sample['true_label']}\n预测: {sample['pred_label']} ({sample['confidence']:.1%})"
        ax.set_title(title, color='red', fontsize=10)
        ax.axis('off')
    
    for idx in range(len(samples), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('错误预测样本分析', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"已保存: {save_path}")
    plt.show()


def plot_top5_predictions(model, data_dir, class_names, num_samples=6,
                          save_path='top5_predictions.png'):
    """可视化 Top-5 预测"""
    print("\n正在生成 Top-5 预测可视化...")
    
    data_dir = Path(data_dir)
    val_dir = data_dir / 'val'
    img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    
    all_images = []
    for class_name in class_names:
        class_dir = val_dir / class_name
        if class_dir.exists():
            images = [f for f in class_dir.iterdir() 
                      if f.is_file() and f.suffix.lower() in img_extensions]
            for img in images:
                all_images.append((img, class_name))
    
    samples = random.sample(all_images, min(num_samples, len(all_images)))
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(14, 3 * num_samples))
    
    for idx, (img_path, true_label) in enumerate(samples):
        ax_img = axes[idx, 0]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax_img.imshow(img)
        ax_img.set_title(f'真实: {true_label}', fontsize=12)
        ax_img.axis('off')
        
        ax_bar = axes[idx, 1]
        results = model.predict(source=str(img_path), verbose=False)
        probs = results[0].probs
        top5_idx = probs.top5
        top5_conf = probs.top5conf.tolist()
        
        top5_names = [class_names[i] for i in top5_idx]
        colors = ['green' if name == true_label else 'steelblue' for name in top5_names]
        
        bars = ax_bar.barh(range(5), top5_conf, color=colors)
        ax_bar.set_yticks(range(5))
        ax_bar.set_yticklabels(top5_names)
        ax_bar.set_xlabel('置信度')
        ax_bar.set_title('Top-5 预测')
        ax_bar.set_xlim([0, 1])
        ax_bar.invert_yaxis()
        
        for bar, conf in zip(bars, top5_conf):
            ax_bar.text(conf + 0.01, bar.get_y() + bar.get_height()/2,
                       f'{conf:.1%}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"已保存: {save_path}")
    plt.show()


def generate_summary_report(results_dir, class_names, stats, y_true=None, y_pred=None,
                            save_path='training_summary.png'):
    """生成训练总结报告"""
    print("\n正在生成总结报告...")
    
    results_dir = Path(results_dir)
    csv_path = results_dir / 'results.csv'
    
    fig = plt.figure(figsize=(18, 14))
    
    # 1. 数据集分布
    ax1 = fig.add_subplot(2, 3, 1)
    classes = list(stats.keys())
    counts = list(stats.values())
    colors = plt.cm.viridis(np.linspace(0, 1, len(classes)))
    bars = ax1.barh(classes, counts, color=colors)
    ax1.set_xlabel('图像数量')
    ax1.set_title('数据集分布')
    ax1.bar_label(bars, padding=3)
    
    # 2. 准确率曲线
    ax2 = fig.add_subplot(2, 3, 2)
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        if 'metrics/accuracy_top1' in df.columns:
            ax2.plot(df['epoch'], df['metrics/accuracy_top1'], 'g-', 
                    label='Top-1', linewidth=2)
        if 'metrics/accuracy_top5' in df.columns:
            ax2.plot(df['epoch'], df['metrics/accuracy_top5'], 'b-', 
                    label='Top-5', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('准确率曲线')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1])
    
    # 3. 损失曲线
    ax3 = fig.add_subplot(2, 3, 3)
    if csv_path.exists():
        if 'train/loss' in df.columns:
            ax3.plot(df['epoch'], df['train/loss'], 'b-', label='训练', linewidth=2)
        if 'val/loss' in df.columns:
            ax3.plot(df['epoch'], df['val/loss'], 'r-', label='验证', linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss')
        ax3.set_title('损失曲线')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. 每类准确率
    ax4 = fig.add_subplot(2, 3, 4)
    if y_true is not None and y_pred is not None:
        cm = confusion_matrix(y_true, y_pred)
        per_class_acc = cm.diagonal() / cm.sum(axis=1)
        per_class_acc = np.nan_to_num(per_class_acc)
        colors_acc = plt.cm.RdYlGn(per_class_acc)
        bars = ax4.barh(class_names, per_class_acc, color=colors_acc)
        ax4.set_xlabel('准确率')
        ax4.set_title('各类别准确率')
        ax4.set_xlim([0, 1.1])
        for bar, acc in zip(bars, per_class_acc):
            ax4.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{acc:.0%}', va='center', fontsize=9)
    
    # 5. 混淆矩阵热力图（简化版）
    ax5 = fig.add_subplot(2, 3, 5)
    if y_true is not None and y_pred is not None:
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)
        sns.heatmap(cm_norm, annot=True, fmt='.0%', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, ax=ax5,
                    annot_kws={'size': 8})
        ax5.set_title('混淆矩阵')
        plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
        plt.setp(ax5.yaxis.get_majorticklabels(), rotation=0, fontsize=8)
    
    # 6. 训练总结文本
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    summary_text = "=" * 40 + "\n"
    summary_text += "        训练总结报告\n"
    summary_text += "=" * 40 + "\n\n"
    
    summary_text += f"【数据集】\n"
    summary_text += f"  类别数: {len(class_names)}\n"
    summary_text += f"  总图像: {sum(counts)}\n"
    summary_text += f"  最大类: {max(stats, key=stats.get)} ({max(counts)})\n"
    summary_text += f"  最小类: {min(stats, key=stats.get)} ({min(counts)})\n\n"
    
    if csv_path.exists():
        summary_text += f"【训练配置】\n"
        summary_text += f"  总轮数: {len(df)}\n\n"
        
        summary_text += f"【最终性能】\n"
        if 'metrics/accuracy_top1' in df.columns:
            best_acc = df['metrics/accuracy_top1'].max()
            final_acc = df['metrics/accuracy_top1'].iloc[-1]
            best_epoch = df.loc[df['metrics/accuracy_top1'].idxmax(), 'epoch']
            summary_text += f"  最佳 Top-1: {best_acc:.2%} (E{int(best_epoch)})\n"
            summary_text += f"  最终 Top-1: {final_acc:.2%}\n"
        
        if 'metrics/accuracy_top5' in df.columns:
            summary_text += f"  最佳 Top-5: {df['metrics/accuracy_top5'].max():.2%}\n"
            summary_text += f"  最终 Top-5: {df['metrics/accuracy_top5'].iloc[-1]:.2%}\n"
    
    if y_true is not None and y_pred is not None:
        overall_acc = np.mean(np.array(y_true) == np.array(y_pred))
        summary_text += f"\n【验证集性能】\n"
        summary_text += f"  整体准确率: {overall_acc:.2%}\n"
        summary_text += f"  平均类准确率: {np.mean(per_class_acc):.2%}\n"
    
    weights_dir = results_dir / 'weights'
    if weights_dir.exists():
        summary_text += f"\n【模型文件】\n"
        for f in weights_dir.glob('*.pt'):
            size_mb = f.stat().st_size / (1024 * 1024)
            summary_text += f"  {f.name}: {size_mb:.1f}MB\n"
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.suptitle('农业病虫害监测模型 - 训练总结', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"已保存: {save_path}")
    plt.show()


def evaluate_and_visualize_all(model, results_dir, data_dir, class_names, stats):
    """综合评估和可视化"""
    results_dir = Path(results_dir)
    
    print("\n" + "=" * 60)
    print("开始综合评估和可视化")
    print("=" * 60)
    
    vis_dir = results_dir / 'visualizations'
    vis_dir.mkdir(exist_ok=True)
    
    # 1. 训练曲线
    print("\n[1/7] 绘制训练曲线...")
    plot_training_curves(results_dir, save_path=str(vis_dir / 'training_curves.png'))
    
    # 2. 混淆矩阵
    print("\n[2/7] 绘制混淆矩阵...")
    cm, y_true, y_pred = plot_confusion_matrix(
        model, data_dir, class_names, 
        save_path=str(vis_dir / 'confusion_matrix.png')
    )
    
    # 3. 每类准确率
    print("\n[3/7] 绘制每类准确率...")
    plot_per_class_accuracy(
        y_true, y_pred, class_names,
        save_path=str(vis_dir / 'per_class_accuracy.png')
    )
    
    # 4. 预测样本
    print("\n[4/7] 可视化预测样本...")
    visualize_predictions(
        model, data_dir, class_names, num_samples=16,
        save_path=str(vis_dir / 'prediction_samples.png')
    )
    
    # 5. 错误预测
    print("\n[5/7] 可视化错误预测...")
    visualize_wrong_predictions(
        model, data_dir, class_names, num_samples=12,
        save_path=str(vis_dir / 'wrong_predictions.png')
    )
    
    # 6. Top-5 预测
    print("\n[6/7] Top-5 预测可视化...")
    plot_top5_predictions(
        model, data_dir, class_names, num_samples=6,
        save_path=str(vis_dir / 'top5_predictions.png')
    )
    
    # 7. 总结报告
    print("\n[7/7] 生成总结报告...")
    generate_summary_report(
        results_dir, class_names, stats, y_true, y_pred,
        save_path=str(vis_dir / 'training_summary.png')
    )
    
    print("\n" + "=" * 60)
    print(f"所有可视化结果已保存到: {vis_dir}")
    print("=" * 60)
    
    print("\n生成的文件:")
    for f in sorted(vis_dir.glob('*')):
        size_kb = f.stat().st_size / 1024
        print(f"  ✓ {f.name} ({size_kb:.1f} KB)")
    
    return y_true, y_pred


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    
    print("\n" + "=" * 60)
    print("   农业病虫害监测 YOLO 模型训练 (优化版)")
    print("=" * 60)
    
    # ==================
    # 配置参数
    # ==================
    SOURCE_DIR = "augmented"
    OUTPUT_DIR = "dataset"
    
    # 优化后的训练参数
    EPOCHS =50          # 增加轮数
    IMAGE_SIZE = 320      # 增加图像尺寸
    BATCH_SIZE = 32       # 批次大小
    MODEL_SIZE = 's'      # 使用更大模型: n < s < m < l < x
    
    # ==================
    # 步骤 1: 准备数据集
    # ==================
    print("\n" + "=" * 60)
    print("步骤 1: 准备数据集")
    print("=" * 60)
    
    class_names, stats = prepare_classification_dataset(
        source_dir=SOURCE_DIR,
        output_dir=OUTPUT_DIR,
        train_ratio=0.8
    )
    
    # ==================
    # 步骤 2: 可视化数据集
    # ==================
    print("\n" + "=" * 60)
    print("步骤 2: 可视化数据集")
    print("=" * 60)
    
    # 创建临时目录保存数据集可视化
    temp_vis_dir = Path('temp_visualizations')
    temp_vis_dir.mkdir(exist_ok=True)
    
    visualize_dataset(SOURCE_DIR, stats, 
                      save_path=str(temp_vis_dir / 'dataset_distribution.png'))
    show_sample_images(SOURCE_DIR, num_per_class=3, 
                       save_path=str(temp_vis_dir / 'sample_images.png'))
    
    # ==================
    # 步骤 3: 训练模型（使用优化参数）
    # ==================
    print("\n" + "=" * 60)
    print("步骤 3: 训练模型 (优化版)")
    print("=" * 60)
    
    model, results = train_model_optimized(
        data_dir=OUTPUT_DIR,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        model_size=MODEL_SIZE
    )
    
    # ==================
    # 步骤 4: 获取结果目录
    # ==================
    results_dir = Path('runs/classify/pest_disease_optimized')
    if not results_dir.exists():
        classify_dir = Path('runs/classify')
        results_dirs = sorted(classify_dir.glob('pest_disease*'))
        if results_dirs:
            results_dir = results_dirs[-1]
    
    print(f"\n结果目录: {results_dir}")
    
    # ==================
    # 步骤 5: 综合评估和可视化
    # ==================
    print("\n" + "=" * 60)
    print("步骤 4: 综合评估和可视化")
    print("=" * 60)
    
    y_true, y_pred = evaluate_and_visualize_all(
        model=model,
        results_dir=results_dir,
        data_dir=OUTPUT_DIR,
        class_names=class_names,
        stats=stats
    )
    
    # ==================
    # 步骤 6: 复制数据集可视化到结果目录
    # ==================
    vis_dir = results_dir / 'visualizations'
    for f in temp_vis_dir.glob('*.png'):
        shutil.copy2(f, vis_dir / f.name)
        print(f"  已复制: {f.name}")
    
    # 清理临时目录
    shutil.rmtree(temp_vis_dir)
    
    # ==================
    # 步骤 7: 打印最终信息
    # ==================
    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)
    
    print(f"\n📁 结果保存位置: {results_dir}")
    print(f"📊 可视化图表: {vis_dir}")
    
    weights_dir = results_dir / 'weights'
    print(f"\n🔧 模型文件:")
    print(f"   - {weights_dir / 'best.pt'} (最佳模型)")
    print(f"   - {weights_dir / 'last.pt'} (最后模型)")
    
    print("\n📈 所有生成的可视化:")
    for f in sorted(vis_dir.glob('*')):
        print(f"   ✓ {f.name}")
    
    print("\n" + "=" * 60)

