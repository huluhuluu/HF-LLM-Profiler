#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate charts from measured JSON data
"""

import json
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
from matplotlib.lines import Line2D

# Font and style settings
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#fafafa'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.2
plt.rcParams['grid.linestyle'] = '-'
plt.rcParams['grid.color'] = '#aaaaaa'


def load_data(json_file="bench/measured_data.json"):
    """Load test data"""
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"Data file not found: {json_file}\nPlease run benchmark.py first")
    with open(json_file, "r") as f:
        return json.load(f)


def extract_model_data(data):
    """Extract model data from JSON"""
    models = []
    for mid, mdata in data.get("models", {}).items():
        if mdata.get("status") != "success":
            continue
        
        active_params = mdata.get("params_b", 0)
        total_params = mdata.get("total_params_b", active_params)
        
        models.append({
            "id": mid,
            "name": mdata.get("name", mid),
            "active_params_b": active_params,
            "total_params_b": total_params,
            "is_moe": mdata.get("is_moe", False),
            "layers": mdata.get("layers", 0),
            "hidden_size": mdata.get("hidden_size", 0),
            "dtype": mdata.get("dtype", "").replace("torch.", ""),
            # FLOPs
            "fwd_flops_gflops": mdata.get("fwd_flops_gflops", 0),
            "bwd_flops_gflops": mdata.get("bwd_flops_gflops", 0),
            # Time
            "estimated_total_time_fwd_s": mdata.get("estimated_total_time_fwd_s", 0),
            "estimated_total_time_fwd_bwd_s": mdata.get("estimated_total_time_fwd_bwd_s", 0),
            # Memory
            "estimated_total_mem_fwd_gb": mdata.get("estimated_total_mem_fwd_gb", 0),
            "estimated_total_mem_fwd_bwd_gb": mdata.get("estimated_total_mem_fwd_bwd_gb", 0),
        })
    
    # Sort by active params
    models.sort(key=lambda x: x["active_params_b"])
    return models


def plot_training_flops_vs_memory(models, output_dir="bench", config=None):
    """
    绘制训练 FLOPs vs Memory 二维散点图（对数轴）
    Y轴: Forward+Backward FLOPs (TFLOPs)
    X轴: Forward+Backward Memory (GB)
    """
    if not models:
        print("No valid model data for FLOPs vs Memory chart")
        return
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # 收集数据
    points = []
    for m in models:
        fwd_flops = m.get("fwd_flops_gflops", 0) / 1000
        bwd_flops = m.get("bwd_flops_gflops", 0) / 1000
        total_flops = fwd_flops + bwd_flops
        memory = m.get("estimated_total_mem_fwd_bwd_gb", 0)
        
        # 过滤无效数据
        if memory <= 0 or total_flops <= 0:
            continue
        
        points.append({
            "x": memory,
            "y": total_flops,
            "name": m["name"] + ("*" if m["is_moe"] else ""),
            "params": m["active_params_b"],
            "is_moe": m["is_moe"]
        })
    
    if not points:
        print("No valid data points for chart")
        return
    
    # 按参数量排序
    points.sort(key=lambda p: p["params"])
    
    # 颜色渐变
    cmap = plt.cm.viridis
    max_params = max(p["params"] for p in points)
    min_params = min(p["params"] for p in points)
    
    # 设置对数轴
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # 绘制气泡
    for i, p in enumerate(points):
        # 颜色基于参数量
        if max_params != min_params:
            color_val = (p["params"] - min_params) / (max_params - min_params)
        else:
            color_val = 0.5
        color = cmap(color_val * 0.85)
        
        # 气泡大小
        size = 350 + (p["params"] / max_params) * 900
        
        ax.scatter(p["x"], p["y"], s=size, c=[color], 
                   alpha=0.75, edgecolors='white', linewidth=2.5, zorder=5)
        
        # 标签位置 - 在对数空间中计算偏移，更靠近点
        name = p["name"]
        
        # 根据模型名称定制标签位置
        if "DeepSeek" in name:
            # DeepSeek-V3 正下方
            x_offset, y_offset = 1.05, 1.0 / 1.2
            ha, va = 'center', 'top'
        elif "Qwen2.5-0.5B" in name or "Qwen2.5-0.5" in name:
            # Qwen2.5-0.5B 右上角
            x_offset, y_offset = 1.05, 1.12
            ha, va = 'left', 'bottom'
        elif name == "Qwen2.5-3B":
            # Qwen2.5-3B 右上角
            x_offset, y_offset = 1.3, 1.12
            ha, va = 'center', 'bottom'
        elif "Qwen2.5-32B" in name or "Qwen2.5-32" in name:
            # Qwen2.5-32B 正上方
            x_offset, y_offset = 0.9, 1.2
            ha, va = 'center', 'bottom'
        elif "Qwen2.5-7B" in name or name == "Qwen2.5-7B":
            # Qwen2.5-7B 正下方
            x_offset, y_offset = 1.0, 1 / 1.2
            ha, va = 'center', 'top'
        elif "Qwen2.5-72B" in name or "Qwen2.5-72" in name:
            # Qwen2.5-72B 正下方
            x_offset, y_offset = 0.68, 1.05
            ha, va = 'center', 'top'
        elif "Llama-3-8B" in name or "Llama-3-8" in name:
            # Llama-3-8B 右边
            x_offset, y_offset = 1.15, 1.0
            ha, va = 'left', 'center'
        elif "Llama-3.1-70B" in name or "Llama-3.1-70" in name:
            # Llama-3.1-70B 下方
            x_offset, y_offset = 0.85, 1 / 1.25
            ha, va = 'center', 'top'
        else:
            # 其他模型交替方向避免重叠
            direction = i % 4
            if direction == 0:
                x_offset, y_offset = 1.0, 1.12
                ha, va = 'center', 'bottom'
            elif direction == 1:
                # 右上角
                x_offset, y_offset = 1.15, 1.08
                ha, va = 'left', 'bottom'
            elif direction == 2:
                x_offset, y_offset = 1.0, 1 / 1.12
                ha, va = 'center', 'top'
            else:
                # 左上角
                x_offset, y_offset = 1 / 1.15, 1.08
                ha, va = 'right', 'bottom'
        
        # 添加指示线连接标签和点
        ax.annotate(name,
                    xy=(p["x"], p["y"]),
                    xytext=(p["x"] * x_offset, p["y"] * y_offset),
                    fontsize=12,
                    fontweight='bold',
                    color='#1e293b',
                    ha=ha,
                    va=va,
                    bbox=dict(boxstyle='round,pad=0.3',
                             facecolor='white',
                             edgecolor='#94a3b8',
                             alpha=0.9,
                             linewidth=1),
                    zorder=10,
                    arrowprops=dict(arrowstyle='-',
                                   color='#94a3b8',
                                   lw=0.8,
                                   alpha=0.7,
                                   connectionstyle='arc3,rad=0'))
    
    # 坐标轴范围 - 更紧凑，减少空白
    x_vals = [p["x"] for p in points]
    y_vals = [p["y"] for p in points]
    ax.set_xlim(min(x_vals) * 0.9, max(x_vals) * 1.1)
    ax.set_ylim(min(y_vals) * 0.85, max(y_vals) * 1.15)
    
    # 坐标轴标签
    ax.set_xlabel("Training Memory (GB)", fontsize=18, fontweight='bold', labelpad=12)
    ax.set_ylabel("Training FLOPs (TFLOPs)", fontsize=18, fontweight='bold', labelpad=12)
    
    # 设置横轴刻度标签
    from matplotlib.ticker import FuncFormatter, LogLocator
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=6))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x)}' if x >= 1 else f'{x:.1f}'))
    
    bs = config.get('batch_size', 8) if config else 8
    seq_len = config.get('seq_len', 512) if config else 512
    
    ax.set_title(f"Training Compute vs Memory (Log Scale)\nBatch Size: {bs}, Seq Len: {seq_len}", 
                 fontsize=20, fontweight='bold', pad=15)
    
    # Grid - 对数网格
    ax.grid(True, alpha=0.25, linestyle='-', color='#888888', which='both')
    ax.set_axisbelow(True)
    
    # Spine
    for spine in ax.spines.values():
        spine.set_edgecolor('#94a3b8')
        spine.set_linewidth(1.5)
    
    ax.tick_params(axis='both', labelsize=14)
    
    # 颜色条图例
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(min_params, max_params))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, aspect=25, pad=0.02)
    cbar.set_label('Model Parameters (B)', fontsize=14, fontweight='bold')
    cbar.ax.tick_params(labelsize=12)
    
    # MoE 图例 - 更大的标记
    legend_elements = [
        Line2D([0], [0], marker='*', color='w', markerfacecolor='#64748b', 
               markersize=20, label='* MoE model')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=13,
              framealpha=0.95, edgecolor='#94a3b8', handletextpad=0.5)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "training_flops_vs_memory.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_fwd_bwd_time_comparison(models, output_dir="bench", config=None):
    """
    绘制各模型 Forward vs Backward 时间对比图
    """
    if not models:
        print("No valid model data for time comparison chart")
        return
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Prepare data
    names = []
    fwd_times = []
    bwd_times = []
    
    for m in models:
        # 简化名称
        name = m["name"]
        name = name.replace("Qwen2.5-", "Qwen-")
        name = name.replace("Meta-Llama-", "LLaMA-")
        name = name.replace("Mistral-7B-v0.1", "Mistral-7B")
        if m["is_moe"]:
            name += "*"
        names.append(name)
        
        fwd = m.get("estimated_total_time_fwd_s", 0)
        fwd_bwd = m.get("estimated_total_time_fwd_bwd_s", 0)
        fwd_times.append(fwd)
        bwd_times.append(fwd_bwd)
    
    x = np.arange(len(names))
    width = 0.38
    
    # 配色
    fwd_color = '#3b82f6'   # 蓝色
    bwd_color = '#ef4444'   # 红色
    
    # 绘制柱状图
    bars1 = ax.bar(x - width/2, fwd_times, width,
                   label='Forward', color=fwd_color,
                   alpha=0.85, edgecolor='white', linewidth=2)
    
    bars2 = ax.bar(x + width/2, bwd_times, width,
                   label='Forward+Backward (Training)', color=bwd_color,
                   alpha=0.85, edgecolor='white', linewidth=2)
    
    # 数值标签 - 根据数值大小调整位置
    max_time = max(max(fwd_times), max(bwd_times)) if fwd_times and bwd_times else 1
    
    for i, (bar, val) in enumerate(zip(bars1, fwd_times)):
        if val > 0:
            height = bar.get_height()
            # 小数值增加额外垂直偏移，避免标签靠太近
            if val < max_time * 0.1:  # 小于最大值10%的，增加偏移
                y_offset = 8
            else:
                y_offset = 4
            ax.annotate(f'{val:.2f}', 
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, y_offset), textcoords='offset points',
                        ha='center', va='bottom', fontsize=10,
                        fontweight='bold', color='#1e40af')
    
    for i, (bar, val) in enumerate(zip(bars2, bwd_times)):
        if val > 0:
            height = bar.get_height()
            # 小数值增加额外垂直偏移
            if val < max_time * 0.1:
                y_offset = 8
            else:
                y_offset = 4
            ax.annotate(f'{val:.2f}',
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, y_offset), textcoords='offset points',
                        ha='center', va='bottom', fontsize=10,
                        fontweight='bold', color='#b91c1c')
    
    # 样式
    ax.set_xlabel("Model", fontsize=15, fontweight='bold', labelpad=12)
    ax.set_ylabel("Time (seconds)", fontsize=15, fontweight='bold', labelpad=12)
    
    bs = config.get('batch_size', 8) if config else 8
    seq_len = config.get('seq_len', 512) if config else 512
    
    ax.set_title(f"Inference vs Training Time by Model\nBatch Size: {bs}, Seq Len: {seq_len}",
                 fontsize=17, fontweight='bold', pad=15)
    
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=11, fontweight='medium')
    
    # Grid
    ax.grid(True, alpha=0.2, linestyle='-', color='#888888', axis='y')
    ax.set_axisbelow(True)
    ax.set_facecolor('#fafafa')
    
    # Spine
    for spine in ax.spines.values():
        spine.set_edgecolor('#94a3b8')
        spine.set_linewidth(1.5)
    
    ax.tick_params(axis='both', labelsize=11)
    
    # Legend
    ax.legend(loc='upper left', fontsize=12, framealpha=0.95,
              edgecolor='#94a3b8', fancybox=True)
    
    # Y轴范围
    max_val = max(max(fwd_times), max(bwd_times)) if fwd_times and bwd_times else 1
    ax.set_ylim(0, max_val * 1.15)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "fwd_bwd_time_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"✓ Saved: {output_path}")
    plt.close()


def generate_charts(json_file="bench/measured_data.json", output_dir="bench"):
    """Main function to generate all charts"""
    
    # Load data
    data = load_data(json_file)
    os.makedirs(output_dir, exist_ok=True)
    
    # Get metadata
    config = data['metadata'].get('config', {})
    bs = config.get('batch_size', 8)
    seq_len = config.get('seq_len', 512)
    device = data['metadata'].get('device', 'N/A')
    gpu = data['metadata'].get('cuda_device', 'N/A')
    
    print("=" * 60)
    print("Generating Charts")
    print("=" * 60)
    print(f"Data source: {json_file}")
    print(f"Test time: {data['metadata']['timestamp']}")
    print(f"Device: {device} | GPU: {gpu}")
    print(f"Batch Size: {bs}, Seq Len: {seq_len}")
    print()
    
    # Extract model data
    models = extract_model_data(data)
    
    if not models:
        print("No valid test data found!")
        return
    
    print(f"Found {len(models)} models with valid data\n")
    
    # Generate charts
    print("[1/2] Generating Training FLOPs vs Memory chart...")
    plot_training_flops_vs_memory(models, output_dir, config)
    
    print("[2/2] Generating Forward vs Backward Time comparison chart...")
    plot_fwd_bwd_time_comparison(models, output_dir, config)
    
    print("\n" + "=" * 60)
    print("✅ All charts generated successfully!")
    print("=" * 60)


if __name__ == "__main__":
    generate_charts()
