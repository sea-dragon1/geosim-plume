import torch
import numpy as np
import torch.nn as nn
import logging
from datetime import datetime
from matplotlib import pyplot as plt
import os
# utils_visual.py
def print_model_summary(model: nn.Module, model_name: str = "Model", log_dir: str = "logs"):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "model_summary.log")
    logging.basicConfig(filename=log_path, level=logging.INFO, 
                        format='%(asctime)s | %(message)s', datefmt='%m-%d %H:%M')

    total_params = 0
    submodel_params = []

    print("\n" + "="*70)
    print(f"{' '*20} {model_name.upper()} INSIGHTS {' '*20}")
    print("="*70)

    # 关键修复：list(model) 而不是 model.modules
    models_to_iter = list(model) if isinstance(model, nn.ModuleList) else [model]

    for i, sub_model in enumerate(models_to_iter):
        param_count = sum(p.numel() for p in sub_model.parameters() if p.requires_grad)
        total_params += param_count
        submodel_params.append(param_count)

        print(f"\n[Instance {i+1}/{len(models_to_iter)}]")
        print(f"   {'Trainable Params':<18}: {param_count:,} ({param_count/1e3:.2f} K)")
        
        if hasattr(sub_model, 'time_attn') and sub_model.time_attn is not None:
            attn_params = sum(p.numel() for p in sub_model.time_attn.parameters())
            print(f"   └─ TimeAttention      : {attn_params:,} ({attn_params/1e3:.1f} K)")

        print(f"   {'Structure':<18}:")
        print(sub_model)

    print("\n" + "-"*70)
    if isinstance(model, nn.ModuleList):
        avg = sum(submodel_params) // len(submodel_params)
        print(f"{'TOTAL PARAMS':<25}: {total_params:,} ({total_params/1e6:.3f} M)")
        print(f"{'PER OUTPUT':<25}: {avg:,} params")
    else:
        print(f"{'TOTAL PARAMS':<25}: {total_params:,} ({total_params/1e6:.3f} M)")
    print("="*70 + "\n")

    logging.info(f"[{model_name}] Total: {total_params:,} | Per: {avg if isinstance(model, nn.ModuleList) else total_params:,}")
    print(f"Summary → {log_path}")

def loss_plot(loss_history: list, loss_name = 'Training_Loss', save_path: str = "experiments"):
    '''
    绘制损失曲线并保存，默认保存路径为 experiments 文件夹下。
    参数:
        loss_history (list): 损失值列表。
        loss_name (str): 损失名称，用于图表标题和文件名。
        save_path (str): 保存图表的路径。
    返回:
        None
    '''

    
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, label=loss_name, color='blue')
    plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{loss_name} Curve')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_path, f"{loss_name}.png"))
    plt.close()
    print(f"Loss curve saved to {os.path.join(save_path, f'{loss_name}.png')}")

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Union

def field_plot(
    img_data: Union[np.ndarray, "torch.Tensor"],
    loss_name: str = 'inverted_initial_field',
    save_path: str = "experiments",
    cmap: str = 'viridis',
    title: str = None,
    xlabel: str = 'X',
    ylabel: str = 'Y',
    dpi: int = 150,
    figsize: tuple = (6, 5)
) -> None:
    """
    绘制2D场数据并保存为图像文件
    
    参数:
        img_data: 2D numpy数组或PyTorch张量（支持CPU/GPU）
        loss_name: 保存文件名前缀（不含扩展名）
        save_path: 保存目录路径
        cmap: matplotlib colormap名称（默认'viridis'）
        title: 图像标题（若为None则使用loss_name）
        xlabel/ylabel: 坐标轴标签
        dpi: 保存图像分辨率
        figsize: matplotlib图形尺寸 (宽, 高)
    
    异常:
        ValueError: 当输入数据非2D或为空时抛出
        TypeError: 当输入类型不支持时抛出
    """
    # 1. 转换为numpy数组（支持PyTorch张量）
    if hasattr(img_data, 'detach'):  # PyTorch张量
        img_data = img_data.detach().cpu().numpy()
    elif not isinstance(img_data, np.ndarray):
        raise TypeError(f"Unsupported data type: {type(img_data)}. "
                        "Expected np.ndarray or torch.Tensor")
    
    # 2. 验证并处理数据维度
    if img_data.size == 0:
        raise ValueError("Input data is empty")
    
    # 压缩单维度（如(1, H, W) -> (H, W)）
    img_data = np.squeeze(img_data)
    
    if img_data.ndim != 2:
        raise ValueError(f"Expected 2D data after squeeze, got {img_data.ndim}D array with shape {img_data.shape}. "
                         "For multi-channel data, select a channel first (e.g., img_data[..., 0])")
    
    # 3. 处理NaN/Inf（可选：替换为均值或保留原样）
    if np.any(np.isnan(img_data)) or np.any(np.isinf(img_data)):
        # 创建掩码副本避免修改原数据
        img_data = np.ma.masked_invalid(img_data)
        print(f"Warning: Data contains {np.sum(np.isnan(img_data))} NaNs and "
              f"{np.sum(np.isinf(img_data))} Infs (masked in plot)")
    
    # 4. 创建保存目录
    os.makedirs(save_path, exist_ok=True)
    
    # 5. 绘图
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(img_data, cmap=cmap, aspect='auto', origin='lower')
    
    # 设置标签和标题
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title if title else loss_name, fontsize=12, pad=10)
    
    # 添加颜色条并优化布局
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Value', rotation=270, labelpad=15)
    
    plt.tight_layout()  # 防止标签被截断
    
    # 6. 保存并清理
    save_file = os.path.join(save_path, f"{loss_name}.png")
    plt.savefig(save_file, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✓ Field plot saved to: {save_file}")
    print(f"  Data stats - min: {np.nanmin(img_data):.4f}, max: {np.nanmax(img_data):.4f}, "
          f"mean: {np.nanmean(img_data):.4f}")