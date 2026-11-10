import pandas as pd
import torch
from datetime import datetime
from typing import Dict
import os

def pd_metrics(
    err_dict: Dict[str, torch.Tensor],
    exp_id: str,
    model_id: str,
    data_name: str = '2D_Mantle_convection_ds_64x64',
    csv_file: str = 'test_log.csv'
):
    """
    将一次训练得到的误差字典追加到 csv 日志（宽表格式）。
    
    参数
    ----
    err_dict : dict, 键为 {'MSE','RMSE','L2RE','MaxError'}，值为 1-D tensor
    model_id : str, 模型名称（任意标识）
    csv_file : str, 日志文件路径，不存在则自动创建
    
    返回
    ----
    pd.DataFrame : 追加后的完整日志表
    """
    # 1. 拼成一行扁平字典
    flat_row = {'exp_id':exp_id, 'model_id': model_id, 'timestamp': datetime.now(), 'data': data_name}
    fields_names = ["ro", "pr", "tk", "nu", "vx", "vy"]
    for metric, vec in err_dict.items():
        try:
            vec.type == torch.Tensor
            vec_np = vec.detach().cpu().numpy()
            for i, val in enumerate(vec_np, start=1):
                flat_row[f'{metric}_{fields_names[i-1]}'] = val
        except:
            flat_row[f'{metric}/s'] = vec

    # 2. 转为 Series 并确保索引（列名）一致
    row_series = pd.Series(flat_row)

    # 3. 直接追加写入（避免 read-modify-write）
    file_exists = os.path.isfile(csv_file)
    row_series.to_frame().T.to_csv(
        csv_file,
        mode='a',
        header=not file_exists,  # 仅首次写 header
        index=False
    )

def pd_results(
    err_dict: Dict[str, torch.Tensor],
    results_dict: Dict,
    csv_file: str = 'test_log.csv'
)   :
    """
    将一次训练得到的误差字典与相关参数保存到 csv 日志（宽表格式）。
    
    参数
    ----
    err_dict : dict, 键为 {'MSE','RMSE','L2RE','MaxError'}，值为 1-D tensor
    results_dict : dict, 模型与训练信息
    csv_file : str, 日志文件路径，不存在则自动创建
    
    返回
    ----
    pd.DataFrame : 追加后的完整日志表
    """


    # 1. 拼成一行扁平字典
    flat_row = dict(results_dict)
    fields_names = ["ro", "pr", "tk", "nu", "vx", "vy"]
    for metric, vec in err_dict.items():
        try:
            vec.type == torch.Tensor
            vec_np = vec.detach().cpu().numpy()
            for i, val in enumerate(vec_np, start=1):
                flat_row[f'{metric}_{fields_names[i-1]}'] = val
        except:
            flat_row[f'{metric}/s'] = vec
    # 2. 转为 Series 并确保索引（列名）一致
    row_series = pd.Series(flat_row)

    # 3. 直接追加写入（避免 read-modify-write）
    file_exists = os.path.isfile(csv_file)
    row_series.to_frame().T.to_csv(
        csv_file,
        mode='a',
        header=not file_exists,  # 仅首次写 header
        index=False
    )

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import imageio
from tqdm import tqdm   # 放在最顶部

def h5fields_to_gif(h5_path: str,
                    out_dir: str,
                    fields: list,
                    fps: int = 10,
                    sample_idx: int = 0):
    """
    把 HDF5 文件中的 2D 物理场做成 GIF 动图（沿 t 轴）。

    Parameters
    ----------
    h5_path : str
        HDF5 文件路径。
    fields : list[str] | None
        需要生成 GIF 的物理场名字列表；None 时自动扫描所有 4D 数据集。
    out_dir : str
        输出目录，不存在会自动创建。
    fps : int
        GIF 帧率。
    sample_idx : int
        选择第几个样本（axis=0），默认 0。
    """
    os.makedirs(out_dir, exist_ok=True)
    cmap_map = {'nu':'plasma','pr':'viridis','ro':'magma',
                'tk':'viridis','vx':'seismic','vy':'seismic'}

    with h5py.File(h5_path,'r') as f:
        if fields is None:
            fields = [k for k,v in f.items() if isinstance(v,h5py.Dataset) and v.ndim==4]

        for name in fields:
            data = f[name]           # (4,100,401,309)
            nt   = data.shape[1]
            frames = []
            for t in tqdm(range(nt), desc=name):
                img = data[sample_idx, t, :, :]
                fig, ax = plt.subplots(figsize=(5.5,4))
                im = ax.imshow(img.T, cmap=cmap_map.get(name,'viridis'),
                               origin='upper')    # 无坐标，自动像素索引
                ax.set_title(f'{name}  t={t}')
                fig.colorbar(im, ax=ax, shrink=0.6)

                fig.canvas.draw()
                buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
                buf = buf.reshape(fig.canvas.get_width_height()[::-1]+(3,))
                frames.append(buf)
                plt.close(fig)

            imageio.mimsave(os.path.join(out_dir,f'{name}.gif'), frames, fps=fps)
            print(f'Saved {name}.gif')

# ==================
# 使用示例
# ==================
if __name__ == '__main__':

    paths = [os.environ.get('GEOSIM_OUTPUT_DIR', 'output')]
    # 示例：取第5个时间步（可修改为0~96之间的任意值）
    # filenames = '2D_Mantle_convection_processed' #  2D_CFD_Rand_M0.1_Eta0.1_Zeta0.1_periodic_128_Train 2D_Mantle_convection'
    filenames = ['test','pred','res', 'relative_res']
    # filenames = ['2D_Mantle_convection_ds_64x64', '2D_Mantle_convection_ds_128x128', '2D_Mantle_convection_ds_256x256']
    # filename = ['2D_Mantle_convection']
    # 只画 vx, vy
    # h5fields_to_gif('2D_mantle_convection.h5', fields=['vx', 'vy'])
    for path in paths:
        for fname in filenames:
            h5fields_to_gif(
                h5_path=f"{path}/{fname}.hdf5",
                out_dir=f"{path}/data_info/{fname}",
                fields=['ro', 'pr', 'vx', 'vy', 'tk', 'nu'],
                fps=10,
                sample_idx=0
            )

# ---------------- 使用示例 ----------------
    errors = {
        'MSE':    torch.tensor([1.0784e-04, 2.2595e-03, 1.2319e-04, 3.6312e-03, 1.1287e-01, 1.1659e-01]),
        'RMSE':   torch.tensor([0.0099, 0.0471, 0.0107, 0.0593, 0.2886, 0.3144]),
        'L2RE':   torch.tensor([0.0100, 0.0286, 0.0107, 0.0607, 0.4946, 1.3418]),
        'MaxError': torch.tensor([0.2257, 1.2238, 0.2158, 1.1876, 10.6438, 12.9643])
    }
    pd_metrics(errors, model_id='UNet2D', data_name='2D_Mantle_convection_ds_64x64')   # 一键追加

