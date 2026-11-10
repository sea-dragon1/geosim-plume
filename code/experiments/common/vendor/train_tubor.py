# coding=utf-8
import argparse
import numpy as np
import os
import random
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
import shutil
import logging

# [vendor] 路径改为基于 __file__ 推导，不依赖当前工作目录。
# 上游用 sys.path.append("..")（相对 cwd）+ 依赖脚本目录在 sys.path 首位的平级 import，
# 一旦从别处调用就会失败。这里显式固定三处搜索路径。
_HERE = os.path.dirname(os.path.abspath(__file__))              # .../common/vendor
_COMMON = os.path.dirname(_HERE)                                # .../common
# .../common/vendor -> common -> experiments -> code -> code/vendor
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "vendor"))

# code/vendor 必须排在**最前**：本脚本用 `import metrics`（MSE/RMSE/L2RE/MaxError），
# 那是 code/vendor/metrics.py。而 common/ 在本项目里也有一个评估模块 —— 早期它
# 同名 metrics.py，一旦 common/ 排到前面就会把它盖掉，导致
# `AttributeError: module 'metrics' has no attribute 'MSE'`。
# （该冲突已通过把 common 的模块改名为 eval_metrics.py 根治，这里再显式保证顺序。）
for _p in (_COMMON, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
if _REPO in sys.path:
    sys.path.remove(_REPO)
sys.path.insert(0, _REPO)

from utils_tubor import *
from utils_visual import loss_plot
from model import MPNN, MFGNO
# [E02/vendor] 统一的逐场 L2RE 损失：等权与速度加权共用同一公式，只改权重
from losses import per_field_l2re_loss, get_weights

import metrics
import pandas_data
torch.autograd.set_detect_anomaly(False)
def get_dataset(args):
    dataset_args = args["dataset"]
    print("SingleDataset")
    train_data = MPNNDatasetSingle(dataset_args["file_name"],
                                  dataset_args["saved_folder"],
                                  reduced_resolution=dataset_args["reduced_resolution"],
                                  reduced_resolution_t=dataset_args["reduced_resolution_t"],
                                  reduced_batch=dataset_args["reduced_batch"],
                                  test_ratio=dataset_args["test_ratio"],
                                  if_test=False,
                                  variables=args["variables"],
                                  pde_name=args['pde_name'],
                                  total_time_steps=args.get("resolution_t",-1))
    val_data = MPNNDatasetSingle(dataset_args["file_name"],
                                 dataset_args["saved_folder"],
                                 reduced_resolution=dataset_args["reduced_resolution"],
                                 reduced_resolution_t=dataset_args["reduced_resolution_t"],
                                 reduced_batch=dataset_args["reduced_batch"],
                                 test_ratio=dataset_args["test_ratio"],
                                 if_test=True,
                                 variables=args["variables"],
                                 pde_name=args['pde_name'],
                                 total_time_steps=args.get("resolution_t",-1))
    return train_data, val_data


def get_dataloader(train_data, val_data, args):
    dataloader_args = args["dataloader"]
    train_loader = DataLoader(train_data, shuffle=True, **dataloader_args)
    val_loader = DataLoader(val_data, shuffle=False, **dataloader_args)
    return train_loader, val_loader


def get_model(args, pde: PDE):
    if args['model_id'] == 'MPNN':
        model = nn.ModuleList([MPNN(pde, time_window=args["time_window"], hidden_features=args["model"]["hidden_features"],
                    hidden_layers=args["model"]["hidden_layer"], eq_variables=args["variables"]) for _ in range(args["num_outputs"])])
    elif args['model_id'] == 'MFGNO':
        # [E02/vendor] zero_var_diff=True 构造 B 变体（显式多场差分槽置零）；
        # 结构与参数量与 C 完全一致，仅该槽取值不同。默认 False 即 C。
        model = MFGNO(pde, time_window=args["time_window"], hidden_features=args["model"]["hidden_features"],
                hidden_layers=args["model"]["hidden_layer"], num_physics_vars=args["num_outputs"], eq_variables=args["variables"],
                zero_var_diff=args["model"].get("zero_var_diff", False))
    else:
        raise ValueError("未知 model_id {!r}，发布包只支持 MPNN / MFGNO".format(
            args['model_id']))
    return model


def train_loop(dataloader, model, loss_fn, optimizer, device, graph_creator, unrolling):
    model.train()
    losses = []
    start_time = time.time()

    # 用于标记是否已创建静态图
    static_graph_built = False
    graph = None
    # 空间坐标（所有样本共享同一网格，只取一次即可）
    x_static = None
    # 记录已建图的 batch_size，最后一批样本不足 bs 时（bs 变化）需重建图
    batch_size_cst = None

    for u, x, t, variables in dataloader:
        # u: [B, T, nx, V]; B=bn, T=nt, nx=nx*ny, V=num_vars
        batch_size = u.shape[0]
        num_outputs = u.shape[-1]

        # 随机选择 unrolling 步数
        unrolled_graphs = random.choice(unrolling)
        assert (graph_creator.nt - graph_creator.tw - (graph_creator.tw * unrolled_graphs) > graph_creator.tw)
        steps = [tt for tt in range(graph_creator.tw, graph_creator.nt - graph_creator.tw - (graph_creator.tw * unrolled_graphs) + 1)]
        random_steps = random.choices(steps, k=batch_size)

        # 提取输入数据与标签
        data, labels = graph_creator.create_data(u, random_steps)  # 仍使用原有方法（已优化内部为 stack）

        # x 形状 [nx, d]，所有样本相同，取第一个 batch 的 x 缓存（供建图 + 物理损失复用）
        if x_static is None:
            x_static = x[0] if x.dim() == 3 else x  # 确保形状 [nx, d]
            x_static = x_static.to(device)

        # ========== 静态图构建（仅第一次）+ 每 batch 更新动态字段 ==========
        # 注意：最后一批样本数可能不足 bs（batch_size 变化），此时需重建图保证 x/x_pos 的 bs 一致
        if not static_graph_built or batch_size_cst != batch_size:
            graph, _ = graph_creator.build_static_graph(x_static, data, labels, t, random_steps, batch_size, variables, device)
            static_graph_built = True
            batch_size_cst = batch_size
        else:
            # 复用静态图（x_pos/edge_index/batch/variables），只更新输入/标签/时间
            graph = graph_creator.update_graph_input(graph, data, labels, random_steps, t, device)

        # ========== Unrolling（推前向技巧） ==========
        with torch.no_grad():
            for _ in range(unrolled_graphs):
                if args['model_id'] == 'MFGNO':
                    pred = model(graph, t, batch_size)  # 输出已在 GPU
                else:
                    pred = torch.empty_like(graph.y)
                    for i in range(num_outputs):
                        pred[..., i] = model[i](graph, t, i, batch_size)

                # 更新下一个时间步的 steps 和 labels
                random_steps = [rs + graph_creator.tw for rs in random_steps]
                _, labels = graph_creator.create_data(u, random_steps)
                # 更新图为下一个时刻
                graph = graph_creator.update_graph_data(graph, pred, labels, random_steps, t, device)

        # ========== 最终预测（用于 loss） ==========
        if args['model_id'] == 'MFGNO':
            pred = model(graph, t, batch_size)
            pred = pred[:, :graph.y.shape[1], :]
        else:
            pred = torch.empty_like(graph.y)
            for i in range(num_outputs):
                output = model[i](graph, t, i, batch_size)
                pred[..., i] = output[:, :graph.y.shape[1]]
        pred = pred.float()

        loss = loss_fn(pred, graph.y)
        # 论文口径：统一的逐场 L2RE 目标（等权或速度加权由 loss_mode 决定）
        if args["if_physical_loss"]:
            pred5 = to_PDEBench_format(pred, batch_size, graph_creator.pde)
            gt5 = to_PDEBench_format(graph.y, batch_size, graph_creator.pde)
            loss = per_field_l2re_loss(pred5, gt5,
                                       get_weights(args.get("loss_mode", "equal")))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    end_time = time.time()
    return np.mean(losses), end_time - start_time


def val_loop(dataloader, model, loss_fn, device, graph_creator):
    model.eval()
    losses = []

    for u, x, t, variables in dataloader:
        batch_size = u.shape[0]
        num_outputs = u.shape[-1]

        # 初始 steps（起始点固定为 tw）
        steps = [graph_creator.tw] * batch_size
        data, labels = graph_creator.create_data(u, steps)

        # 每个 batch 重建静态图
        x_static = x[0] if x.dim() == 3 else x
        graph, _ = graph_creator.build_static_graph(x_static, data, labels, t, steps, batch_size, variables, device)

        # 使用列表收集各时间步的预测和真实值
        targets = [graph.y]      # [bs*nx, tw, v]
        preds = []

        # 初始预测
        with torch.no_grad():
            if args['model_id'] == 'MFGNO':
                output = model(graph, t, batch_size)
            else:
                output = torch.empty_like(graph.y)
                for i in range(num_outputs):
                    output[..., i] = model[i](graph, t, i, batch_size)
            preds.append(output)

        # 自回归滚动（从 2*tw 到 nt-tw，步长为 tw）
        for step in range(2 * graph_creator.tw, graph_creator.nt - graph_creator.tw + 1, graph_creator.tw):
            steps = [step] * batch_size
            _, labels = graph_creator.create_data(u, steps)
            # 更新图为下一时间步（使用上一步的 output 作为预测）
            graph = graph_creator.update_graph_data(graph, output, labels, steps, t, device)
            targets.append(graph.y)

            with torch.no_grad():
                if args['model_id'] == 'MFGNO':
                    output = model(graph, t, batch_size)
                else:
                    output = torch.empty_like(graph.y)
                    for i in range(num_outputs):
                        output[..., i] = model[i](graph, t, i, batch_size)
                preds.append(output)

        # 合并所有时间步
        target = torch.cat(targets, dim=-2)   # [bs*nx, total_t, v]
        pred = torch.cat(preds, dim=-2)       # 同上

        loss = loss_fn(pred, target)
        losses.append(loss.item())

    return np.mean(losses)

def test_loop(dataloader, model, device, graph_creator, metric_names=['MSE', 'RMSE', 'L2RE', 'MaxError', 'SSIM']):
    model.eval()
    res_dict = {}
    # pred_all = []
    # y_all = []
    for name in metric_names:
        res_dict[name] = []
    
    total_infer_time = 0.0
    num_infer_calls = 0

    for u, x, t, variables in dataloader:
        batch_size = u.shape[0]
        num_outputs = u.shape[-1]

        # 初始 steps
        steps = [graph_creator.tw] * batch_size
        data, labels = graph_creator.create_data(u, steps)

        # 每个 batch 重建静态图
        x_static = x[0] if x.dim() == 3 else x
        graph, _ = graph_creator.build_static_graph(x_static, data, labels, t, steps, batch_size, variables, device)

        # 使用列表收集各时间步输出，避免动态 cat
        targets_list = [graph.y]   # graph.y 形状 [bs*nx, tw, v]
        preds_list = []

        # 初始预测
        with torch.no_grad():
            # 计时（略，与原代码相同）
            if args['model_id'] == 'MFGNO':
                # --- 开始计时 ---
                if device.type == 'cuda':
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                else:
                    start_time = time.perf_counter()
                # -----------------
                # Single model predicts all fields together
                output = model(graph, t, batch_size)
                # --- 结束计时 ---
                if device.type == 'cuda':
                    end_event.record()
                    torch.cuda.synchronize()
                    total_infer_time += start_event.elapsed_time(end_event)
                else:
                    end_time = time.perf_counter()
                    total_infer_time += (end_time - start_time)
                num_infer_calls += 1
                
            else:
                # Separate models for each variable
                output = torch.empty_like(graph.y)
                # --- 开始计时 ---
                if device.type == 'cuda':
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                else:
                    start_time = time.perf_counter()
                # -----------------
                for i in range(num_outputs):
                    output[..., i] = model[i](graph, t, i, batch_size)
                # --- 结束计时 ---
                if device.type == 'cuda':
                    end_event.record()
                    torch.cuda.synchronize()
                    total_infer_time += start_event.elapsed_time(end_event)
                else:
                    end_time = time.perf_counter()
                    total_infer_time += (end_time - start_time)
                num_infer_calls += 1
            preds_list.append(output)

        # 时间步进循环
        for step in range(2 * graph_creator.tw, graph_creator.nt - graph_creator.tw + 1, graph_creator.tw):
            steps = [step] * batch_size
            _, labels = graph_creator.create_data(u, steps)
            graph = graph_creator.update_graph_data(graph, output, labels, steps, t, device)
            targets_list.append(graph.y)

            with torch.no_grad():
                if args['model_id'] == 'MFGNO':
                    # --- 开始计时 ---
                    if device.type == 'cuda':
                        start_event = torch.cuda.Event(enable_timing=True)
                        end_event = torch.cuda.Event(enable_timing=True)
                        start_event.record()
                    else:
                        start_time = time.perf_counter()
                    # -----------------
                    output = model(graph, t, batch_size)
                                        # --- 结束计时 ---
                    if device.type == 'cuda':
                        end_event.record()
                        torch.cuda.synchronize()
                        total_infer_time += start_event.elapsed_time(end_event)
                    else:
                        end_time = time.perf_counter()
                        total_infer_time += (end_time - start_time)
                    num_infer_calls += 1
                else:
                    output = torch.empty_like(graph.y)
                    # --- 开始计时 ---
                    if device.type == 'cuda':
                        start_event = torch.cuda.Event(enable_timing=True)
                        end_event = torch.cuda.Event(enable_timing=True)
                        start_event.record()
                    else:
                        start_time = time.perf_counter()
                    # -----------------
                    for i in range(num_outputs):
                        output[..., i] = model[i](graph, t, i, batch_size)
                    # --- 结束计时 ---
                    if device.type == 'cuda':
                        end_event.record()
                        torch.cuda.synchronize()
                        total_infer_time += start_event.elapsed_time(end_event)
                    else:
                        end_time = time.perf_counter()
                        total_infer_time += (end_time - start_time)
                    num_infer_calls += 1
                preds_list.append(output)

        # 合并所有时间步
        target = torch.cat(targets_list, dim=-2)  # [bs*nx, total_t, v]
        pred = torch.cat(preds_list, dim=-2)
        pred = to_PDEBench_format(pred, batch_size, graph_creator.pde)
        target = to_PDEBench_format(target, batch_size, graph_creator.pde)

        for name in metric_names:
            metric_fn = getattr(metrics, name)
            res_dict[name].append(metric_fn(pred, target))
    # === 计算平均推理时间 ===
    if num_infer_calls > 0:
        if device.type == 'cuda':
            avg_infer_time_s = total_infer_time / num_infer_calls / 1000
            print(f"Average inference time per call: {avg_infer_time_s:.6f} s in GPU")
        else:
            avg_infer_time_s = total_infer_time / num_infer_calls
            print(f"Average inference time per call: {avg_infer_time_s:.6f} s in CPU")
        res_dict["avg_infer_time_s"] = avg_infer_time_s
    else:
        print("No inference calls recorded.")

    # post process
    for name in metric_names:
        res_list = res_dict[name]
        if name == "MaxError":
            res = torch.stack(res_list, dim=0)
            res, _ = torch.max(res, dim=0)
        else:
            res = torch.cat(res_list, dim=0) # (num_samples, v)
            res = torch.mean(res, dim=0) # (v)
        res_dict[name] = res


    pred_np = pred.detach().cpu().numpy()   # (bs, H, W, T, v)
    y_np = target.detach().cpu().numpy()         # (bs, H, W, T, v)
    print('pred_np shape:', pred_np.shape) # pred_np shape: (8, 128, 128, 14, 6)
    # pred_all.append(pred_np)
    # y_all.append(y_np)           # 加入列表

    pred_all = pred_np
    y_all = y_np
    # 循环结束后
    # pred_all = np.stack(pred_all, axis=0)       # (9,100,401,309)
    pred_all = pred_all.transpose(0, 3, 1, 2, 4)
    # y_all = np.stack(y_all, axis=0)
    y_all = y_all.transpose(0, 3, 1, 2, 4)
    res_all = y_all - pred_all
    relative_res_all = res_all / (y_all + 1e-20)
    print('pred_all shape:', pred_all.shape)

    # 定义要保存的文件及其对应的数据
    files_to_save = {
        'pred.hdf5': pred_all,
        'test.hdf5': y_all,
        'res.hdf5': res_all,
        'relative_res.hdf5': relative_res_all
    }


    # 输出平均推理时间等
    return res_dict, files_to_save

def test_and_save_results(args, model, log_path, device, val_loader, output_path, graph_creator):
    os.makedirs(os.path.join(log_path,"test"), exist_ok=True)
    test_log_file = os.path.join(log_path,"test", f"{args['exp_id']}test.log")
    logging.basicConfig(filename=test_log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f"Test mode, load checkpoint from {args.get('model_path', '')}")

    print("start testing...")
    # coding...
    res,files_to_save = test_loop(val_loader, model, device, graph_creator)
    print(res)

    # 1.保存指标到log
    for u, v in res.items():
        if u == 'avg_infer_time_s':
            logging.info(f"{u}\t{v:.6f}s")
        else:
            dim = len(v)
            if dim == 1:
                print(u, "{0:.6f}".format(v.item()))
            else:
                for i in range(dim):
                    if i == 0:
                        print(u, "\t{0:.6f}".format(v[i].item()), end='\t')
                        logging.info(f"{u}\t{v[i].item():.6f}")
                    else:
                        print("{0:.6f}".format(v[i].item()), end='\t')
                        logging.info(f"{u}\t{v[i].item():.6f}")
                print("")
    print("Done")

    # 2.Save metrics to global CSV (as in MPNN)
    pandas_data.pd_metrics(
        err_dict=res,
        exp_id=args['exp_id'],
        model_id=args['model_id'],
        data_name=os.path.splitext(args["dataset"]["file_name"])[0],
        csv_file=f'{args["root_path"]}/test_log.csv'
    )

    # Save full experiment config + results
    results_dict = {
        "exp_id": args["exp_id"],
        "model_id": args["model_id"],
        "hidden_features": args["model"].get("hidden_features", "N/A"),
        "lr": args["optimizer"]["lr"],
        "weight_decay": args["optimizer"].get("weight_decay", 0),
        "unrolling": args["unrolling"],
        "dataset": args["dataset"]["file_name"],
        "pde_name": args["pde_name"]
    }
    pandas_data.pd_results(
        err_dict=res,
        results_dict=results_dict,
        csv_file=f'{args["root_path"]}/results_log.csv'
    )

    # 3. 保存测试结果到hdf5并绘图

    var_names = ["ro", "pr", "tk", "nu", "vx", "vy"]

    output_path_id = output_path + f'/{args["exp_id"]}'
    if not os.path.exists(output_path_id):
        os.makedirs(output_path_id)

    # 循环保存每个文件
    for filename, data in files_to_save.items():
        file_path = os.path.join(output_path_id, filename)
        with h5py.File(file_path, 'w') as f:
            for i, name in enumerate(var_names):
                f.create_dataset(name,
                                data=data[..., i],        # 提取第 i 个通道
                                compression='gzip',       # 启用 gzip 压缩
                                compression_opts=4)       # 压缩等级
                
        pandas_data.h5fields_to_gif(
            h5_path=os.path.join(output_path_id, filename),
            out_dir=f"{output_path_id}/data_info/{filename}",
            fields=var_names,
            fps=10,
            sample_idx=0
        )

    return res

def save_checkpoint(obj, filepath, max_retries=3):
    """Save to memory buffer, then write to temp file + atomic rename.

    Avoids torch.save writing directly to Lustre (zip corruption on parallel FS).
    """
    import uuid
    import io
    import time as _time

    # Serialize to memory first
    buffer = io.BytesIO()
    torch.save(obj, buffer)
    data = buffer.getvalue()
    buffer.close()

    # Write to temp file in target directory, then atomic rename
    dirpath = os.path.dirname(filepath)
    tmp_path = os.path.join(dirpath, f".tmp_{uuid.uuid4().hex}.pt")
    for attempt in range(max_retries):
        try:
            with open(tmp_path, 'wb') as f:
                f.write(data)
            os.rename(tmp_path, filepath)
            return
        except (RuntimeError, OSError):
            for p in [tmp_path]:
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass
            if attempt == max_retries - 1:
                raise
            print(f"Checkpoint save failed (attempt {attempt+1}/{max_retries}), retrying...")
            _time.sleep(2)


def main(args):
    """训练/测试模型的主函数"""
    
    # ==================== 初始化设置 ====================
    setup_seed(args["seed"])  # 设置随机种子
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 模型保存路径设置
    exp_dir = os.path.join(args['root_path'], args['exp_id'])
    output_path = os.path.join(exp_dir, 'output')
    checkpoints_path = os.path.join(args['root_path'], args['exp_id'], 'checkpoints')
    log_path = os.path.join(args['root_path'], args['exp_id'], 'logs')
    saved_model_name = args["exp_id"]

    os.makedirs(log_path, exist_ok=True)
    os.makedirs(checkpoints_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(exp_dir, exist_ok=True)

    # log_path
    loss_history_file = os.path.join(log_path, f'{log_path}/{args["exp_id"]}_loss_history.npy')
    val_loss_history_file = os.path.join(log_path, f'{log_path}/{args["exp_id"]}_val_loss_history.npy')
    train_png = f'{args["exp_id"]}_Training_Loss' # f'{args["exp_id"]}_Training_Loss'.png
    val_png = f'{args["exp_id"]}_Validation_Loss' # f'{args["exp_id"]}_Validation_Loss'.png

    # output_path = exp_dir+f'/{args["model_id"]}/output/{args["exp_id"]}'          # 可自行修改路径
    
    src_path = cmd_args.config_file               # 源文件，如 configs/train.yml
    dst_path = os.path.join(exp_dir, f'{args["exp_id"]}.yaml')  # 目标路径（可改名）
    shutil.copy(src_path, dst_path)

    # ==================== 加载检查点 ====================
    checkpoint = None
    if (not args["if_training"]) or args["continue_training"]:
        checkpoint = torch.load(args["model_path"])
        print(f"Loaded checkpoint from {args['model_path']}")
    
    # ==================== 优化器参数转换 ====================
    args['optimizer']['lr'] = float(args['optimizer']['lr'])
    args['optimizer']['weight_decay'] = float(args['optimizer']['weight_decay'])
    
    # ==================== PDE定义 ====================
    pde = PDE(
        args["pde_name"],
        variables=args["variables"],
        temporal_domain=eval(args["temporal_domain"]),
        resolution_t=args["resolution_t"],
        spatial_domain=eval(args["spatial_domain"]),
        resolution=args["resolution"],
        reduced_resolution_t=args["dataset"]["reduced_resolution_t"],
        reduced_resolution=args["dataset"]["reduced_resolution"]
    )

    args['pde'] = pde
    
    # ==================== 数据加载 ====================
    train_data, val_data = get_dataset(args)
    train_loader, val_loader = get_dataloader(train_data, val_data, args)
    
    # ==================== 图创建器 ====================
    graph_creator = GraphCreator(pde, neighbors=args["neighbors"],
                                 time_window=args["time_window"]).to(device)
    
    # ==================== 模型初始化 ====================
    model = get_model(args, pde)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")
    
    # ==================== 测试模式 ====================
    if not args["if_training"]:
        model.load_state_dict(checkpoint["model_state_dict"])
        test_and_save_results(
            args,
            model=model.to(device),
            log_path=log_path, 
            device=device, 
            val_loader=val_loader, 
            output_path=output_path, 
            graph_creator=graph_creator
        )

        return
    
    # ==================== 继续训练设置 ====================
    start_epoch = 0
    min_val_loss = torch.inf
    
    if args["continue_training"]:
        print("Continue training from checkpoint")
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint['epoch']
        min_val_loss = checkpoint['loss']
    
    # ==================== 设备 ====================
    model.to(device)

    # ==================== 优化器设置 ====================
    optim_args = args["optimizer"]
    optim_name = optim_args.pop("name")

    if args["continue_training"]:
        optimizer = getattr(torch.optim, optim_name)(model.parameters())
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        optimizer = getattr(torch.optim, optim_name)(model.parameters(), **optim_args)

    # ==================== 学习率调度器 ====================
    sched_args = args["scheduler"]
    sched_name = sched_args.pop("name")
    scheduler = getattr(
        torch.optim.lr_scheduler, 
        sched_name
    )(optimizer, last_epoch=start_epoch-1, **sched_args)
    
    # ==================== 损失函数 ====================
    loss_fn = nn.MSELoss(reduction="mean")
    
    # ==================== 训练循环 ====================
    print(f"Start training from epoch {start_epoch}")
    total_time = 0
    loss_history = []
    val_loss_history = []
    if args["continue_training"]:
        loss_history = np.load(loss_history_file).tolist() if os.path.exists(loss_history_file) else []
        val_loss_history = np.load(val_loss_history_file).tolist() if os.path.exists(val_loss_history_file) else []

    for epoch in range(start_epoch, args["epochs"]):
        epoch_time = 0
        losses = []
        
        # 设置unrolling步数
        max_unrolling = args["unrolling"]
        unrolling = list(range(max_unrolling + 1))
        
        # 每个epoch的训练步骤
        for _ in range(args["unroll_step"] // args["time_window"]):
            loss, time_spend = train_loop(
                train_loader,
                model,
                loss_fn,
                optimizer,
                device,
                graph_creator,
                unrolling
            )
            losses.append(loss)
            epoch_time += time_spend
        
        # 打印epoch信息
        print(f"[Epoch {epoch}] MSELoss: {np.mean(losses)}, time: {epoch_time:.3f}s")

        loss_history.append(losses)
        total_time += epoch_time
        scheduler.step()
        
        # ==================== 保存检查点 ====================
        model_state_dict = model.state_dict()
        saved_path = os.path.join(checkpoints_path, saved_model_name)

        # 保存最新检查点
        save_checkpoint({
            "epoch": epoch + 1,
            "loss": min_val_loss,
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer.state_dict()
        }, saved_path + "-latest.pt")
        # 保存模型时保存一次loss
        # loss_history = np.array(loss_history)
        np.save(loss_history_file, np.array(loss_history))

        # ==================== 验证和保存最佳模型 ====================
        if (epoch + 1) % args["save_period"] == 0:
            print("==================== Validation ====================")
            val_loss = val_loop(val_loader, model, loss_fn, device, graph_creator)
            print(f"[Epoch {epoch}] val_loss: {val_loss}")
            val_loss_history.append(val_loss)
            np.save(val_loss_history_file, val_loss_history)

            print("===================================================")
            
            # 保存最佳模型
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                save_checkpoint({
                    "epoch": epoch + 1,
                    "loss": min_val_loss,
                    "model_state_dict": model_state_dict,
                    "optimizer_state_dict": optimizer.state_dict()
                }, saved_path + "-best.pt")

    
    # ==================== 训练结束 ====================
    print("Training completed.")
    try:
        print(f"Total train time: {total_time}s. Average per epoch: {total_time / (args['epochs'] - start_epoch)}s")
    except ZeroDivisionError:
        print("设定训练epoch等于已经训练epoch")
    # ==================== 保存损失历史 ====================
    loss_history = np.array(loss_history)
    np.save(loss_history_file, loss_history)

    # 绘制损失曲线
    loss_plot(
        loss_history=np.mean(loss_history, axis=1).tolist(),
        loss_name=train_png,
        save_path=log_path
    )
    
    if val_loss_history:
        loss_plot(
            loss_history=val_loss_history,
            loss_name=val_png,
            save_path=log_path
        )
        np.save(val_loss_history_file, val_loss_history)
    
    # ==================== 最终测试 ====================
    if args["if_training"]:
        test_and_save_results(
            args, 
            model=model, 
            log_path=log_path, 
            device=device,
            val_loader=val_loader, 
            output_path=output_path, 
            graph_creator=graph_creator
        )




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="Path to config file")
    cmd_args = parser.parse_args()
    with open(cmd_args.config_file, 'r') as f:
        args = yaml.safe_load(f)
    print(args)

    main(args)