# coding=utf-8
import shutil
import argparse
import numpy as np
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from timeit import default_timer
import yaml
import logging
import h5py

from utils_visual import loss_plot, print_model_summary
from utils import *
from model.fno import FNO1d, FNO2d, FNO3d, FNO_maxwell

# 发布版：改为相对本文件的 vendor 根（原实现依赖当前工作目录）
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import metrics
import pandas_data  # Custom module for saving results to CSV


def get_dataset(args):
    dataset_args = args["dataset"]
    if(dataset_args["single_file"]):
        print("FNODatasetSingle")
        train_data = FNODatasetSingle(dataset_args["file_name"],
                                reduced_resolution=dataset_args["reduced_resolution"],
                                reduced_resolution_t=dataset_args["reduced_resolution_t"],
                                reduced_batch=dataset_args["reduced_batch"],
                                initial_step=args["initial_step"],
                                t_train=args["t_train"],
                                if_test=False,
                                saved_folder = dataset_args["saved_folder"],
                                pde_name=args['pde_name']
                                )
        val_data = FNODatasetSingle(dataset_args["file_name"],
                                reduced_resolution=dataset_args["reduced_resolution"],
                                reduced_resolution_t=dataset_args["reduced_resolution_t"],
                                reduced_batch=dataset_args["reduced_batch"],
                                initial_step=args["initial_step"],
                                t_train=args["t_train"],
                                if_test=True,
                                saved_folder = dataset_args["saved_folder"],
                                pde_name=args['pde_name']
                                )
    else:
        print("FNODatasetMult")
        train_data = FNODatasetMult(dataset_args["file_name"],
                                reduced_resolution=dataset_args["reduced_resolution"],
                                reduced_resolution_t=dataset_args["reduced_resolution_t"],
                                reduced_batch=dataset_args["reduced_batch"],
                                initial_step=args["initial_step"],
                                if_test=False,
                                saved_folder = dataset_args["saved_folder"]
                                )
        val_data = FNODatasetMult(dataset_args["file_name"],
                                reduced_resolution=dataset_args["reduced_resolution"],
                                reduced_resolution_t=dataset_args["reduced_resolution_t"],
                                reduced_batch=dataset_args["reduced_batch"],
                                initial_step=args["initial_step"],
                                if_test=True,
                                saved_folder = dataset_args["saved_folder"]
                                )
    return train_data, val_data


def get_dataloader(train_data, val_data, args):
    """Create dataloaders for train and validation."""
    dataloader_args = args["dataloader"]
    train_loader = DataLoader(train_data, shuffle=True, multiprocessing_context = 'spawn', generator=torch.Generator(device = 'cpu'), **dataloader_args)
    if args["if_training"]:
        val_loader = DataLoader(val_data, shuffle=False, multiprocessing_context = 'spawn', generator=torch.Generator(device = 'cpu'), **dataloader_args)
    else:
        # val_loader = DataLoader(val_data, shuffle=False, drop_last=True, **dataloader_args)
        val_loader = DataLoader(val_data, shuffle=False, **dataloader_args)
    return train_loader, val_loader


def get_model(spatial_dim, args):
    assert spatial_dim <= 3, "Spatial dimension of data can not exceed 3."

    model_args = args["model"]
    initial_step = args["initial_step"]
    if args['pde_name'] == "3D_Maxwell": #maxwell, input without grid
        model = FNO_maxwell(num_channels=model_args["num_channels"],
                      width=model_args["width"],
                      modes1=model_args["modes"],
                      modes2=model_args["modes"],
                      modes3=model_args["modes"],
                      initial_step=initial_step)
    elif spatial_dim == 1:
        model = FNO1d(num_channels=model_args["num_channels"],
                      width=model_args["width"],
                      modes=model_args["modes"],
                      initial_step=initial_step)
    elif spatial_dim == 2:
        model = FNO2d(num_channels=model_args["num_channels"],
                      width=model_args["width"],
                      modes1=model_args["modes"],
                      modes2=model_args["modes"],
                      initial_step=initial_step)
    elif spatial_dim == 3:
        model = FNO3d(num_channels=model_args["num_channels"],
                      width=model_args["width"],
                      modes1=model_args["modes"],
                      modes2=model_args["modes"],
                      modes3=model_args["modes"],
                      initial_step=initial_step)

    return model


def train_loop(model, train_loader, initial_step, t_train, optimizer, loss_fn, scheduler, device, args):
    model.train()
    t1 = default_timer()
    train_l2 = 0
    train_l_inf = 0
    for x, y, grid in train_loader:
        # batch_size = x.size(0)
        loss = 0
        if args['pde_name'] == "3D_Maxwell": #maxwell, we get x, y, global maximum
            grid = grid[0] #maximum
            x = x.detach().clone() / grid #scale
            y = y.detach().clone() / grid

        x = x.to(device) # x: input tensor (first few time steps) [b, x1, ..., xd, t_init, v]
        y = y # y: target tensor [b, x1, ..., xd, t_train, v]
        grid = grid.to(device) # grid: meshgrid [b, x1, ..., xd, dims]
        # initialize the prediction tensor
        pred = y[..., :initial_step, :] # (bs, x1, ..., xd, init_t, v)
        pred = pred.to(device)
        # reshape input
        input_shape = list(x.shape)[:-2] # (bs, x1, ..., xd)
        input_shape.append(-1) # (bs, x1, ..., xd, -1)

        if args["training_type"] in ['autoregressive']:
            # Autoregressive loop
            for t in range(initial_step, t_train):
                # Reshape input tensor into [b, x1, ..., xd, t_init*v]
                model_input = x.reshape(input_shape)
                # Extract target at current time step
                target = y[..., t:t+1, :]
                target = target.to(device) # [1, 401, 309, 1, 6]
                # Model run
                model_output = model(model_input, grid) # [1, 401, 309, 1, 6]
                # Loss calculation
                _batch = model_output.size(0)
                loss += loss_fn(model_output.reshape(_batch, -1), target.reshape(_batch, -1))
                # add physical loss
                if args["if_physical_loss"]:    
                    phy_loss = compute_physical_loss(model_output, x, grid, args)
                    loss += args["physical_loss_weight"]*phy_loss
                # Concatenate the prediction at current time step into the
                # prediction tensor
                pred = torch.cat((pred, model_output), -2)

                # Concatenate the prediction at the current time step to be used
                # as input for the next time step
                x = torch.cat((x[..., 1:, :], model_output), dim=-2)
            _batch = y.size(0)
            pred = pred.to('cpu')
            # y.to('cpu')
            torch.cuda.empty_cache()
            train_l2 += loss_fn(pred.reshape(_batch, -1), y.reshape(_batch, -1)).item()
            train_l_inf = max(train_l_inf, torch.max((torch.abs(pred.reshape(_batch, -1) - y.reshape(_batch, -1)))))

        elif args["training_type"] in ['single']:
            x = x[..., 0 , :]
            y = y[..., t_train-1:t_train, :]
            pred = model(x, grid)
            _batch = y.size(0)
            loss += loss_fn(pred.reshape(_batch, -1), y.reshape(_batch, -1))
            train_l2 += loss.item()
            train_l_inf = max(train_l_inf, torch.max((torch.abs(pred.reshape(_batch, -1) - y.reshape(_batch, -1)))))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
    t2 = default_timer()
    return train_l2, train_l_inf, t2 - t1


def val_loop(val_loader, model, loss_fn, device, training_type, t_train, initial_step):
    model.eval()
    val_l2_full = 0
    val_l_inf_full = 0
    with torch.no_grad():
        for x, y, grid in val_loader:
            loss = 0
            x = x.to(device)
            y = y.to(device)
            grid = grid.to(device)
            
            if training_type == 'autoregressive':
                pred = y[..., :initial_step, :]
                input_shape = list(x.shape)[:-2] # (bs, x1, ..., xd)
                input_shape.append(-1) # (bs, x1, ..., xd, -1)
                for t in range(initial_step, y.shape[-2]):
                    model_input = x.reshape(input_shape)
                    target = y[..., t:t+1, :]
                    model_output = model(model_input, grid)
                    _batch = model_output.size(0)
                    loss += loss_fn(model_output.reshape(_batch, -1), target.reshape(_batch, -1))
                    pred = torch.cat((pred, model_output), -2)
                    x = torch.cat((x[..., 1:, :], model_output), dim=-2)
    
                _batch = y.size(0)
                _pred = pred[..., initial_step:t_train, :]
                _y = y[..., initial_step:t_train, :]
                val_l2_full += loss_fn(_pred.reshape(_batch, -1), _y.reshape(_batch, -1)).item()
                val_l_inf_full = max(torch.max(torch.abs(_pred.reshape(_batch, -1) - _y.reshape(_batch, -1))).item(), val_l_inf_full)

            elif training_type == 'single':
                x = x[..., 0 , :]
                y = y[..., t_train-1:t_train, :]
                pred = model(x, grid)
                _batch = y.size(0)
                loss += loss_fn(pred.reshape(_batch, -1), y.reshape(_batch, -1))
    
                val_l2_full += loss.item()
                val_l_inf_full = max(torch.max(torch.abs(pred.reshape(_batch, -1) - y.reshape(_batch, -1))).item(), val_l_inf_full)
    return val_l2_full, val_l_inf_full


def test_loop(dataloader, model, device, initial_step, metric_names=['MSE', 'RMSE', 'L2RE', 'MaxError', 'SSIM']):
    model.eval()
    # initial result dict
    res_dict = {}
    pred_all = []
    y_all = []
    total_infer_time = 0
    num_infer_calls = 0
    for name in metric_names:
        res_dict[name] = []
    # test
    for x, y, grid in dataloader:
        if args['pde_name'] == "3D_Maxwell": #maxwell, we get x, y, global maximum
            grid = grid[0] #maximum
            x = x.detach().clone() / grid #scale
            y = y.detach().clone() / grid
        x = x.to(device)
        y = y.to(device)
        grid = grid.to(device)
        pred = y[..., :initial_step, :]
        input_shape = list(x.shape)[:-2] # (bs, x1, ..., xd)
        input_shape.append(-1) # (bs, x1, ..., xd, -1)
        for t in range(initial_step, y.shape[-2]):
            model_input = x.reshape(input_shape)
            with torch.no_grad():
                # --- 开始计时 ---
                if device.type == 'cuda':
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                else:
                    start_time = time.perf_counter()
                # -----------------
                model_output = model(model_input, grid)
                # --- 结束计时 ---
                if device.type == 'cuda':
                    end_event.record()
                    torch.cuda.synchronize()
                    total_infer_time += start_event.elapsed_time(end_event)
                else:
                    end_time = time.perf_counter()
                    total_infer_time += (end_time - start_time)
                num_infer_calls += 1

            pred = torch.cat((pred, model_output), dim=-2)
            x = torch.cat((x[..., 1:, :], model_output), dim=-2)
        for name in metric_names:
            metric_fn = getattr(metrics, name)
            res_dict[name].append(metric_fn(pred[..., initial_step:, :], y[..., initial_step:, :]))
    
    
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
            res = torch.cat(res_list, dim=0)
            res = torch.mean(res, dim=0)
        res_dict[name] = res

    # Collect predictions for HDF5 output
    pred_np = pred.detach().cpu().numpy().squeeze()   # (100,401,309)
    y_np = y.detach().cpu().numpy().squeeze()         # (100,401,309)
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
    relative_res_all = res_all / (y_all + 1e-8)

    print('pred_all shape:', pred_all.shape)

    # 定义要保存的文件及其对应的数据
    files_to_save = {
        'pred.hdf5': pred_all,
        'test.hdf5': y_all,
        'res.hdf5': res_all,
        'relative_res.hdf5': relative_res_all
    }

    return res_dict,files_to_save


def test_and_save_results(args, model, log_path, device, val_loader, output_path, initial_step):
    """
    Perform final test and save results to log file and CSV.
    Mirrors MPNN's test function behavior.
    """

    #  ======================   
    # 1.Save detailed test log
    os.makedirs(os.path.join(log_path,"test"), exist_ok=True)
    test_log_file = os.path.join(log_path,"test", f"{args['exp_id']}test.log")
    logging.basicConfig(filename=test_log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    print("Start final testing...")
    res,files_to_save = test_loop(val_loader, model, device, initial_step)
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

    # Save metrics to global CSV (as in MPNN)
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
        "unrolling": args.get("unrolling", None),
        "dataset": args["dataset"]["file_name"],
        "pde_name": args["pde_name"]
    }
    pandas_data.pd_results(
        err_dict=res,
        results_dict=results_dict,
        csv_file=f'{args["root_path"]}/results_log.csv'
    )

    # ===================================
    # 2. 保存测试结果到hdf5并绘图

    if args['pde_name'] == '2D_Mantle_Convection':
        var_names = ["ro", "pr", "tk", "nu", "vx", "vy"]
    elif args['pde_name'] == '2D_Compressible_NS':
        var_names = ["density", "pressure", "Vx", "Vy"]
    elif args['pde_name'] == 'ReacDiff':
        var_names = ["tensor"]

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
    """Save to local /tmp first, then copy to Lustre to avoid I/O errors on large files."""
    import time
    import uuid
    import shutil
    local_tmp = os.path.join("/tmp", f"ckpt_{uuid.uuid4().hex}.pt")
    try:
        torch.save(obj, local_tmp)
    except RuntimeError:
        if os.path.exists(local_tmp):
            os.remove(local_tmp)
        raise

    for attempt in range(max_retries):
        try:
            shutil.copy2(local_tmp, filepath)
            os.remove(local_tmp)
            return
        except OSError:
            if attempt == max_retries - 1:
                if os.path.exists(local_tmp):
                    os.remove(local_tmp)
                raise
            print(f"Checkpoint copy to Lustre failed (attempt {attempt+1}/{max_retries}), retrying...")
            time.sleep(2)


def main(args):
    """
    Main training/test pipeline.
    Now includes automatic testing after training completes.
    """
    # Setup directories
    # f'{args["root_path"]}/results_log.csv'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    exp_dir = os.path.join(args['root_path'], args['exp_id'])
    output_path = os.path.join(exp_dir, 'output')
    checkpoints_path = os.path.join(exp_dir, 'checkpoints')
    log_path = os.path.join(exp_dir, 'log')

    # log_path
    loss_history_file = os.path.join(log_path, 'loss_history.npy')
    val_loss_history_file = os.path.join(log_path, 'val_loss_history.npy')
    val_png = f'{args["exp_id"]}_Validation_Loss' # log_path/val_png.png
    train_png = f'{args["exp_id"]}_Training_Loss' # log_path/train_png.png
    test_log_file = os.path.join(log_path, "test.log")

    os.makedirs(log_path, exist_ok=True)
    os.makedirs(checkpoints_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(exp_dir, exist_ok=True)

    src_path = cmd_args.config_file               # 源文件，如 configs/train.yml
    dst_path = os.path.join(exp_dir, f'{args["exp_id"]}.yaml')  # 目标路径（可改名）
    shutil.copy(src_path, dst_path)

    # Load checkpoint if needed
    checkpoint = torch.load(args["model_path"]) if not args["if_training"] or args["continue_training"] else None

    # Model naming
    saved_model_name = args["model_id"] + f"_lr{args['optimizer']['lr']}_bs{args['dataloader']['batch_size']}"

    dataset_args = args["dataset"]
    if dataset_args["single_file"]:
        saved_model_name = dataset_args["file_name"][:-5] + '_FNO'
    else:
        saved_model_name = dataset_args["file_name"] + '_FNO'

    # Data loading
    train_data, val_data = get_dataset(args)
    train_loader, val_loader = get_dataloader(train_data, val_data, args)

    # Determine spatial dimension from sample
    _, sample, _ = next(iter(val_loader))
    spatial_dim = len(sample.shape) - 3
    initial_step = args["initial_step"]
    t_train = min(args["t_train"], sample.shape[-2])

    # Model initialization
    model = get_model(spatial_dim, args)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")
    
    # === Test mode ===
    if not args["if_training"]:
        print(f"Test mode, load checkpoint from {args['model_path']}")
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        test_and_save_results(args, model, log_path, device, val_loader, output_path, initial_step)
        return

    # === Training mode ===
    torch.backends.cudnn.enabled = False
    if args["continue_training"]:
        model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.train()

    # Optimizer
    optim_args = args["optimizer"]
    optim_name = optim_args.pop("name")
    ## if continue training, resume optimizer and scheduler from checkpoint    if args["continue_training"]:
    if args["continue_training"]:
        optimizer = getattr(torch.optim, optim_name)([{'params': model.parameters(), 'initial_lr': optim_args["lr"]}], **optim_args)
    else:
        optimizer = getattr(torch.optim, optim_name)(model.parameters(), **optim_args)

    # Resume training state
    start_epoch = 0
    min_val_loss = float('inf')
    if args["continue_training"]:
        start_epoch = checkpoint['epoch']
        min_val_loss = checkpoint['loss']

    # Scheduler
    sched_args = args["scheduler"]
    sched_name = sched_args.pop("name")
    scheduler = getattr(torch.optim.lr_scheduler, sched_name)(optimizer, last_epoch=start_epoch-1, **sched_args)

    # Loss function
    loss_fn = nn.MSELoss(reduction="mean")

    # Load previous loss history if continuing
    loss_history = []
    val_loss_history = []
    if args["continue_training"]:
        loss_history = np.load(loss_history_file).tolist()
        val_loss_history = np.load(val_loss_history_file).tolist()
    # === Training loop ===
    print("Start training...")
    total_time = 0
    for epoch in range(start_epoch, args["epochs"]):
        train_l2, train_l_inf, time = train_loop(model, train_loader, initial_step, t_train, optimizer, loss_fn, scheduler, device, args)
        scheduler.step()
        total_time += time
        loss_history.append(train_l2)
        print(f"[Epoch {epoch}] train_l2: {train_l2:.6f}, train_l_inf: {train_l_inf:.6f}, time: {time:.3f}s")

        # Save latest checkpoint
        saved_path = os.path.join(checkpoints_path, saved_model_name)
        model_state_dict = model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict()
        save_checkpoint({
            "epoch": epoch+1, "loss": min_val_loss,
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer.state_dict()
        }, saved_path + "-latest.pt")

        # Validation every `save_period` epochs
        if (epoch+1) % args["save_period"] == 0:
            print("====================validate====================")
            val_l2_full, val_l_inf = val_loop(val_loader, model, loss_fn, device, args["training_type"], t_train, initial_step)
            print(f"[Epoch {epoch}] val_l2: {val_l2_full:.6f}, val_l_inf: {val_l_inf:.6f}")
            print("================================================")
            if val_l2_full < min_val_loss:
                min_val_loss = val_l2_full
                save_checkpoint({
                    "epoch": epoch + 1, "loss": min_val_loss,
                    "model_state_dict": model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()
                }, saved_path + "-best.pt")
            val_loss_history.append(val_l2_full)

    # Save training loss history
    np.save(loss_history_file, np.array(loss_history))
    np.save(val_loss_history_file, np.array(val_loss_history))
    try:
        print(f"Total train time: {total_time}s. Average per epoch: {total_time / (args['epochs'] - start_epoch)}s")
    except ZeroDivisionError:
        print("设定训练epoch等于已经训练epoch")
    # 绘制损失曲线
    loss_plot(
        loss_history=loss_history,
        loss_name=train_png,
        save_path=log_path
    )
    
    if val_loss_history:
        loss_plot(
            loss_history=val_loss_history,
            loss_name=val_png,
            save_path=log_path
        )

    # 🔥 NEW: Automatic testing after training using best model
    best_path = saved_path + "-best.pt"
    print(f"Loading best model from {best_path} for final test...")
    model.load_state_dict(torch.load(best_path)["model_state_dict"])
    model.eval()
    test_and_save_results(args, model, log_path, device, val_loader, output_path, initial_step)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="Path to config file")
    cmd_args = parser.parse_args()
    with open(cmd_args.config_file, 'r') as f:
        args = yaml.safe_load(f)
    setup_seed(args["seed"])
    print(args)
    main(args)