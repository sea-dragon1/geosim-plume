# coding=utf-8
import shutil
import argparse
import numpy as np
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from timeit import default_timer
import yaml
import logging
import h5py

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import metrics
import pandas_data
from Transolver.transolver_model import Transolver2D


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


class TransolverDataset(Dataset):
    """Dataset for Transolver: reads HDF5 and returns (coords, input_fields, target_fields)."""

    def __init__(self, filename, saved_folder, initial_step=10, t_train=21,
                 test_ratio=0.1, if_test=False, variables=None):
        filepath = os.path.join(saved_folder, filename)
        self.initial_step = initial_step
        self.t_train = t_train
        self.if_test = if_test
        self.variables = variables or ["ro", "pr", "tk", "nu", "vx", "vy"]

        with h5py.File(filepath, 'r') as f:
            n_samples = f[self.variables[0]].shape[0]
            n_t_total = f[self.variables[0]].shape[1]
            self.H = f[self.variables[0]].shape[2]
            self.W = f[self.variables[0]].shape[3]

            # t_train clamping
            self.t_train = min(t_train, n_t_total)

            # Load all fields: [n_samples, t_train, H, W, n_vars]
            self.data = np.zeros([n_samples, self.t_train, self.H, self.W, len(self.variables)],
                                 dtype=np.float32)
            for vi, vname in enumerate(self.variables):
                self.data[..., vi] = f[vname][:, :self.t_train, :, :]

            # Coordinates
            x_coord = np.array(f["x-coordinate"], dtype=np.float32)
            y_coord = np.array(f["y-coordinate"], dtype=np.float32)
            X, Y = np.meshgrid(x_coord, y_coord, indexing='ij')  # (H, W)
            self.grid = np.stack([X, Y], axis=-1).reshape(-1, 2)  # [H*W, 2]

        # Train/val split
        n_train = int(n_samples * (1 - test_ratio))
        if not if_test:
            self.data = self.data[:n_train]
        else:
            self.data = self.data[n_train:]

        self.n_samples = self.data.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        sample = self.data[idx]  # [t_train, H, W, n_vars]
        # Reshape to [H*W, t_train, n_vars] then flatten t and vars for input
        # Input: first initial_step timesteps → [H*W, initial_step * n_vars]
        # Target: remaining timesteps → [H*W, t_train - initial_step, n_vars]
        # 注意: 必须先把 t 轴移到中间再 reshape（transpose(1,2,0,3)），
        # 直接 reshape 会把 [t, H, W, v] 按 C-order 重排，导致时间/空间轴混叠。
        # 这样节点 n=i*W+j 的数据才对应空间位置 (i,j) 的时间演化。
        sample_flat = sample.transpose(1, 2, 0, 3).reshape(self.H * self.W, self.t_train, len(self.variables))
        x = sample_flat[:, :self.initial_step, :].reshape(self.H * self.W, -1)  # [N, tw * n_vars]
        y = sample_flat[:, self.initial_step:, :]  # [N, t_train-tw, n_vars]
        grid = self.grid.copy()  # [N, 2]
        return torch.from_numpy(grid).float(), torch.from_numpy(x).float(), torch.from_numpy(y).float()


def get_dataloader(train_data, val_data, batch_size, num_workers=4, pin_memory=True, if_training=True):
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader


def train_epoch(model, train_loader, optimizer, loss_fn, device, n_vars, initial_step, t_train):
    model.train()
    total_loss = 0
    n_batches = 0
    for grid, x, y in train_loader:
        B = x.size(0)
        N = x.size(1)
        grid = grid.to(device)    # [B, N, 2]
        x = x.to(device)          # [B, N, tw * n_vars]
        y = y.to(device)          # [B, N, t_out, n_vars]
        t_out = y.shape[-2]

        current_input = x  # [B, N, tw * n_vars]
        loss = 0
        for t in range(t_out):
            model_output = model(grid, current_input)  # [B, N, n_vars]
            target = y[:, :, t, :]  # [B, N, n_vars]
            loss += loss_fn(model_output.reshape(B, -1), target.reshape(B, -1))
            # Slide window: remove oldest n_vars channels, append prediction
            current_input = torch.cat([current_input[:, :, n_vars:], model_output], dim=-1)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


@torch.no_grad()
def val_epoch(model, val_loader, loss_fn, device, n_vars, initial_step, t_train):
    model.eval()
    total_loss = 0
    n_batches = 0
    for grid, x, y in val_loader:
        B = x.size(0)
        grid = grid.to(device)
        x = x.to(device)
        y = y.to(device)
        t_out = y.shape[-2]

        current_input = x
        loss = 0
        for t in range(t_out):
            model_output = model(grid, current_input)
            target = y[:, :, t, :]
            loss += loss_fn(model_output.reshape(B, -1), target.reshape(B, -1))
            current_input = torch.cat([current_input[:, :, n_vars:], model_output], dim=-1)

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


@torch.no_grad()
def test_loop(dataloader, model, device, n_vars, initial_step, t_train, metric_names=None):
    if metric_names is None:
        metric_names = ['MSE', 'RMSE', 'L2RE', 'MaxError']
    model.eval()
    res_dict = {name: [] for name in metric_names}
    pred_all = []
    y_all = []
    total_infer_time = 0
    num_infer_calls = 0

    for grid, x, y in dataloader:
        B = x.size(0)
        grid = grid.to(device)
        x = x.to(device)
        y = y.to(device)  # [B, N, t_out, n_vars]
        t_out = y.shape[-2]
        N = x.shape[1]

        pred = torch.zeros(B, N, t_out, n_vars, device=device)
        current_input = x
        for t in range(t_out):
            if device.type == 'cuda':
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            model_output = model(grid, current_input)
            if device.type == 'cuda':
                end_event.record()
                torch.cuda.synchronize()
                total_infer_time += start_event.elapsed_time(end_event)
                num_infer_calls += 1
            pred[:, :, t, :] = model_output
            current_input = torch.cat([current_input[:, :, n_vars:], model_output], dim=-1)

        # pred: [B, N, t_out, n_vars] → [B, t_out, H, W, n_vars]
        H = W = int(N ** 0.5)
        pred_reshaped = pred.reshape(B, H, W, t_out, n_vars).permute(0, 3, 1, 2, 4)  # [B, t_out, H, W, n_vars]
        y_reshaped = y.reshape(B, H, W, t_out, n_vars).permute(0, 3, 1, 2, 4)

        for name in metric_names:
            metric_fn = getattr(metrics, name)
            res_dict[name].append(metric_fn(pred_reshaped, y_reshaped))

    if num_infer_calls > 0:
        avg_infer_time_s = total_infer_time / num_infer_calls / 1000
        print(f"Average inference time per call: {avg_infer_time_s:.6f} s")
        res_dict["avg_infer_time_s"] = avg_infer_time_s

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
    pred_all_list = []
    y_all_list = []
    for grid, x, y in dataloader:
        B = x.size(0)
        N = x.shape[1]
        H = W = int(N ** 0.5)
        grid = grid.to(device)
        x = x.to(device)
        y = y.to(device)
        t_out = y.shape[-2]

        pred = torch.zeros(B, N, t_out, n_vars, device=device)
        current_input = x
        for t in range(t_out):
            model_output = model(grid, current_input)
            pred[:, :, t, :] = model_output
            current_input = torch.cat([current_input[:, :, n_vars:], model_output], dim=-1)

        pred_np = pred.cpu().numpy().reshape(B, H, W, t_out, n_vars).transpose(0, 3, 1, 2, 4)
        y_np = y.cpu().numpy().reshape(B, H, W, t_out, n_vars).transpose(0, 3, 1, 2, 4)
        pred_all_list.append(pred_np)
        y_all_list.append(y_np)

    pred_all = np.concatenate(pred_all_list, axis=0)
    y_all = np.concatenate(y_all_list, axis=0)
    res_all = y_all - pred_all
    relative_res_all = res_all / (y_all + 1e-8)

    files_to_save = {
        'pred.hdf5': pred_all,
        'test.hdf5': y_all,
        'res.hdf5': res_all,
        'relative_res.hdf5': relative_res_all
    }

    return res_dict, files_to_save


def save_checkpoint(obj, filepath, max_retries=3):
    """Save to memory buffer, then write to temp file + atomic rename."""
    import time as _time
    import uuid
    import io

    buffer = io.BytesIO()
    torch.save(obj, buffer)
    data = buffer.getvalue()
    buffer.close()

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


def test_and_save_results(args, model, log_path, device, val_loader, output_path, n_vars, initial_step, t_train):
    os.makedirs(os.path.join(log_path, "test"), exist_ok=True)
    test_log_file = os.path.join(log_path, "test", f"{args['exp_id']}test.log")
    logging.basicConfig(filename=test_log_file, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    print("Start final testing...")
    res, files_to_save = test_loop(val_loader, model, device, n_vars, initial_step, t_train)
    print(res)

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

    pandas_data.pd_metrics(
        err_dict=res, exp_id=args['exp_id'], model_id=args['model_id'],
        data_name=os.path.splitext(args["dataset"]["file_name"])[0],
        csv_file=f'{args["root_path"]}/test_log.csv'
    )

    results_dict = {
        "exp_id": args["exp_id"], "model_id": args["model_id"],
        "hidden_features": args["model"].get("n_hidden", "N/A"),
        "lr": args["optimizer"]["lr"],
        "weight_decay": args["optimizer"].get("weight_decay", 0),
        "unrolling": args.get("unrolling", None),
        "dataset": args["dataset"]["file_name"],
        "pde_name": args.get("pde_name", "2D_Mantle_Convection")
    }
    pandas_data.pd_results(
        err_dict=res, results_dict=results_dict,
        csv_file=f'{args["root_path"]}/results_log.csv'
    )

    output_path_id = os.path.join(output_path, args["exp_id"])
    os.makedirs(output_path_id, exist_ok=True)

    var_names = ["ro", "pr", "tk", "nu", "vx", "vy"]
    for filename, data in files_to_save.items():
        file_path = os.path.join(output_path_id, filename)
        with h5py.File(file_path, 'w') as f:
            for i, name in enumerate(var_names):
                f.create_dataset(name, data=data[..., i], compression='gzip', compression_opts=4)
        pandas_data.h5fields_to_gif(
            h5_path=file_path, out_dir=f"{output_path_id}/data_info/{filename}",
            fields=var_names, fps=10, sample_idx=0
        )

    return res


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    exp_dir = os.path.join(args['root_path'], args['exp_id'])
    output_path = os.path.join(exp_dir, 'output')
    checkpoints_path = os.path.join(exp_dir, 'checkpoints')
    log_path = os.path.join(exp_dir, 'log')
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(checkpoints_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(exp_dir, exist_ok=True)

    src_path = cmd_args.config_file
    dst_path = os.path.join(exp_dir, f'{args["exp_id"]}.yaml')
    shutil.copy(src_path, dst_path)

    dataset_args = args["dataset"]
    initial_step = args["initial_step"]
    t_train = args["t_train"]
    n_vars = args.get("num_outputs", 6)

    train_data = TransolverDataset(
        dataset_args["file_name"], dataset_args["saved_folder"],
        initial_step=initial_step, t_train=t_train,
        test_ratio=dataset_args.get("test_ratio", 0.1), if_test=False
    )
    val_data = TransolverDataset(
        dataset_args["file_name"], dataset_args["saved_folder"],
        initial_step=initial_step, t_train=t_train,
        test_ratio=dataset_args.get("test_ratio", 0.1), if_test=True
    )

    H, W = train_data.H, train_data.W
    train_loader, val_loader = get_dataloader(
        train_data, val_data, args["dataloader"]["batch_size"],
        num_workers=args["dataloader"].get("num_workers", 4),
        pin_memory=args["dataloader"].get("pin_memory", True)
    )

    # Model: fun_dim = n_vars * initial_step
    model_args = args["model"]
    model = Transolver2D(
        space_dim=2, n_layers=model_args["n_layers"], n_hidden=model_args["n_hidden"],
        dropout=model_args.get("dropout", 0.0), n_head=model_args["n_head"],
        Time_Input=model_args.get("Time_Input", False), act=model_args.get("act", "gelu"),
        mlp_ratio=model_args.get("mlp_ratio", 1),
        fun_dim=n_vars * initial_step, out_dim=n_vars,
        slice_num=model_args.get("slice_num", 64),
        ref=model_args.get("ref", 8),
        unified_pos=model_args.get("unified_pos", False), H=H, W=W
    )
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")
    model.to(device)

    saved_model_name = f"{args['model_id']}_lr{args['optimizer']['lr']}_bs{args['dataloader']['batch_size']}"

    # Test mode
    if not args["if_training"]:
        print(f"Test mode, load checkpoint from {args['model_path']}")
        checkpoint = torch.load(args["model_path"])
        model.load_state_dict(checkpoint["model_state_dict"])
        test_and_save_results(args, model, log_path, device, val_loader, output_path,
                              n_vars, initial_step, t_train)
        return

    # Training mode
    torch.backends.cudnn.enabled = False
    model.train()

    optim_args = args["optimizer"].copy()
    optim_name = optim_args.pop("name")
    optimizer = getattr(torch.optim, optim_name)(model.parameters(), **optim_args)

    sched_args = args["scheduler"].copy()
    sched_name = sched_args.pop("name")
    scheduler = getattr(torch.optim.lr_scheduler, sched_name)(optimizer, **sched_args)

    loss_fn = nn.MSELoss(reduction="mean")

    loss_history = []
    val_loss_history = []
    min_val_loss = float('inf')

    start_epoch = 0
    if args.get("continue_training", False) and args.get("model_path"):
        checkpoint = torch.load(args["model_path"], map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        min_val_loss = checkpoint.get("loss", float('inf'))
        for _ in range(start_epoch):
            scheduler.step()
        print(f"Resumed from {args['model_path']}, start_epoch={start_epoch}, best_val_loss={min_val_loss:.6f}")

    print("Start training...")
    for epoch in range(start_epoch, args["epochs"]):
        t1 = default_timer()
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device,
                                 n_vars, initial_step, t_train)
        scheduler.step()
        t2 = default_timer()
        loss_history.append(train_loss)
        print(f"[Epoch {epoch}] train_loss: {train_loss:.6f}, time: {t2-t1:.2f}s")

        # Save latest
        saved_path = os.path.join(checkpoints_path, saved_model_name)
        save_checkpoint({
            "epoch": epoch + 1, "loss": min_val_loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        }, saved_path + "-latest.pt")

        if (epoch + 1) % args["save_period"] == 0:
            val_loss = val_epoch(model, val_loader, loss_fn, device, n_vars, initial_step, t_train)
            print(f"[Epoch {epoch}] val_loss: {val_loss:.6f}")
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                save_checkpoint({
                    "epoch": epoch + 1, "loss": min_val_loss,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()
                }, saved_path + "-best.pt")
            val_loss_history.append(val_loss)

    # Save loss history
    np.save(os.path.join(log_path, 'loss_history.npy'), np.array(loss_history))
    np.save(os.path.join(log_path, 'val_loss_history.npy'), np.array(val_loss_history))

    # Final test with best model
    best_path = saved_path + "-best.pt"
    print(f"Loading best model from {best_path} for final test...")
    model.load_state_dict(torch.load(best_path)["model_state_dict"])
    test_and_save_results(args, model, log_path, device, val_loader, output_path,
                          n_vars, initial_step, t_train)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="Path to config file")
    cmd_args = parser.parse_args()
    with open(cmd_args.config_file, 'r') as f:
        args = yaml.safe_load(f)
    setup_seed(args.get("seed", 0))
    print(args)
    main(args)
