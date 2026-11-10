"""
GNOT training script for plume2 dataset.
Uses original GNOT single-step supervised training (no pushforward).
Saves results in same HDF5 format as MFGNO/MPNN/FNO for consistent comparison.
"""
import argparse
import os
import sys
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import yaml
import shutil
import h5py
import dgl

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.mmgpt import GNOT
from utils import MultipleTensors, get_seed

# Add top-level for metrics and pandas_data
# 发布版：改为相对本文件的 vendor 根（原实现硬编码了作者机器的仓库根）
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
import metrics
import pandas_data


def collate(items):
    """Batch DGL graphs, identical to original GNOT collate_op."""
    transposed = zip(*items)
    batched = []
    for sample in transposed:
        if isinstance(sample[0], dgl.DGLGraph):
            batched.append(dgl.batch(list(sample)))
        elif isinstance(sample[0], torch.Tensor):
            batched.append(torch.stack(sample))
        elif isinstance(sample[0], MultipleTensors):
            from torch.nn.utils.rnn import pad_sequence
            sample_ = MultipleTensors([
                pad_sequence([sample[i][j] for i in range(len(sample))]).permute(1, 0, 2)
                for j in range(len(sample[0]))
            ])
            batched.append(sample_)
        else:
            raise NotImplementedError
    return batched


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    losses = []
    for g, u_p, inputs in dataloader:
        u_p = u_p.to(device)
        # Move branch inputs (MultipleTensors) to device
        inputs = inputs.to(device)
        # Keep DGL graph on CPU (DGL is CPU-only), ndata moved inside model
        target = g.ndata['y'].to(device)  # [total_nodes, output_dim]

        out = model(g, u_p, inputs)
        pred = out  # [total_nodes, output_dim]

        loss = nn.functional.mse_loss(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return np.mean(losses)


def validate(model, dataloader, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for g, u_p, inputs in dataloader:
            u_p = u_p.to(device)
            inputs = inputs.to(device)
            target = g.ndata['y'].to(device)

            out = model(g, u_p, inputs)
            pred = out

            loss = nn.functional.mse_loss(pred, target)
            losses.append(loss.item())

    return np.mean(losses)


def test_and_save(model, dataloader, device, args, output_dir, log_dir):
    """Evaluate model and save predictions in consistent format."""
    model.eval()

    res_dict = {'MSE': [], 'RMSE': [], 'L2RE': [], 'MaxError': []}
    all_preds = []
    all_targets = []
    nx, ny = args['resolution'][0], args['resolution'][1]
    tw = args['time_window']
    n_vars = 6

    total_infer_time = 0.0
    n_calls = 0

    with torch.no_grad():
        for g, u_p, inputs in dataloader:
            u_p = u_p.to(device)
            inputs = inputs.to(device)

            # Time inference
            if device.type == 'cuda':
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
            else:
                t0 = time.perf_counter()

            out = model(g, u_p, inputs)

            if device.type == 'cuda':
                end.record()
                torch.cuda.synchronize()
                total_infer_time += start.elapsed_time(end)
            else:
                total_infer_time += (time.perf_counter() - t0)
            n_calls += 1

            pred = out.cpu()  # [total_nodes, tw*6]
            target = g.ndata['y'].cpu()  # [total_nodes, tw*6]

            # Reshape to [B, tw, nx, ny, 6] for metrics
            batch_size = u_p.shape[0]
            pred_reshape = pred.reshape(batch_size, nx, ny, tw, n_vars)
            pred_reshape = pred_reshape.permute(0, 3, 1, 2, 4)  # [B, tw, nx, ny, 6]
            target_reshape = target.reshape(batch_size, nx, ny, tw, n_vars)
            target_reshape = target_reshape.permute(0, 3, 1, 2, 4)

            all_preds.append(pred_reshape)
            all_targets.append(target_reshape)

    # Concatenate all batches
    pred_all = torch.cat(all_preds, dim=0)  # [num_samples, tw, nx, ny, 6]
    target_all = torch.cat(all_targets, dim=0)

    # ---- metric shape: [bs, x1, ..., xd, t, v] for metrics functions ----
    # The metrics functions expect: [bs, x1, ..., xd, t, v]
    # Our shape is [num_samples, tw, nx, ny, 6]
    # So: [bs, tw, nx, ny, 6] → need [bs, nx, ny, tw, 6]
    pred_for_metric = pred_all.permute(0, 2, 3, 1, 4)  # [bs, nx, ny, tw, 6]
    target_for_metric = target_all.permute(0, 2, 3, 1, 4)

    for name in ['MSE', 'RMSE', 'L2RE', 'MaxError']:
        metric_fn = getattr(metrics, name)
        res_dict[name] = metric_fn(pred_for_metric, target_for_metric)

    # Post-process metrics
    for name in ['MSE', 'RMSE', 'L2RE', 'MaxError']:
        res_list = res_dict[name]
        if name == "MaxError":
            res, _ = torch.max(torch.stack([res_list], dim=0), dim=0)
        else:
            res = torch.cat([res_list], dim=0)
            res = torch.mean(res, dim=0)
        res_dict[name] = res

    if n_calls > 0:
        avg_time = total_infer_time / n_calls
        if device.type == 'cuda':
            avg_time = avg_time / 1000
        print(f"Average inference time: {avg_time:.6f} s")
        res_dict["avg_infer_time_s"] = avg_time

    # ---- Save HDF5 files in same format as MFGNO/MPNN/FNO ----
    # pred_all shape: [num_samples, tw, nx, ny, 6] → save as [num_samples, tw, nx, ny, 6]
    res_all = target_all - pred_all
    relative_res_all = res_all / (target_all + 1e-20)

    os.makedirs(output_dir, exist_ok=True)
    var_names = ["ro", "pr", "tk", "nu", "vx", "vy"]

    files_to_save = {
        'pred.hdf5': pred_all.numpy(),
        'test.hdf5': target_all.numpy(),
        'res.hdf5': res_all.numpy(),
        'relative_res.hdf5': relative_res_all.numpy(),
    }

    for filename, data in files_to_save.items():
        filepath = os.path.join(output_dir, filename)
        with h5py.File(filepath, 'w') as f:
            for i, name in enumerate(var_names):
                f.create_dataset(name,
                                 data=data[..., i],
                                 compression='gzip',
                                 compression_opts=4)

        pandas_data.h5fields_to_gif(
            h5_path=filepath,
            out_dir=f"{output_dir}/data_info/{filename}",
            fields=var_names,
            fps=10,
            sample_idx=0
        )

    return res_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="Path to YAML config")
    cmd_args = parser.parse_args()
    with open(cmd_args.config_file, 'r') as f:
        args = yaml.safe_load(f)

    # Paths
    root_path = args['root_path']
    exp_dir = os.path.join(root_path, args['exp_id'])
    checkpoints_dir = os.path.join(exp_dir, 'checkpoints')
    log_dir = os.path.join(exp_dir, 'logs')
    output_dir = os.path.join(exp_dir, 'output', args['exp_id'])

    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Backup config
    shutil.copy(cmd_args.config_file, os.path.join(exp_dir, f"{args['exp_id']}.yaml"))

    # Setup
    get_seed(args['seed'], printout=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset
    from dataset import Plume1Dataset
    ds_args = args['dataset']
    tw = args['time_window']
    nt = args['resolution_t']

    train_ds = Plume1Dataset(
        file_name=ds_args['file_name'],
        saved_folder=ds_args['saved_folder'],
        time_window=tw,
        total_time_steps=nt,
        if_test=False,
        test_ratio=ds_args.get('test_ratio', 0.1)
    )
    test_ds = Plume1Dataset(
        file_name=ds_args['file_name'],
        saved_folder=ds_args['saved_folder'],
        time_window=tw,
        total_time_steps=nt,
        if_test=True,
        test_ratio=ds_args.get('test_ratio', 0.1)
    )

    print(f"Train samples: {len(train_ds)}, Test samples: {len(test_ds)}")
    print(f"Nodes per graph: {train_ds.n_nodes}")

    # DataLoaders
    dl_args = args['dataloader']
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=dl_args['batch_size'], shuffle=True,
        num_workers=dl_args.get('num_workers', 0),
        pin_memory=dl_args.get('pin_memory', False),
        collate_fn=collate
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=dl_args['batch_size'], shuffle=False,
        num_workers=dl_args.get('num_workers', 0),
        pin_memory=dl_args.get('pin_memory', False),
        collate_fn=collate
    )

    # Model
    m_args = args['model']
    # trunk_size = spatial_dim (2) + theta_dim (1 for step index) = 3
    trunk_size = 2 + 1  # x, y coords + step index
    # branch_sizes: [tw * n_vars] = [60]
    branch_sizes = [tw * 6]  # joint branch: 10 timesteps × 6 fields
    output_size = tw * 6  # predict 10 timesteps × 6 fields

    model = GNOT(
        trunk_size=trunk_size,
        branch_sizes=branch_sizes,
        space_dim=2,
        output_size=output_size,
        n_layers=m_args.get('n_layers', 3),
        n_hidden=m_args.get('n_hidden', 128),
        n_head=m_args.get('n_head', 1),
        n_experts=m_args.get('n_experts', 2),
        n_inner=m_args.get('n_inner', 4),
        mlp_layers=m_args.get('mlp_layers', 2),
        attn_type=m_args.get('attn_type', 'linear'),
        act=m_args.get('act', 'gelu'),
        ffn_dropout=m_args.get('ffn_dropout', 0.0),
        attn_dropout=m_args.get('attn_dropout', 0.0),
        horiz_fourier_dim=m_args.get('horiz_fourier_dim', 0),
    )
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: GNOT, Parameters: {total_params}")

    # Test-only mode
    if not args.get('if_training', True):
        checkpoint = torch.load(args['model_path'], map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {args['model_path']}")

        res = test_and_save(model, test_loader, device, args, output_dir, log_dir)

        # Log metrics
        test_log_file = os.path.join(log_dir, f"{args['exp_id']}_test.log")
        logging.basicConfig(filename=test_log_file, level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        for k, v in res.items():
            if k == 'avg_infer_time_s':
                logging.info(f"{k}\t{v:.6f}s")
            else:
                for i in range(len(v)):
                    logging.info(f"{k}\t{v[i].item():.6f}")

        pandas_data.pd_metrics(
            err_dict={k: v for k, v in res.items() if k != 'avg_infer_time_s'},
            exp_id=args['exp_id'],
            model_id=args['model_id'],
            data_name=os.path.splitext(args['dataset']['file_name'])[0],
            csv_file=f'{args["root_path"]}/test_log.csv'
        )
        pandas_data.pd_results(
            err_dict={k: v for k, v in res.items() if k != 'avg_infer_time_s'},
            results_dict={
                "exp_id": args["exp_id"],
                "model_id": args["model_id"],
                "hidden_features": m_args.get("n_hidden", "N/A"),
                "lr": args["optimizer"]["lr"],
                "weight_decay": args["optimizer"].get("weight_decay", 0),
                "dataset": args["dataset"]["file_name"],
                "pde_name": args.get("pde_name", ""),
            },
            csv_file=f'{args["root_path"]}/results_log.csv'
        )
        return

    # Optimizer
    optim_args = args['optimizer']
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(optim_args['lr']),
        weight_decay=float(optim_args.get('weight_decay', 0))
    )

    # Scheduler
    sched_args = args['scheduler']
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=sched_args.get('step_size', 100),
        gamma=sched_args.get('gamma', 0.5)
    )

    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float('inf')
    if args.get('continue_training', False) and args.get('model_path'):
        checkpoint = torch.load(args['model_path'], map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint.get('loss', float('inf'))
        # Step scheduler to match resumed epoch
        for _ in range(start_epoch):
            scheduler.step()
        print(f"Resumed from {args['model_path']}, start_epoch={start_epoch}, best_val_loss={best_val_loss:.6f}")

    # Training loop
    epochs = args['epochs']
    save_period = args.get('save_period', 10)
    loss_history = []
    val_loss_history = []

    print(f"\nStart training from epoch {start_epoch} to {epochs}...")
    total_time = 0
    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, device)
        epoch_time = time.time() - t0
        total_time += epoch_time
        loss_history.append(train_loss)
        scheduler.step()

        print(f"[Epoch {epoch}] train_loss: {train_loss:.6f}, time: {epoch_time:.1f}s, "
              f"lr: {optimizer.param_groups[0]['lr']:.2e}")

        # Validation and checkpointing
        if (epoch + 1) % save_period == 0:
            val_loss = validate(model, test_loader, device)
            val_loss_history.append(val_loss)
            print(f"  val_loss: {val_loss:.6f}")

            # Save checkpoint
            saved_path = os.path.join(checkpoints_dir, args['exp_id'])
            torch.save({
                "epoch": epoch + 1,
                "loss": val_loss,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, saved_path + "-latest.pt")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    "epoch": epoch + 1,
                    "loss": best_val_loss,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }, saved_path + "-best.pt")

    # Save loss history
    np.save(os.path.join(log_dir, f"{args['exp_id']}_loss_history.npy"),
            np.array(loss_history))
    if val_loss_history:
        np.save(os.path.join(log_dir, f"{args['exp_id']}_val_loss_history.npy"),
                np.array(val_loss_history))

    print(f"\nTraining completed. Total time: {total_time:.1f}s")

    # Final evaluation
    print("\nRunning final evaluation...")
    res = test_and_save(model, test_loader, device, args, output_dir, log_dir)

    # Log metrics
    test_log_file = os.path.join(log_dir, f"{args['exp_id']}_test.log")
    logging.basicConfig(filename=test_log_file, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    for k, v in res.items():
        if k == 'avg_infer_time_s':
            print(f"{k}: {v:.6f}s")
            logging.info(f"{k}\t{v:.6f}s")
        else:
            dim = len(v)
            for i in range(dim):
                print(f"{k}[{i}]: {v[i].item():.6f}")
                logging.info(f"{k}\t{v[i].item():.6f}")

    pandas_data.pd_metrics(
        err_dict={k: v for k, v in res.items() if k != 'avg_infer_time_s'},
        exp_id=args['exp_id'],
        model_id=args['model_id'],
        data_name=os.path.splitext(args['dataset']['file_name'])[0],
        csv_file=f'{args["root_path"]}/test_log.csv'
    )
    pandas_data.pd_results(
        err_dict={k: v for k, v in res.items() if k != 'avg_infer_time_s'},
        results_dict={
            "exp_id": args["exp_id"],
            "model_id": args["model_id"],
            "hidden_features": m_args.get("n_hidden", "N/A"),
            "lr": args["optimizer"]["lr"],
            "weight_decay": args["optimizer"].get("weight_decay", 0),
            "dataset": args["dataset"]["file_name"],
            "pde_name": args.get("pde_name", ""),
        },
        csv_file=f'{args["root_path"]}/results_log.csv'
    )

    print("Done.")


if __name__ == "__main__":
    main()
