import torch

def MSE(pred, target):
    """return mean square error

    pred: model output tensor of shape (bs, x1, ..., xd, t, v)
    target: ground truth tensor of shape (bs, x1, ..., xd, t, v)
    """
    assert pred.shape == target.shape
    temp_shape = [0, len(pred.shape)-1]
    temp_shape.extend([i for i in range(1, len(pred.shape)-1)])
    pred = pred.permute(temp_shape) # (bs, x1, ..., xd, t, v) -> (bs, v, x1, ..., xd, t)
    target = target.permute(temp_shape) # (bs, x1, ..., xd, t, v) -> (bs, v, x1, ..., xd, t)
    nb, nc = pred.shape[0], pred.shape[1]
    errors = pred.reshape([nb, nc, -1]) - target.reshape([nb, nc, -1]) # (bs, v, x1*x2*...*xd*t)
    res = torch.mean(errors**2, dim=2)
    return res # (bs, v)


def RMSE(pred, target):
    """return root mean square error

    pred: model output tensor of shape (bs, x1, ..., xd, t, v)
    target: ground truth tensor of shape (bs, x1, ..., xd, t, v)
    """
    return torch.sqrt(MSE(pred, target)) # (bs, v)


def L2RE(pred, target):
    """l2 relative error (nMSE in PDEBench)

    pred: model output tensor of shape (bs, x1, ..., xd, t, v)
    target: ground truth tensor of shape (bs, x1, ..., xd, t, v)
    """
    assert pred.shape == target.shape
    temp_shape = [0, len(pred.shape)-1]
    temp_shape.extend([i for i in range(1, len(pred.shape)-1)])
    pred = pred.permute(temp_shape) # (bs, x1, ..., xd, t, v) -> (bs, v, x1, ..., xd, t)
    target = target.permute(temp_shape) # (bs, x1, ..., xd, t, v) -> (bs, v, x1, ..., xd, t)
    nb, nc = pred.shape[0], pred.shape[1]
    errors = pred.reshape([nb, nc, -1]) - target.reshape([nb, nc, -1]) # (bs, v, x1*x2*...*xd*t)
    res = torch.sum(errors**2, dim=2) / torch.sum(target.reshape([nb, nc, -1])**2, dim=2)
    return torch.sqrt(res) # (bs, v)

def MaxError(pred, target):
    """return max error in a batch

    pred: model output tensor of shape (bs, x1, ..., xd, t, v)
    target: ground truth tensor of shape (bs, x1, ..., xd, t, v)
    """
    errors = torch.abs(pred - target)
    nc = errors.shape[-1]
    res, _ = torch.max(errors.reshape([-1, nc]), dim=0) # retain the last dim
    return res # (v)


def SSIM(pred, target, win_size=7, K1=0.01, K2=0.03):
    """Compute per-sample per-variable mean SSIM over spatial and temporal dims.

    pred, target: (bs, H, W, T, v)
    Returns: (bs, v) — mean SSIM over (H, W, T) per sample and variable.

    Uses scipy.ndimage.uniform_filter for CPU implementation.
    """
    import numpy as np
    import scipy.ndimage
    bs, H, W, T, v = pred.shape
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    res = torch.zeros(bs, v)
    for s in range(bs):
        for c in range(v):
            vals = []
            for t_idx in range(T):
                p = pred_np[s, :, :, t_idx, c]
                g = target_np[s, :, :, t_idx, c]
                data_range = max(float(g.max() - g.min()), 1e-8)
                C1 = (K1 * data_range) ** 2
                C2 = (K2 * data_range) ** 2
                mu1 = scipy.ndimage.uniform_filter(p, win_size)
                mu2 = scipy.ndimage.uniform_filter(g, win_size)
                sigma1_sq = scipy.ndimage.uniform_filter(p * p, win_size) - mu1 * mu1
                sigma2_sq = scipy.ndimage.uniform_filter(g * g, win_size) - mu2 * mu2
                sigma12 = scipy.ndimage.uniform_filter(p * g, win_size) - mu1 * mu2
                num = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
                den = (mu1 * mu1 + mu2 * mu2 + C1) * (sigma1_sq + sigma2_sq + C2)
                ssim_map = num / (den + 1e-8)
                vals.append(ssim_map.mean())
            res[s, c] = float(np.mean(vals))
    return res  # (bs, v)

## 还没写好下面的error
# def RelativeError(pred, target):
#     """return relative error in a batch

#     pred: model output tensor of shape (bs, x1, ..., xd, t, v)
#     target: ground truth tensor of shape (bs, x1, ..., xd, t, v)
#     """
#     assert pred.shape == target.shape
#     temp_shape = [0, len(pred.shape)-1]
#     temp_shape.extend([i for i in range(1, len(pred.shape)-1)])
#     pred = pred.permute(temp_shape) # (bs, x1, ..., xd, t, v) -> (bs, v, x1, ..., xd, t)
#     target = target.permute(temp_shape) # (bs, x1, ..., xd, t, v) -> (bs, v, x1, ..., xd, t)
#     nb = pred.shape[0]
#     errors = torch.abs(pred.reshape([nb, -1]) - target.reshape([nb, -1])) # (bs, v*x1*x2*...*xd*t)
#     res = torch.mean(errors / torch.abs(target.reshape([nb, -1]) + 1e-8), dim=0) # (v*x1*x2*...*xd*t)
#     return res # (v*x1*x2*...*xd*t)