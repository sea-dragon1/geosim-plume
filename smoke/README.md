# 冒烟测试

本目录验证外部基线能从 `code/vendor` 独立导入，并完成一次 2-epoch 训练。
它不产生论文结果，只检查依赖与训练链路是否完整。

随包始终检查 **FNO / Transolver**。**GNOT 是可选项**：上游 GNOT 模型代码
没有随包（许可证原因），只有把上游文件放入
`code/vendor/GNOT-master/` 后，脚本才会检查 GNOT；否则明确打印 `SKIP GNOT`。

## 运行

从 `geosim-plume/` 目录提交：

```bash
yhbatch smoke/smoke_test.sh
```

脚本依次：

1. 分别导入 FNO / Transolver（每个模型单独进程，避免同名 `utils.py` 互相遮蔽）；
2. 如果 GNOT 上游代码已放入，再导入并训练 GNOT；
3. 用 `smoke/configs/` 下的 2-epoch 配置各训练一次；
4. 检查 `smoke/runs/<exp_id>/` 是否生成。

产物写入 `smoke/runs/`，运行后生成，不随包提供。

`smoke/configs/` 中的 `dataset.saved_folder` 是绝对路径。换机器运行时，
需要把它改成实际数据目录（参见
[`code/experiments/E01_main/README.md`](../code/experiments/E01_main/README.md)
的 `GEOSIM_DATA` 说明）。

GNOT 上游文件的放置方式见
[`code/vendor/GNOT-master/NOTICE.md`](../code/vendor/GNOT-master/NOTICE.md)。

## 记录

- `smoke/logs/smoke-3853203.out` / `.err`：2026-11-09 的成功运行。FNO、
  Transolver 通过，GNOT 按设计跳过；`.err` 只有 tqdm 进度条和
  `torch.load` 的 FutureWarning，无报错。
- `smoke/logs/smoke-3795827.out` / `.err`：加入 GNOT 可选依赖之前的一次成功
  运行日志；当时三个基线都在包内。
