from matplotlib import pyplot as plt
import os


def loss_plot(loss_history: list, loss_name='Training_Loss', save_path: str = "experiments"):
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
    plt.ylabel('Loss (Log Scale)')
    plt.title(f'{loss_name} Curve')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_path, f"{loss_name}.png"))
    plt.close()
    print(f"Loss curve saved to {os.path.join(save_path, f'{loss_name}.png')}")
