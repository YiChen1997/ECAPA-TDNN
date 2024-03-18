import os
import glob
import torch


def save_checkpoint(
        epoch, checkpoints_path, model, optimizer, lr_scheduler=None, wandb_run=None
):
    """
    Save the current state of the model, optimizer and learning rate scheduler,
    both locally and on wandb (if available and enabled)

    Args:
        epoch: 当前epoch
        checkpoints_path: 保存路径
        model: 模型
        optimizer: 模型优化器
        lr_scheduler: 学习率调整策略
        wandb_run: wandb实例
    """
    # Create state dictionary
    state_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": (
            lr_scheduler.state_dict() if lr_scheduler is not None else dict()
        ),
        "epoch": epoch,
    }

    # Save state dictionary
    checkpoint_file = os.path.join(checkpoints_path, f"epoch_{epoch}.pth")
    torch.save(state_dict, checkpoint_file)
    print(f"{checkpoint_file} was saved!")
    if wandb_run is not None:
        pass
        # TODO Linux应该能正常save，windows如何save？
        # wandb_run.save(checkpoint_file)


def load_last_checkpoint(model, optimizer, lr_scheduler, checkpoint_root_path):
    # epoch, checkpoints_path, model, optimizer, lr_scheduler = None, wandb_run = None
    # 检查现有模型并加载最后一个模型
    # 参考自：https: // zhuanlan.zhihu.com / p / 82038049
    model_files = glob.glob('%s/epoch_*.pth' % checkpoint_root_path)
    model_files.sort(key=os.path.getmtime)
    if len(model_files) >= 1:
        print("load from previous model %s!" % model_files[-1])
        next_epoch = int(os.path.splitext(os.path.basename(model_files[-1]))[0][6:]) + 1
    else:
        raise Exception("Can not find initial model")
    loaded_state = torch.load(model_files[-1])

    model.load_state_dict(loaded_state['model'])
    optimizer.load_state_dict(loaded_state['optimizer'])
    lr_scheduler.load_state_dict(loaded_state['lr_scheduler'])
    epoch = loaded_state['epoch']
    # loss = loaded_state['loss']

    return model, optimizer, lr_scheduler, epoch


def load_single_checkpoint(model, checkpoint_path):
    """
    evaluate用的加载函数
    """
    print(f"loading model in {checkpoint_path}")
    loaded_state = torch.load(checkpoint_path)
    print(f"before: {type(model)}")
    model.load_state_dict(loaded_state['model'])
    print(f"after: {type(model)}")
    # optimizer.load_state_dict(loaded_state['optimizer'])
    # lr_scheduler.load_state_dict(loaded_state['lr_scheduler'])
    epoch = loaded_state['epoch']
    return model, epoch
