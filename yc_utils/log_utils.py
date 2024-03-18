from rich.console import Console
from yc_utils import Table


def log_step(
        current_epoch,
        total_epochs,
        current_step,
        total_steps,
        loss,
        times,
        prefix,
        console=None):
    """
    Log metrics to the console after a forward pass

    Args:
        current_epoch: 当前epoch
        total_epochs: 总共epoch
        current_step: 当前steps
        total_steps: 总共steps
        loss: 损失值
        times
        prefix: 打印前缀
        console: 打印到的控制台
    """
    table = Table(show_header=True, header_style="bold")
    table.add_column("SPLIT")
    table.add_column("EPOCH")
    table.add_column("STEP")
    table.add_column("LOSS")
    for item in times:
        table.add_column(f"{item.upper()} TIME")
    time_values = [f"{t:.2f}" for t in times.values()]
    table.add_row(
        prefix.capitalize(),
        f"{current_epoch} / {total_epochs}",
        f"{current_step} / {total_steps}",
        f"{loss:.2f}",
        *tuple(time_values),
    )
    if console is not None:
        assert isinstance(console, Console)
        console.print(table)


def log_epoch(
        current_epoch,
        total_epochs,
        metrics,
        prefix,
        console=None):
    """
    Log metrics to the console after an epoch

    Args:
        current_epoch: 当前epoch
        total_epochs: 总共的epoch
        metrics: 指标
        prefix: log前缀
        console: 打印到的控制台
    """
    table = Table(show_header=True, header_style="bold", width=200)  # , width=128
    table.add_column("SPLIT")
    table.add_column("EPOCH")
    for k in metrics:
        table.add_column(k.replace(prefix, "").replace("/", "").upper())
    metric_values = [f"{m:.4f}" for m in metrics.values()]
    table.add_row(
        prefix.capitalize(),
        f"{current_epoch} / {total_epochs}",
        *tuple(metric_values),
    )
    if console is not None:
        assert isinstance(console, Console)
        console.print(table)
