import torch
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    SpinnerColumn,
    MofNCompleteColumn,
    TaskProgressColumn,
)
from torch.profiler import profile, record_function, ProfilerActivity
from rich.live import Live
from rich.table import Table
from rich.console import Console

from cs336_basics.train.checkpoint import save_checkpoint
from cs336_basics.train.trainer import TrainerConfig, ModelConfig, LMTrainer


def run_training():
    console = Console()

    model_cfg = ModelConfig(
        vocab_size=10000,
        context_length=256,
        d_model=512,
        num_layers=4,
        num_heads=16,
        d_ff=1344,
        rope_theta=10000.0,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    trainer_cfg = TrainerConfig(
        lr=1e-4,
        weight_decay=0.01,
        beta1=0.99,
        beta2=0.999,
        max_samples=320000000,
        train_batch_size=32,
        val_batch_size=16,
        context_length=256,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    trainer = LMTrainer(
        trainer_config=trainer_cfg,
        model_config=model_cfg,
        train_dataset_path="data/tinystories_train.npy",
        val_dataset_path="data/tinystories_val.npy",
    )

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn("[progress.percentage, progress.speed]{task.percentage:>3.0f}%({task.speed})"),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console,
    )

    train_task = progress.add_task("[cyan]Training...", total=trainer.get_total_steps())

    with Live(progress, console=console, refresh_per_second=4):
        for step_idx in range(trainer.get_total_steps()):
            trainer.step(log=step_idx % 10 == 0, test=step_idx % 500 == 0)

            progress.update(train_task, advance=1)

            if step_idx % 10 == 0:
                progress.update(train_task, description=f"[cyan]Step {step_idx}[/]")

            if step_idx % 2000 == 0:
                progress.console.print(f"[bold yellow][{step_idx}][/] Saving checkpoint...")
                save_checkpoint(trainer.model, trainer.optimizer, trainer.cur_step, "./data/checkpoint.pt")

    trainer.finish()


if __name__ == "__main__":
    run_training()
