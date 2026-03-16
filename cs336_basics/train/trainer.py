from cs336_basics.model.softmax import Softmax
from cs336_basics.model.transformer import TransformerLM
from cs336_basics.tokenizer.tokenizer import Tokenizer
from cs336_basics.train.cross_entropy import cross_entropy
from cs336_basics.train.data_loader import get_batch
from dataclasses import dataclass, asdict
from cs336_basics.train.adam import AdamW
import torch
import numpy as np

from cs336_basics.train.lr_schedule import cosine_lr_schedule


@dataclass
class TrainerConfig:
    lr: float
    weight_decay: float
    beta1: float
    beta2: float
    max_samples: int
    train_batch_size: int
    val_batch_size: int
    context_length: int
    device: str


@dataclass
class ModelConfig:
    vocab_size: int
    context_length: int
    d_model: int
    num_layers: int
    num_heads: int
    d_ff: int
    rope_theta: float
    device: str


class LMTrainer:
    def __init__(
        self, trainer_config: TrainerConfig, model_config: ModelConfig, train_dataset_path: str, val_dataset_path
    ):
        torch.set_float32_matmul_precision("high")
        self.trainer_config = trainer_config
        self.model_config = model_config
        self.model = TransformerLM(**asdict(self.model_config))
        self.model.compile()
        self.optimizer = AdamW(
            self.model.parameters(),
            self.trainer_config.lr,
            self.trainer_config.weight_decay,
            [self.trainer_config.beta1, self.trainer_config.beta2],
            1e-8,
        )
        self.softmax = Softmax()
        self.train_dataset_path = train_dataset_path
        self.val_dataset_path = val_dataset_path
        self.train_dataset = np.memmap(self.train_dataset_path, dtype="int16", mode="r")
        self.val_dataset = np.memmap(self.val_dataset_path, dtype="int16", mode="r")
        print(f"- Train dataset: {len(self.train_dataset)} tokens")
        self.total_steps = self.trainer_config.max_samples / self.trainer_config.train_batch_size
        self.cur_step = 1
        print(f"- total_steps: {self.total_steps}")
        for p in self.model.named_parameters():
            print(f"{p[0]}:{p[1].shape}")

    def step(self):
        input_ids, labels = get_batch(
            self.train_dataset,
            self.trainer_config.train_batch_size,
            self.trainer_config.context_length,
            self.trainer_config.device,
        )
        lr = cosine_lr_schedule(self.cur_step, self.trainer_config.lr, 0.0, 100, self.total_steps)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        logits = self.model(input_ids)
        self.optimizer.zero_grad()
        loss = cross_entropy(logits, labels)
        loss.backward()
        self.optimizer.step()
        self.cur_step += 1
        return self.test()

    def test(self):
        with torch.no_grad():
            input_ids, labels = get_batch(
                self.val_dataset,
                self.trainer_config.val_batch_size,
                self.trainer_config.context_length,
                self.trainer_config.device,
            )
            logits = self.model(input_ids)
            loss = cross_entropy(logits, labels)
            return loss
