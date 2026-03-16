import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        """
        初始化自定义的词嵌入模块。

        参数:
            num_embeddings (int): 词表大小 (Vocabulary size)
            embedding_dim (int): 嵌入向量维度 (d_model)
            device (torch.device | None): 参数存储的设备
            dtype (torch.dtype | None): 参数的数据类型
        """
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device))
        torch.nn.init.trunc_normal_(self.weight, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        根据给定的 token IDs 查找对应的嵌入向量。

        参数:
            token_ids (torch.Tensor): 包含词表索引的张量，形状任意 (通常为 batch_size, sequence_length)

        返回:
            torch.Tensor: 查表得到的嵌入张量，形状为 (*token_ids.shape, embedding_dim)
        """
        return self.weight[token_ids]
