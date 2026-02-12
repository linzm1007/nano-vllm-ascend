import torch
from torch import nn


class Sampler(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        real_bs = temperatures.shape[0]
        if logits.shape[0] != real_bs:
            logits = logits[:real_bs, :]
        logits = logits.float() / temperatures.unsqueeze(-1)

        probs = torch.softmax(logits, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
        return sampled
