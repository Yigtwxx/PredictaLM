
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MiniGPTConfig:
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 1024,
        max_len: int = 256,
        dropout: float = 0.1,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.max_len = max_len
        self.dropout = dropout


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer("mask", None, persistent=False)

    def _get_causal_mask(self, seq_len, device):
        if self.mask is None or self.mask.size(0) < seq_len:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0)
            self.mask = mask
        return self.mask[:, :seq_len, :seq_len]

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv(x)  # (B, T, 3*C)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = self._get_causal_mask(T, x.device)
        att = att.masked_fill(mask[:, None, :, :] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        y = att @ v  # (B, n_heads, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.out_proj(y)
        return y


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, config: MiniGPTConfig):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    config.d_model,
                    config.n_heads,
                    config.d_ff,
                    config.dropout,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, input_ids, targets=None):
        B, T = input_ids.shape
        device = input_ids.device

        pos = torch.arange(0, T, device=device).unsqueeze(0)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        x = self.drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=10, temperature=1.0, top_k=5):
        self.eval()
        device = input_ids.device

        for _ in range(max_new_tokens):
            if input_ids.size(1) > self.config.max_len:
                input_ids = input_ids[:, -self.config.max_len :]

            logits, _ = self(input_ids)
            logits = logits[:, -1, :] / max(temperature, 1e-6)

            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, top_k)
                min_v = v[:, -1].unsqueeze(-1)
                logits = torch.where(
                    logits < min_v, torch.full_like(logits, float("-inf")), logits
                )

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids