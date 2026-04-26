import torch
import torch.nn as nn


class SimpleGPT(nn.Module):
    def __init__(self, vocab_size=65, n_embd=128, n_layer=4, n_head=4, block_size=256):
        super().__init__()
        self.block_size = block_size

        # 1. Direct attributes (Fixes the AttributeError)
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)

        # 2. Use a list of blocks for transparency
        self.h = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=n_embd,
                nhead=n_head,
                dim_feedforward=4 * n_embd,
                dropout=0.0,
                activation='gelu',
                batch_first=True,
                norm_first=True
            ) for _ in range(n_layer)
        ])

        self.ln_f = nn.LayerNorm(n_embd)
        # NC Analysis happens here:
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        b, t = idx.shape
        device = idx.device

        pos = torch.arange(0, t, device=device).unsqueeze(0)
        x = self.wte(idx) + self.wpe(pos)

        # In modern PyTorch, 'is_causal=True' handles the mask automatically.
        # We pass None for the mask and let the optimized kernel handle it.
        for block in self.h:
            x = block(x, src_mask=None, is_causal=True)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits


# if __name__ == "__main__":
#     # Simple test
#     model = SimpleGPT(vocab_size=65, n_embd=128, n_layer=4, n_head=4)
#     input_ids = torch.randint(0, 65, (2, 10))  # batch_size=2, seq_length=10
#     outputs = model(input_ids)
#     print(outputs.shape)  # Expected: (2, 10, 1000)
#
#     # Count parameters
#     def count_parameters(model):
#         return sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"Number of parameters: {count_parameters(model)}")