import torch
import torch.nn as nn


class SimpleGPT(nn.Module):

    def __init__(
            self,
            vocab_size=65,
            n_embd=128,
            n_layer=4,
            n_head=4,
            block_size=256
    ):
        super().__init__()
        self.block_size = block_size

        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)

        self.h = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=n_embd,
                    nhead=n_head,
                    dim_feedforward=4 * n_embd,
                    dropout=0.0,
                    activation='gelu',
                    batch_first=True,
                    norm_first=True
                ) for _ in range(n_layer)
            ]
        )

        self.ln_f = nn.LayerNorm(n_embd)

        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        # Tie weights - wte and lm_head share the same tensor
        # self.lm_head.weight = self.wte.weight
        #  tie the input token embeddings directly to the output classifier layer weights to save memory and parameter count
        #  For Neural Collapse, this acts as a severe structural constraint.
        #  If the input geometry and output geometry are forced to be identical,
        #  the terminal classifier layer cannot freely collapse into an Equiangular Tight Frame (ETF)
        #  and align properly without also forcing the input embeddings to satisfy identical rigid conditions immediately,
        #  which usually halts training before hitting 100% accuracy.

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        b, t = idx.shape
        device = idx.device

        pos = torch.arange(0, t, device=device).unsqueeze(0)
        x = self.wte(idx) + self.wpe(pos)

        seq_len = x.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=x.device
        )

        for block in self.h:
            x = block(x, src_mask=causal_mask, is_causal=True)

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
