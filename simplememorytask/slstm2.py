import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class sLSTM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.W = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.U = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd)
        self.f_bias = nn.Parameter(torch.zeros(config.n_embd).fill_(3.0))  # Initialize with values between 3 and 6
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, hidden_states):
        batch_size, seq_len, _ = x.size()

        Z = self.W(x) + self.U(hidden_states)
        i, f, c, o = Z.chunk(4, dim=-1)

        # Stabilization technique to avoid overflow
        stab_factor = torch.max(torch.max(i), torch.max(f))
        i = torch.exp(i - stab_factor)
        f = torch.exp(f + self.f_bias - stab_factor)

        cell_state = f * hidden_states + i * torch.tanh(c)
        normalizer_state = f + i

        cell_state = cell_state / normalizer_state

        o = torch.sigmoid(o)
        hidden_state = o * torch.tanh(cell_state)

        hidden_state = self.o_proj(hidden_state)
        hidden_state = self.dropout(hidden_state)

        return x, hidden_state

class mLSTM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.W_q = nn.Linear(config.n_embd, config.n_embd)
        self.W_k = nn.Linear(config.n_embd, config.n_embd)
        self.W_v = nn.Linear(config.n_embd, config.n_embd)
        self.W_f = nn.Linear(config.n_embd, config.n_embd)
        self.W_i = nn.Linear(config.n_embd, config.n_embd)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd)
        self.f_bias = nn.Parameter(torch.ones(config.n_embd))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        f = torch.sigmoid(self.W_f(x) + self.f_bias)
        i = torch.exp(self.W_i(x))

        memory = torch.softmax(torch.matmul(q, k.transpose(1, 2)) / math.sqrt(k.size(-1)), dim=-1)
        memory = torch.matmul(memory, v)

        memory = f * memory + i * v

        normalizer = f + i

        memory = memory / normalizer

        out = self.o_proj(memory)
        out = self.dropout(out)

        return out

class xLSTMBlock(nn.Module):
    def __init__(self, config, ratio_mLSTM=0.5):
        super().__init__()
        self.num_sLSTM = int(config.n_layer * (1 - ratio_mLSTM))
        self.num_mLSTM = config.n_layer - self.num_sLSTM
        self.sLSTM_blocks = nn.ModuleList([sLSTM(config) for _ in range(self.num_sLSTM)])
        self.mLSTM_blocks = nn.ModuleList([mLSTM(config) for _ in range(self.num_mLSTM)])
        self.ln_s = nn.LayerNorm(config.n_embd)
        self.ln_m = nn.LayerNorm(config.n_embd)

    def forward(self, x, hidden_states):
        for i in range(self.num_sLSTM):
            x, hidden_states = self.sLSTM_blocks[i](self.ln_s(x), hidden_states)
        for i in range(self.num_mLSTM):
            x = self.mLSTM_blocks[i](self.ln_m(x))
        return x, hidden_states
