import torch.nn as nn
import torch
import math
from torch import Tensor
import torch.jit as jit

class CustomLSTM(nn.Module):
    def __init__(self, input_sz, hidden_sz):
        super().__init__()
        self.input_sz = input_sz
        self.hidden_size = hidden_sz
        self.W = nn.Parameter(torch.Tensor(input_sz, hidden_sz * 4))
        self.U = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz * 4))
        self.bias = nn.Parameter(torch.Tensor(hidden_sz * 4))
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x,
                init_states=None):
        bs, seq_sz, _ = x.size()
        hidden_seq = []
        if init_states is None:
            h_t, c_t = (torch.zeros(bs, self.hidden_size).to(x.device),
                        torch.zeros(bs, self.hidden_size).to(x.device))
        else:
            h_t, c_t = init_states[0][0], init_states[1][0]

        HS = self.hidden_size
        for t in range(seq_sz):
            x_t = x[:, t, :]
            gates = x_t @ self.W + h_t @ self.U + self.bias
            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:, :HS]), # input gate
                torch.sigmoid(gates[:, HS:HS*2]), # forget gate
                torch.tanh(gates[:, HS*2:HS*3]),
                torch.sigmoid(gates[:, HS*3:]), # output gate
            )
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)
    

class LSTMCell(jit.ScriptModule):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.randn(4 * hidden_size))
        self.expres = nn.Parameter(torch.Tensor([0.3]))
        self.init_weights(hidden_size)

    def init_weights(self, hidden_size):
        stdv = 1.0 / math.sqrt(hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)    

    @jit.script_method
    def forward(
        self, input: Tensor, state: tuple[Tensor, Tensor]
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        hx, cx = state
        gates = (
            torch.mm(input, self.weight_ih.t())
            + self.bias_ih
            + torch.mm(hx, self.weight_hh.t())
            + self.bias_hh
        )
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)
        forgetgate_cx = (forgetgate * cx)
        ingate_cellgate = ingate * cellgate   
        cy = self.expres * (forgetgate_cx * ingate_cellgate) - (forgetgate_cx + ingate_cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, (hy, cy)

class LSTMLayerJIT(jit.ScriptModule):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.cell = LSTMCell(input_size=input_size, hidden_size=hidden_size)

    def init_hidden(self, batch_size: int, device: torch.device) -> tuple[Tensor, Tensor]:
      h_0 = torch.zeros(batch_size, 128).to(device)
      c_0 = torch.zeros(batch_size, 128).to(device)
      return (h_0, c_0)    

    @jit.script_method
    def forward(
        self, input: Tensor, state: tuple[Tensor, Tensor]
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        #state = torch.jit.annotate(tuple[Tensor, Tensor], self.init_hidden(input.shape[0], input.device))
        inputs = input.unbind(1)
        outputs = torch.jit.annotate(list[Tensor], [])
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state)
            outputs += [out.unsqueeze(0)]
        outputs = torch.cat(outputs, dim=0)
        outputs = outputs.transpose(0, 1).contiguous()    
        return outputs, state            
