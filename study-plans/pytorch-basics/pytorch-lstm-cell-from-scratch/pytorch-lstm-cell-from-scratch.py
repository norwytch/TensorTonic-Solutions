import torch
import torch.nn as nn

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        """
        Returns: None
        """
        super().__init__()
        # input gate
        self.W_ii = nn.Parameter(torch.randn(hidden_size, input_size))
        self.W_hi = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.b_ii = nn.Parameter(torch.randn(hidden_size,))
        self.b_hi = nn.Parameter(torch.randn(hidden_size,))
        # forget gate
        self.W_if = nn.Parameter(torch.randn(hidden_size, input_size))
        self.W_hf = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.b_if = nn.Parameter(torch.randn(hidden_size,))
        self.b_hf = nn.Parameter(torch.randn(hidden_size,))
        # cell candidate
        self.W_ig = nn.Parameter(torch.randn(hidden_size, input_size))
        self.W_hg = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.b_ig = nn.Parameter(torch.randn(hidden_size,))
        self.b_hg = nn.Parameter(torch.randn(hidden_size,))
        # output gate
        self.W_io = nn.Parameter(torch.randn(hidden_size, input_size))
        self.W_ho = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.b_io = nn.Parameter(torch.randn(hidden_size,))
        self.b_ho = nn.Parameter(torch.randn(hidden_size,))

    def forward(self, x, h_prev, c_prev):
        """
        Returns: tuple of (h_t, c_t) tensors
        """
        i = torch.sigmoid(x@self.W_ii.T + self.b_ii + h_prev@self.W_hi.T + self.b_hi)
        f = torch.sigmoid(x@self.W_if.T + self.b_if + h_prev@self.W_hf.T + self.b_hf)
        g = torch.tanh(x@self.W_ig.T + self.b_ig + h_prev@self.W_hg.T + self.b_hg)
        o = torch.sigmoid(x@self.W_io.T + self.b_io + h_prev@self.W_ho.T + self.b_ho)
        c_t = f * c_prev + i * g
        h_t = o * torch.tanh(c_t)

        return (h_t, c_t)
