import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        """
        Returns: None
        """
        super().__init__()
        self.d_model = d_model
        self.h = num_heads
        self.d_k = d_model//num_heads
        self.W_q = nn.Parameter((torch.randn(self.d_model,self.d_model)))
        self.W_k = nn.Parameter((torch.randn(self.d_model,self.d_model)))
        self.W_v = nn.Parameter((torch.randn(self.d_model,self.d_model)))
        self.W_o = nn.Parameter((torch.randn(self.d_model,self.d_model)))
    

    def forward(self, Q, K, V):
        """
        Returns: output tensor
        """
        Q_proj = Q @ self.W_q
        K_proj = K @ self.W_k
        V_proj = V @ self.W_v
        batch = Q_proj.shape[0]
        seq_len = Q_proj.shape[1]
        Q_proj=Q_proj.reshape(batch, seq_len, self.h, self.d_k).transpose(1,2)
        K_proj=K_proj.reshape(batch, seq_len, self.h, self.d_k).transpose(1,2)
        V_proj=V_proj.reshape(batch, seq_len, self.h, self.d_k).transpose(1,2)
        scores = Q_proj@K_proj.transpose(-2, -1)/self.d_k**.5
        weights = torch.nn.functional.softmax(scores, dim=-1)
        output = weights @ V_proj
        output = output.transpose(1,2).contiguous().reshape(batch,seq_len,self.d_model)
        output = output @ self.W_o
        return output