import torch
import torch.nn as nn

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        """
        Returns: None
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))

    def forward(self, x):
        """
        Returns: convolved output tensor of shape (batch, out_channels, H-k+1, W-k+1)
        """
        N, C_in, H, W = x.shape
        output = torch.zeros(N, self.out_channels, H-self.kernel_size + 1, W - self.kernel_size + 1)
        for i in range(H-self.kernel_size+1):
            for j in range(W-self.kernel_size+1):
                window = x[:,:, i:i+self.kernel_size, j:j+self.kernel_size]
                output[:,:,i,j] = (window[:,None] * self.weight[None]).sum(dim=(2,3,4)) + self.bias
        return output