import torch
import torch.nn as nn

from einops import einsum, reduce, rearrange
from jaxtyping import Float, Int


# TODO: Does this need special initialization treatment?
class ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.hidden_dim = dim
        # Pre-Norm architecture stabilizes deep gradients
        self.ln = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, input):
        residual = input
        out = self.ln(input)
        out = self.fc1(out)
        out = self.act(out)
        out = self.fc2(out)
        return residual + out   # Additive skip connection
    
# wang20261000layernetworksselfsupervised paper's residual block
class CustomResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.hidden_dim = dim
        self.fc1 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.act1 = nn.SiLU()

        self.fc2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.act2 = nn.SiLU()
        
        self.fc3 = nn.Linear(dim, dim)
        self.ln3 = nn.LayerNorm(dim)
        self.act3 = nn.SiLU()
        
        self.fc4 = nn.Linear(dim, dim)
        self.ln4 = nn.LayerNorm(dim)
        self.act4 = nn.SiLU()
    
    def forward(self, input):
        residual = input
        out = self.fc1(input)
        out = self.ln1(out)
        out = self.act1(out)

        out = self.fc2(out)
        out = self.ln2(out)
        out = self.act2(out)

        out = self.fc3(out)
        out = self.ln3(out)
        out = self.act3(out)

        out = self.fc4(out)
        out = self.ln4(out)
        out = self.act4(out)
        return residual + out

class CustomResNetConfigDepth(nn.Module):
    def __init__(self, depth: int, dim: int):
        self.depth = depth
        self.hidden_dim = dim

        self.fc0 = nn.Linear(dim, dim)
        self.ln0 = nn.LayerNorm(dim)
        self.act0 = nn.SiLU()

        self.layers = nn.ModuleList(
            [CustomResidualBlock(dim) for _ in range(depth)]
        )

        self.head = nn.Linear(dim, dim)
    
    def forward(self, input):
        out = self.fc0(input)
        out = self.ln0(out)
        out = self.act0(out)

        for layer in self.layers:
            out = layer(out)
        
        return self.head(out)


    
## TODO: As future work, we could fuse these into a single 
# Triton kernel and benchmark performance gain

class ResNetConfigDepth(nn.Module):
    def __init__(self, depth: int, state_dim: int, hidden_dim: int):
        super().__init__()
        self.depth = depth
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        
        self.state_proj = nn.Linear(state_dim, hidden_dim)

        self.layers = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(depth)
        ])


        # self.head = nn.Linear(dim, dim)
        self.final_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, input):
        out = self.state_proj(input)

        for layer in self.layers:
            out = layer(out)
        
        out = self.final_norm(out)
        return out


if __name__ == '__main__':
    dim = 256
    batch = 32
    dummy_input = torch.randn((batch, dim), dtype=torch.float16)
    
    # Test single ResidualBlock
    residual_block = ResidualBlock(dim).to(torch.float16)
    print(residual_block(dummy_input))

    # Test ResNetCustomDepth module
    depth = 3
    resnet_depth_3 = ResNetConfigDepth(depth, dim).to(torch.float16)
    print(resnet_depth_3(dummy_input))
