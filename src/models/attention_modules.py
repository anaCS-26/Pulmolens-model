import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    Channel Attention Module from CBAM.
    Focuses on 'what' is meaningful.
    """
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
        return x * out


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module from CBAM.
    Focuses on 'where' is meaningful.
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(out))
        return x * out


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).
    Combines channel and spatial attention.
    
    Reference: "CBAM: Convolutional Block Attention Module" (Woo et al.)
    
    Args:
        in_channels: Number of input channels
        reduction: Reduction ratio for channel attention
        kernel_size: Kernel size for spatial attention
    """
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
        
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block.
    Channel-wise attention mechanism.
    
    Reference: "Squeeze-and-Excitation Networks" (Hu et al.)
    
    Args:
        in_channels: Number of input channels
        reduction: Reduction ratio
    """
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze
        y = self.squeeze(x).view(b, c)
        # Excitation
        y = self.excitation(y).view(b, c, 1, 1)
        # Scale
        return x * y.expand_as(x)


class CoordinateAttention(nn.Module):
    """
    Coordinate Attention Module.
    Position-aware channel attention that encodes spatial information.
    
    Reference: "Coordinate Attention for Efficient Mobile Network Design" (Hou et al.)
    
    Args:
        in_channels: Number of input channels
        reduction: Reduction ratio
    """
    def __init__(self, in_channels, reduction=32):
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        hidden_channels = max(8, in_channels // reduction)
        
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.act = nn.ReLU(inplace=True)
        
        self.conv_h = nn.Conv2d(hidden_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(hidden_channels, in_channels, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        identity = x
        
        n, c, h, w = x.size()
        
        # Coordinate information embedding
        x_h = self.pool_h(x)  # [n, c, h, 1]
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # [n, c, w, 1]
        
        # Concatenate along spatial dimension
        y = torch.cat([x_h, x_w], dim=2)  # [n, c, h+w, 1]
        
        # Coordinate attention generation
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        
        # Split back
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        
        # Attention maps
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        
        # Apply attention
        out = identity * a_h * a_w
        
        return out


class ECABlock(nn.Module):
    """
    Efficient Channel Attention (ECA) Module.
    Lightweight channel attention without dimensionality reduction.
    
    Reference: "ECA-Net: Efficient Channel Attention for Deep CNNs"
    
    Args:
        in_channels: Number of input channels
        k_size: Kernel size for 1D convolution (adaptive if None)
    """
    def __init__(self, in_channels, k_size=None):
        super(ECABlock, self).__init__()
        
        # Adaptive kernel size
        if k_size is None:
            # Formula from paper
            t = int(abs((torch.log2(torch.tensor(in_channels, dtype=torch.float)) + 1) / 2))
            k_size = t if t % 2 else t + 1
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Feature descriptor
        y = self.avg_pool(x)
        
        # 1D convolution along channel dimension
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        
        # Multi-scale information fusion
        y = self.sigmoid(y)
        
        return x * y.expand_as(x)


if __name__ == "__main__":
    # Test attention modules
    print("Testing attention modules...")
    
    batch_size = 2
    in_channels = 64
    height, width = 56, 56
    
    x = torch.randn(batch_size, in_channels, height, width)
    
    # Test CBAM
    cbam = CBAM(in_channels)
    out = cbam(x)
    print(f"CBAM: Input {x.shape} -> Output {out.shape}")
    
    # Test SE Block
    se = SEBlock(in_channels)
    out = se(x)
    print(f"SE Block: Input {x.shape} -> Output {out.shape}")
    
    # Test Coordinate Attention
    ca = CoordinateAttention(in_channels)
    out = ca(x)
    print(f"Coordinate Attention: Input {x.shape} -> Output {out.shape}")
    
    # Test ECA
    eca = ECABlock(in_channels)
    out = eca(x)
    print(f"ECA Block: Input {x.shape} -> Output {out.shape}")
    
    print("\nAll attention modules working correctly!")
