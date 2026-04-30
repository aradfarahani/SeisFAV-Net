"""
SeisFAV-Net: Seismic Fourier-Attention-Variational Network

A modular hybrid architecture combining:
- Fourier Neural Operator (FNO) for spectral feature extraction
- U-Net with Self-Attention for multi-scale processing
- Variational Autoencoder (VAE) for robust latent representations

Author: [Your Name]
License: MIT
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class SeisFAVConfig:
    """Configuration for SeisFAV-Net architecture"""
    input_channels: int = 1
    latent_dim: int = 128
    fno_modes: int = 32
    fno_width: int = 16
    encoder_channels: Tuple[int, ...] = (32, 64, 128)
    use_attention: Tuple[bool, ...] = (False, True, True)
    dropout_rate: float = 0.1
    attention_reduction: int = 8


class SpectralConv1d(nn.Module):
    """Fourier Neural Operator spectral convolution layer"""
    
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.modes = modes
        scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def compl_mul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Complex multiplication in Fourier space"""
        return torch.einsum("bix,iox->box", a, b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, N = x.shape
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(
            B, self.weights.shape[1], x_ft.shape[-1],
            dtype=torch.cfloat, device=x.device
        )
        out_ft[:, :, :self.modes] = self.compl_mul(
            x_ft[:, :, :self.modes], self.weights
        )
        return torch.fft.irfft(out_ft, n=N)


class SelfAttention1d(nn.Module):
    """Self-attention mechanism for 1D signals"""
    
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.query = nn.Conv1d(channels, channels // reduction, 1)
        self.key = nn.Conv1d(channels, channels // reduction, 1)
        self.value = nn.Conv1d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        q = self.query(x).view(B, -1, L).permute(0, 2, 1)
        k = self.key(x).view(B, -1, L)
        v = self.value(x).view(B, -1, L)
        
        attn = torch.softmax(torch.bmm(q, k), dim=-1)
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(B, C, L)
        
        return self.gamma * out + x


class ResidualConvBlock(nn.Module):
    """Residual convolutional block with optional attention"""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 use_attention: bool = False, attention_reduction: int = 8):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.residual = (nn.Conv1d(in_channels, out_channels, 1) 
                        if in_channels != out_channels else nn.Identity())
        self.attention = (SelfAttention1d(out_channels, attention_reduction) 
                         if use_attention else None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.residual(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.attention:
            out = self.attention(out)
        
        return self.relu(out + identity)


class FNOFrontend(nn.Module):
    """Fourier Neural Operator frontend for spectral preprocessing"""
    
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.spectral = SpectralConv1d(in_channels, out_channels, modes)
        self.bypass = nn.Conv1d(in_channels, out_channels, 1)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.spectral(x) + self.bypass(x))


class UNetEncoder(nn.Module):
    """U-Net style encoder with progressive downsampling"""
    
    def __init__(self, in_channels: int, channels: Tuple[int, ...], 
                 use_attention: Tuple[bool, ...], attention_reduction: int = 8):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        prev_ch = in_channels
        for ch, attn in zip(channels, use_attention):
            self.blocks.append(
                ResidualConvBlock(prev_ch, ch, use_attention=attn, 
                                 attention_reduction=attention_reduction)
            )
            self.pools.append(nn.MaxPool1d(2))
            prev_ch = ch

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, list]:
        skip_connections = []
        for i, (block, pool) in enumerate(zip(self.blocks, self.pools)):
            x = block(x)
            if i < len(self.blocks) - 1:  # Don't save last layer as skip
                skip_connections.append(x)
                x = pool(x)
        return x, skip_connections


class VAEBottleneck(nn.Module):
    """Variational bottleneck for latent space regularization"""
    
    def __init__(self, feature_dim: int, latent_dim: int):
        super().__init__()
        self.fc_mu = nn.Linear(feature_dim, latent_dim)
        self.fc_logvar = nn.Linear(feature_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, feature_dim)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C, L = x.shape
        x_flat = x.view(B, -1)
        
        mu = self.fc_mu(x_flat)
        logvar = self.fc_logvar(x_flat)
        z = self.reparameterize(mu, logvar)
        
        x_decoded = self.fc_decode(z).view(B, C, L)
        return x_decoded, mu, logvar


class UNetDecoder(nn.Module):
    """U-Net style decoder with skip connections"""
    
    def __init__(self, channels: Tuple[int, ...], use_attention: Tuple[bool, ...],
                 attention_reduction: int = 8):
        super().__init__()
        self.ups = nn.ModuleList()
        self.blocks = nn.ModuleList()
        
        for i in range(len(channels) - 1):
            in_ch = channels[i]
            out_ch = channels[i + 1]
            self.ups.append(nn.ConvTranspose1d(in_ch, out_ch, 2, stride=2))
            self.blocks.append(
                ResidualConvBlock(in_ch, out_ch, use_attention=use_attention[i],
                                 attention_reduction=attention_reduction)
            )

    def forward(self, x: torch.Tensor, skip_connections: list) -> torch.Tensor:
        for up, block, skip in zip(self.ups, self.blocks, reversed(skip_connections)):
            x = up(x)
            if x.shape[2] != skip.shape[2]:
                x = nn.functional.interpolate(x, size=skip.shape[2], 
                                             mode='linear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = block(x)
        return x


class SeisFAVNet(nn.Module):
    """
    SeisFAV-Net: Seismic Fourier-Attention-Variational Network
    
    A hybrid deep learning architecture for seismic data denoising that combines:
    - Fourier Neural Operator for spectral domain processing
    - U-Net with self-attention for multi-scale spatial feature extraction
    - Variational bottleneck for robust latent space representation
    
    Args:
        input_len: Length of input seismic traces
        config: Configuration object (optional, uses defaults if None)
    
    Input:
        x: Noisy seismic data [batch_size, channels, trace_length]
    
    Output:
        denoised: Denoised seismic data [batch_size, channels, trace_length]
        mu: Latent space mean [batch_size, latent_dim]
        logvar: Latent space log variance [batch_size, latent_dim]
    
    Example:
        >>> config = SeisFAVConfig(latent_dim=128, fno_modes=32)
        >>> model = SeisFAVNet(input_len=1024, config=config)
        >>> noisy = torch.randn(4, 1, 1024)
        >>> denoised, mu, logvar = model(noisy)
    """
    
    def __init__(self, input_len: int, config: Optional[SeisFAVConfig] = None):
        super().__init__()
        self.config = config or SeisFAVConfig()
        self.input_len = input_len
        
        # FNO frontend for spectral processing
        self.fno_frontend = FNOFrontend(
            self.config.input_channels,
            self.config.fno_width,
            self.config.fno_modes
        )
        
        # U-Net encoder with attention
        self.encoder = UNetEncoder(
            self.config.fno_width,
            self.config.encoder_channels,
            self.config.use_attention,
            self.config.attention_reduction
        )
        
        # Bottleneck attention
        self.bottleneck_attn = SelfAttention1d(
            self.config.encoder_channels[-1],
            self.config.attention_reduction
        )
        
        # Compute feature dimensions after encoding
        self.feature_len, self.feature_dim = self._compute_feature_dims()
        
        # VAE bottleneck
        self.vae_bottleneck = VAEBottleneck(self.feature_dim, self.config.latent_dim)
        
        # U-Net decoder with attention
        decoder_channels = tuple(reversed(self.config.encoder_channels))
        decoder_attention = tuple(reversed(self.config.use_attention[:-1]))
        self.decoder = UNetDecoder(
            decoder_channels,
            decoder_attention,
            self.config.attention_reduction
        )
        
        # Output head for noise prediction
        self.output_head = nn.Sequential(
            nn.Dropout(self.config.dropout_rate),
            nn.Conv1d(self.config.encoder_channels[0], self.config.input_channels, 1)
        )

    def _compute_feature_dims(self) -> Tuple[int, int]:
        """Compute feature dimensions after encoder for VAE"""
        with torch.no_grad():
            dummy = torch.zeros(1, self.config.input_channels, self.input_len)
            x = self.fno_frontend(dummy)
            x, _ = self.encoder(x)
            return x.shape[2], x.shape[1] * x.shape[2]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through SeisFAV-Net
        
        Args:
            x: Input noisy seismic data [B, C, L]
            
        Returns:
            denoised: Denoised output [B, C, L]
            mu: Latent mean [B, latent_dim]
            logvar: Latent log variance [B, latent_dim]
        """
        noisy = x
        
        # Spectral preprocessing with FNO
        x = self.fno_frontend(x)
        
        # Encode with skip connections
        x, skip_connections = self.encoder(x)
        x = self.bottleneck_attn(x)
        
        # VAE bottleneck
        x, mu, logvar = self.vae_bottleneck(x)
        
        # Decode with skip connections
        x = self.decoder(x, skip_connections)
        
        # Predict noise and subtract from input
        noise_pred = self.output_head(x)
        denoised = noisy - noise_pred
        
        return denoised, mu, logvar

    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def from_pretrained(cls, path: str, input_len: int, 
                       config: Optional[SeisFAVConfig] = None) -> 'SeisFAVNet':
        """Load pretrained model from checkpoint"""
        model = cls(input_len, config)
        model.load_state_dict(torch.load(path, map_location='cpu'))
        return model


# ============================
# Loss Function
# ============================

class SeisFAVLoss(nn.Module):
    """
    Combined loss function for SeisFAV-Net training
    
    Combines:
    - MSE reconstruction loss
    - L1 loss for sparsity
    - KL divergence for VAE regularization
    
    Args:
        beta_kl: Weight for KL divergence term (default: 0.01)
        l1_weight: Weight for L1 loss term (default: 0.1)
    """
    
    def __init__(self, beta_kl: float = 0.01, l1_weight: float = 0.1):
        super().__init__()
        self.beta_kl = beta_kl
        self.l1_weight = l1_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Compute combined loss
        
        Args:
            pred: Predicted denoised output
            target: Ground truth clean data
            mu: Latent mean from VAE
            logvar: Latent log variance from VAE
            
        Returns:
            Combined loss value
        """
        # Reconstruction loss
        mse_loss = nn.functional.mse_loss(pred, target)
        
        # L1 loss for sparsity
        l1_loss = nn.functional.l1_loss(pred, target)
        
        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        return mse_loss + self.l1_weight * l1_loss + self.beta_kl * kl_loss


# ============================
# Example Usage
# ============================

if __name__ == "__main__":
    # Create model with custom config
    config = SeisFAVConfig(
        latent_dim=128,
        fno_modes=32,
        encoder_channels=(32, 64, 128),
        use_attention=(False, True, True),
        dropout_rate=0.1
    )
    
    model = SeisFAVNet(input_len=1024, config=config)
    print(f"Model parameters: {model.get_num_parameters():,}")
    
    # Test forward pass
    batch_size = 4
    noisy_input = torch.randn(batch_size, 1, 1024)
    denoised, mu, logvar = model(noisy_input)
    
    print(f"Input shape: {noisy_input.shape}")
    print(f"Output shape: {denoised.shape}")
    print(f"Latent mu shape: {mu.shape}")
    print(f"Latent logvar shape: {logvar.shape}")
    
    # Test loss
    clean_target = torch.randn_like(noisy_input)
    criterion = SeisFAVLoss(beta_kl=0.01, l1_weight=0.1)
    loss = criterion(denoised, clean_target, mu, logvar)
    print(f"Loss: {loss.item():.6f}")
