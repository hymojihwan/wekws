import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1DBlock(nn.Module):
    """
    1D Convolutional block used in the encoder, decoder, and separation network.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, activation=True):
        super(Conv1DBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.PReLU() if activation else None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x


class ConvTasNet(nn.Module):
    def __init__(
        self, 
        enc_kernel_size=16, 
        enc_num_feats=512, 
        num_blocks=8, 
        num_layers=4, 
        bottleneck_channels=128, 
        num_sources=2
    ):
        """
        Conv-TasNet Model
        Args:
            enc_kernel_size: Kernel size of the encoder.
            enc_num_feats: Number of features in the encoder output.
            num_blocks: Number of blocks in the temporal convolution network (TCN).
            num_layers: Number of convolutional layers per TCN block.
            bottleneck_channels: Number of channels in the bottleneck layer.
            num_sources: Number of sources to separate.
        """
        super(ConvTasNet, self).__init__()
        self.enc_kernel_size = enc_kernel_size
        self.enc_num_feats = enc_num_feats
        self.num_sources = num_sources

        # Encoder
        self.encoder = nn.Conv1d(1, enc_num_feats, kernel_size=enc_kernel_size, stride=enc_kernel_size // 2, padding=0)

        # Bottleneck layer
        self.bottleneck = nn.Conv1d(enc_num_feats, bottleneck_channels, kernel_size=1)

        # Temporal Convolution Network (TCN)
        self.tcn = nn.ModuleList()
        for _ in range(num_blocks):
            self.tcn.append(
                TCNBlock(
                    in_channels=bottleneck_channels,
                    out_channels=bottleneck_channels,
                    num_layers=num_layers
                )
            )

        # Mask Generators for each source
        self.mask_generators = nn.ModuleList([
            nn.Conv1d(bottleneck_channels, enc_num_feats, kernel_size=1) for _ in range(num_sources)
        ])

        # Decoder
        self.decoder = nn.ConvTranspose1d(
            enc_num_feats, 1, kernel_size=enc_kernel_size, stride=enc_kernel_size // 2, padding=0
        )

    def forward(self, mixture):
        """
        Forward pass of the Conv-TasNet model.
        Args:
            mixture: Input mixture signal, shape (batch, time).
        Returns:
            List of separated signals, each of shape (batch, time).
        """
        # Encoder
        mixture = mixture.unsqueeze(1)  # (batch, 1, time)
        enc_out = self.encoder(mixture)  # (batch, enc_num_feats, time)

        # Bottleneck
        bottleneck_out = self.bottleneck(enc_out)  # (batch, bottleneck_channels, time)

        # Temporal Convolution Network
        tcn_out = bottleneck_out
        for block in self.tcn:
            tcn_out = block(tcn_out)

        # Mask generation
        masks = [F.relu(mask_gen(tcn_out)) for mask_gen in self.mask_generators]  # List of (batch, enc_num_feats, time)

        # Apply masks and decode
        separated_signals = []
        for mask in masks:
            masked_enc_out = enc_out * mask
            separated_signal = self.decoder(masked_enc_out)
            separated_signal = separated_signal.squeeze(1)  # (batch, time)
            separated_signals.append(separated_signal)

        return separated_signals


class TCNBlock(nn.Module):
    """
    A single block of Temporal Convolution Network (TCN).
    """
    def __init__(self, in_channels, out_channels, num_layers, kernel_size=3, dilation_base=2):
        super(TCNBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            dilation = dilation_base**i
            self.layers.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size, padding=(kernel_size - 1) * dilation, dilation=dilation),
                    nn.BatchNorm1d(out_channels),
                    nn.PReLU()
                )
            )
            self.layers.append(nn.Conv1d(out_channels, out_channels, kernel_size=1))  # Residual connection

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


if __name__ == "__main__":
    # Example usage
    model = ConvTasNet()
    mixture = torch.rand(4, 16000)  # Batch size = 4, 1-second audio at 16kHz
    outputs = model(mixture)
    print(f"Output shape for source 1: {outputs[0].shape}")  # (batch, time)
    print(f"Output shape for source 2: {outputs[1].shape}")  # (batch, time)