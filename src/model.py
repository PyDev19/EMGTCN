import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MeanOverTime(nn.Module):
    """
    Computes the mean over the time dimension (dim=1) and supports masking.

    Example:
        rnn_out, _ = rnn(x)               # (batch, seq_len, hidden)
        mask = (x != pad_value).any(-1)   # (batch, seq_len)
        mean = MeanOverTime()(rnn_out, mask)
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, mask=None):
        """
        x: Tensor of shape (batch, time, features)
        mask: Optional boolean mask (batch, time), where True = keep
        """
        if mask is not None:
            mask = mask.unsqueeze(-1).float()  # (batch, time, 1)
            x_masked = x * mask
            summed = x_masked.sum(dim=1)  # sum over time
            counts = mask.sum(dim=1).clamp(min=1.0)  # avoid divide by zero
            mean = summed / counts
        else:
            mean = x.mean(dim=1)

        return mean


class CausalConv1d(nn.Module):
    """
    1D Causal Convolution Layer with optional masking.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output filters.
        kernel_size: Size of convolution kernel.
        stride: Convolution stride.
        dilation: Dilation factor.
        activation: Optional activation (e.g., nn.ReLU()).
        use_bias: Whether to use bias.

    Input:
        x: Tensor of shape (batch, channels, time)
        mask (optional): Boolean mask (batch, time)

    Output:
        Tensor of shape (batch, out_channels, time)
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        activation=None,
        use_bias=True,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.activation = activation

        # Initialize convolution
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            bias=use_bias,
        )

    def forward(self, x, mask=None):
        """
        Forward pass through causal convolution.
        x: (batch, channels, time)
        mask: optional (batch, time)
        """
        # Causal padding: pad only on the left (past)
        pad_left = (self.kernel_size - 1) * self.dilation
        x = F.pad(x, (pad_left, 0))

        if mask is not None:
            # Expand mask to match channel dimension
            mask = mask.unsqueeze(1).float()  # (batch, 1, time)
            x = x * mask

        # Perform convolution
        out = self.conv(x)

        # Apply mask again to keep padded timesteps zeroed
        if mask is not None:
            out = out * mask[..., : out.shape[-1]]

        # Activation
        if self.activation is not None:
            out = self.activation(out)

        return out


class TCNBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, dilation, dropout_rate=None
    ):
        super().__init__()

        self.cconv = CausalConv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            activation=nn.ReLU(),
        )
        self.dropout = (
            nn.Dropout(dropout_rate) if dropout_rate is not None else nn.Identity()
        )

    def forward(self, x):
        x = self.cconv(x)
        x = self.dropout(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        filters_per_layer,
        kernel_sizes,
        base_dilation=1,
        dropout_rate=None,
    ):
        super().__init__()

        assert len(filters_per_layer) == len(
            kernel_sizes
        ), "filters and kernel_sizes must match"

        self.blocks = nn.ModuleList()
        dilation = base_dilation

        for out_ch, ksize in zip(filters_per_layer, kernel_sizes):
            block = TCNBlock(in_channels, out_ch, ksize, dilation, dropout_rate)
            self.blocks.append(block)

            in_channels = out_ch
            dilation *= 2

        self.dilation_out = dilation

        self.residual_conv = CausalConv1d(
            in_channels=filters_per_layer[-1],
            out_channels=filters_per_layer[-1],
            kernel_size=1,
            activation=None,
        )

    def forward(self, x):
        out = x
        for block in self.blocks:
            out = block(out)

        residual = self.residual_conv(x)
        return F.relu(out + residual)


class TCNAot(nn.Module):
    def __init__(self, config: dict[str, any]):
        super().__init__()

        tcn_layers = config["tcn_layers"]
        residual_blocks = config["residual_blocks"]
        filters = config["filters"]
        filters_size = [
            [config["filters_size"] for _ in range(tcn_layers[i])]
            for i in range(residual_blocks)
        ]
        masking = config["masking"]

        assert (
            len(tcn_layers) == residual_blocks
        ), "Length of tcn_layers must match residual_blocks"

        for i, layer_size in enumerate(tcn_layers):
            if layer_size != len(filters[i]):
                raise AssertionError(
                    "Number of filters have to be same to layers. Found filters {}, layers {}".format(
                        np.sum(np.array(filters).shape), np.sum(tcn_layers)
                    )
                )
        
        self.blocks = nn.ModuleList()
        in_channels = config["input_channels"]
        dilation = 1

        for b in range(residual_blocks):
            filters_per_layer = [filters for _ in range(tcn_layers[b])]
            kernel_sizes = [filters_size for _ in range(tcn_layers[b])]

            block = ResidualBlock(
                in_channels=in_channels,
                filters_per_layer=filters_per_layer,
                kernel_sizes=kernel_sizes,
                base_dilation=dilation,
                dropout_rate=config["dropout_rate"],
            )
            self.blocks.append(block)
            in_channels = filters_per_layer[-1]
            dilation = block.dilation_out

        self.output_channels = in_channels
    
    def forward(self, x, mask=None):
        """
        x: (batch, channels, time)
        mask: optional (batch, time)
        """
        out = x
        for block in self.blocks:
            out = block(out)

        return out
