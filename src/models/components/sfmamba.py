from functools import partial
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from monai.networks.layers.utils import get_rel_pos_embedding_layer
from monai.utils import pytorch_after
from src.utils.utils import add_torch_shape_forvs
from mamba_ssm import Mamba, Mamba2
from torchinfo import summary
import collections.abc
from itertools import repeat


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=None,
        bias=True,
        drop=0.0,
        use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = (
            norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        )
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class ConvStem(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvStem, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class MambaLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        expand = 2
        headdim = dim * expand // 8
        # self.mamba = Mamba(
        #     d_model=dim,  # Model dimension d_model
        #     d_state=8,  # SSM state expansion factor
        #     d_conv=2,  # Local convolution width
        #     expand=2,  # Block expansion factor
        # )
        self.mamba = Mamba2(
            d_model=dim,  # Model dimension d_model
            d_state=128,  # SSM state expansion factor
            d_conv=4,  # Local convolution width
            expand=2,  # Block expansion factor
            headdim=headdim,
        )

    def forward(self, x):
        B, C = x.shape[:2]
        act_x = x
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_mamba = self.mamba(x_flat)
        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)
        out += act_x
        return out


class MSFEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, scales=[3, 5, 7]):
        super(MSFEncoder, self).__init__()
        self.out_channels = out_channels
        # 有128通道后再进行mamba提取，即最后两层encoder,最后两层不用卷积
        if self.out_channels >= 128:
            self.mamba = MambaLayer(dim=in_channels)
        else:
            # Multi-scale convolutions
            self.multi_scale_convs = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv3d(
                            in_channels, in_channels, kernel_size=s, padding=s // 2
                        ),
                        nn.BatchNorm3d(in_channels),
                        nn.ReLU(inplace=True),
                    )
                    for s in scales
                ]
            )

            # 1x1 convolution to fuse channels after concatenation of multi-scale features
            self.fusion_conv = nn.Sequential(
                nn.Conv3d(in_channels * len(scales), in_channels, kernel_size=1),
                nn.BatchNorm3d(in_channels),
                nn.ReLU(inplace=True),
            )

        # Downsampling layer to reduce spatial dimensions
        self.downsample = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        if self.out_channels >= 128:
            fused = x
            mamba_res = self.mamba(fused)
            downscaled = self.downsample(mamba_res)  # Downsample spatially
        else:
            # Apply each scale and concatenate along the channel dimension
            features = [conv(x) for conv in self.multi_scale_convs]
            fused = torch.cat(features, dim=1)  # Concatenate along channels
            fused = self.fusion_conv(fused)  # Reduce channels back to `out_channels`
            downscaled = self.downsample(fused)  # Downsample spatially
        return downscaled


class FeatureExtractor(nn.Module):
    def __init__(self, in_channels):
        super(FeatureExtractor, self).__init__()
        out_channels_1 = in_channels * 2**1
        out_channels_2 = in_channels * 2**2
        out_channels_3 = in_channels * 2**3
        out_channels_4 = in_channels * 2**4

        self.msf_encoder1 = MSFEncoder(
            in_channels=in_channels, out_channels=out_channels_1
        )
        self.msf_encoder2 = MSFEncoder(
            in_channels=out_channels_1, out_channels=out_channels_2
        )
        self.msf_encoder3 = MSFEncoder(
            in_channels=out_channels_2, out_channels=out_channels_3
        )
        self.msf_encoder4 = MSFEncoder(
            in_channels=out_channels_3, out_channels=out_channels_4
        )

        self.shortcut_convs = nn.ModuleList(
            [
                nn.Conv3d(
                    in_channels, out_channels_1, kernel_size=1, stride=2, padding=0
                ),
                nn.Conv3d(
                    out_channels_1, out_channels_2, kernel_size=1, stride=2, padding=0
                ),
                nn.Conv3d(
                    out_channels_2, out_channels_3, kernel_size=1, stride=2, padding=0
                ),
                nn.Conv3d(
                    out_channels_3, out_channels_4, kernel_size=1, stride=2, padding=0
                ),
            ]
        )

    def forward(self, x):
        # Layer 1
        residual = x
        x = self.msf_encoder1(x)
        residual = self.shortcut_convs[0](residual)  # Downsample residual
        x = x + residual

        # Layer 2
        residual = x
        x = self.msf_encoder2(x)
        residual = self.shortcut_convs[1](residual)  # Downsample residual
        x = x + residual

        # Layer 3
        residual = x
        x = self.msf_encoder3(x)
        residual = self.shortcut_convs[2](residual)  # Downsample residual
        x = x + residual

        # Layer 4
        residual = x
        x = self.msf_encoder4(x)
        residual = self.shortcut_convs[3](residual)  # Downsample residual
        x = x + residual

        return x


class CrossModalAttention(nn.Module):
    """
    A cross-attention block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"
    One can setup relative positional embedding as described in <https://arxiv.org/abs/2112.01526>
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        hidden_input_size: int | None = None,
        context_input_size: int | None = None,
        dim_head: int | None = None,
        qkv_bias: bool = False,
        save_attn: bool = False,
        causal: bool = False,
        sequence_length: int | None = None,
        rel_pos_embedding: Optional[str] = None,
        input_size: Optional[Tuple] = None,
        attention_dtype: Optional[torch.dtype] = None,
        use_flash_attention: bool = False,
    ) -> None:
        """
        Args:
            hidden_size (int): dimension of hidden layer.
            num_heads (int): number of attention heads.
            dropout_rate (float, optional): fraction of the input units to drop. Defaults to 0.0.
            hidden_input_size (int, optional): dimension of the input tensor. Defaults to hidden_size.
            context_input_size (int, optional): dimension of the context tensor. Defaults to hidden_size.
            dim_head (int, optional): dimension of each head. Defaults to hidden_size // num_heads.
            qkv_bias (bool, optional): bias term for the qkv linear layer. Defaults to False.
            save_attn (bool, optional): to make accessible the attention matrix. Defaults to False.
            causal (bool, optional): whether to use causal attention.
            sequence_length (int, optional): if causal is True, it is necessary to specify the sequence length.
            rel_pos_embedding (str, optional): Add relative positional embeddings to the attention map. For now only
                "decomposed" is supported (see https://arxiv.org/abs/2112.01526). 2D and 3D are supported.
            input_size (tuple(spatial_dim), optional): Input resolution for calculating the relative positional
                parameter size.
            attention_dtype: cast attention operations to this dtype.
            use_flash_attention: if True, use Pytorch's inbuilt flash attention for a memory efficient attention mechanism
                (see https://pytorch.org/docs/2.2/generated/torch.nn.functional.scaled_dot_product_attention.html).
        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if dim_head:
            inner_size = num_heads * dim_head
            self.head_dim = dim_head
        else:
            if hidden_size % num_heads != 0:
                raise ValueError("hidden size should be divisible by num_heads.")
            inner_size = hidden_size
            self.head_dim = hidden_size // num_heads

        if causal and sequence_length is None:
            raise ValueError("sequence_length is necessary for causal attention.")

        if use_flash_attention and not pytorch_after(minor=13, major=1, patch=0):
            raise ValueError(
                "use_flash_attention is only supported for PyTorch versions >= 2.0."
                "Upgrade your PyTorch or set the flag to False."
            )
        if use_flash_attention and save_attn:
            raise ValueError(
                "save_attn has been set to True, but use_flash_attention is also set"
                "to True. save_attn can only be used if use_flash_attention is False"
            )

        if use_flash_attention and rel_pos_embedding is not None:
            raise ValueError(
                "rel_pos_embedding must be None if you are using flash_attention."
            )

        self.num_heads = num_heads
        self.hidden_input_size = hidden_input_size if hidden_input_size else hidden_size
        self.context_input_size = (
            context_input_size if context_input_size else hidden_size
        )
        self.out_proj = nn.Linear(inner_size, self.hidden_input_size)
        # key, query, value projections
        self.to_q = nn.Linear(self.hidden_input_size, inner_size, bias=qkv_bias)
        self.to_k = nn.Linear(self.context_input_size, inner_size, bias=qkv_bias)
        self.to_v = nn.Linear(self.context_input_size, inner_size, bias=qkv_bias)
        self.input_rearrange = Rearrange("b h (l d) -> b l h d", l=num_heads)

        self.out_rearrange = Rearrange("b l h d -> b h (l d)")
        self.drop_output = nn.Dropout(dropout_rate)
        self.drop_weights = nn.Dropout(dropout_rate)
        self.dropout_rate = dropout_rate

        self.scale = self.head_dim**-0.5
        self.save_attn = save_attn
        self.attention_dtype = attention_dtype

        self.causal = causal
        self.sequence_length = sequence_length
        self.use_flash_attention = use_flash_attention

        if causal and sequence_length is not None:
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "causal_mask",
                torch.tril(torch.ones(sequence_length, sequence_length)).view(
                    1, 1, sequence_length, sequence_length
                ),
            )
            self.causal_mask: torch.Tensor
        else:
            self.causal_mask = torch.Tensor()

        self.att_mat = torch.Tensor()
        self.rel_positional_embedding = (
            get_rel_pos_embedding_layer(
                rel_pos_embedding, input_size, self.head_dim, self.num_heads
            )
            if rel_pos_embedding is not None
            else None
        )
        self.input_size = input_size

        mlp_ratio = 4
        drop_out_rate = 0.2
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(
            in_features=hidden_size, hidden_features=mlp_hidden_dim, drop=drop_out_rate, norm_layer=nn.LayerNorm
        )

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None):
        """
        Args:
            x (torch.Tensor): input tensor. B x (s_dim_1 * ... * s_dim_n) x C
            context (torch.Tensor, optional): context tensor. B x (s_dim_1 * ... * s_dim_n) x C

        Return:
            torch.Tensor: B x (s_dim_1 * ... * s_dim_n) x C
        """
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        b, t, c = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (hidden_size)

        q = self.input_rearrange(self.to_q(x))
        kv = context if context is not None else x
        _, kv_t, _ = kv.size()
        k = self.input_rearrange(self.to_k(kv))
        v = self.input_rearrange(self.to_v(kv))

        if self.attention_dtype is not None:
            q = q.to(self.attention_dtype)
            k = k.to(self.attention_dtype)

        if self.use_flash_attention:
            x = torch.nn.functional.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                scale=self.scale,
                dropout_p=self.dropout_rate,
                is_causal=self.causal,
            )
        else:
            att_mat = torch.einsum("blxd,blyd->blxy", q, k) * self.scale
            # apply relative positional embedding if defined
            if self.rel_positional_embedding is not None:
                att_mat = self.rel_positional_embedding(x, att_mat, q)

            if self.causal:
                att_mat = att_mat.masked_fill(
                    self.causal_mask[:, :, :t, :kv_t] == 0, float("-inf")
                )

            att_mat = att_mat.softmax(dim=-1)

            if self.save_attn:
                # no gradients and new tensor;
                # https://pytorch.org/docs/stable/generated/torch.Tensor.detach.html
                self.att_mat = att_mat.detach()

            att_mat = self.drop_weights(att_mat)
            x = torch.einsum("bhxy,bhyd->bhxd", att_mat, v)

        x = self.out_rearrange(x)
        x = self.out_proj(x)
        x = self.drop_output(x)
        x = self.mlp(x)
        return x


class MambaFusionBlock(nn.Module):
    def __init__(
        self,
        dim,
    ):
        super(MambaFusionBlock, self).__init__()
        # self.mamba = Mamba(
        #     d_model=dim,  # Model dimension d_model
        #     d_state=8,  # SSM state expansion factor
        #     d_conv=2,  # Local convolution width
        #     expand=2,  # Block expansion factor
        # )
        # causal_conv1d要求步幅(x.s arstride(0)和x.s arstride(2))为8的倍数
        # d_model * expand / headdim 是 8 的 倍数
        self.mamba = Mamba2(
            d_model=dim,  # Model dimension d_model
            d_state=128,  # SSM state expansion factor
            d_conv=4,  # Local convolution width
            expand=2,  # Block expansion factor
            headdim=64,
        )
        mlp_ratio = 4
        drop_out_rate = 0.2
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, drop=drop_out_rate, norm_layer=nn.LayerNorm
        )

    def forward(self, x):
        x_mamba = self.mamba(x) + x
        out = x_mamba
        out = self.mlp(out)
        return out


class ClsHead(nn.Module):
    def __init__(self, input_dim=256, num_classes=2):
        super(ClsHead, self).__init__()
        # Pooling layer to aggregate sequence information
        self.pooling = nn.AdaptiveAvgPool1d(1)
        # Fully connected layers
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        # x: [batch_size, sequence_length, feature_dim] -> [6, 512, 256]

        # Permute and pool to reduce sequence dimension
        x = x.permute(0, 2, 1)  # [batch_size, feature_dim, sequence_length]
        x = self.pooling(x).squeeze(-1)  # [batch_size, feature_dim]

        # Classification layers
        x = self.fc(x)
        return x


class MultiSeqMambaModel(nn.Module):
    def __init__(
        self,
        in_channels=1,
        stem_channels=16,
        num_classes=2,
    ):
        super(MultiSeqMambaModel, self).__init__()
        embed_dim = stem_channels * 2**4
        # Define ConvStem for initial feature extraction
        self.conv_stem = ConvStem(in_channels=in_channels, out_channels=stem_channels)

        # Define feature extractor
        self.feature_extractor = FeatureExtractor(in_channels=stem_channels)

        # Cross-modal attention block (hidden size is adjustable based on encoder channels and model design)
        self.cross_modal_attn_block = CrossModalAttention(
            hidden_size=embed_dim,
            num_heads=16,
            dropout_rate=0.1,
        )

        # Mamba Fusion Block
        self.mamba_fusion_block = MambaFusionBlock(dim=embed_dim)

        # Classification Head
        self.cls_head = ClsHead(input_dim=embed_dim, num_classes=num_classes)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv3d):
        #         nn.init.kaiming_normal_(
        #             torch.as_tensor(m.weight), mode="fan_out", nonlinearity="relu"
        #         )
        #     elif isinstance(m, nn.Linear) and m.bias is not None:
        #         nn.init.constant_(torch.as_tensor(m.bias), 0)

    def forward(self, t1, t2, t1c):
        # Move tensors through ConvStem
        t1_features = self.conv_stem(t1)
        t2_features = self.conv_stem(t2)
        t1c_features = self.conv_stem(t1c)

        # Feature extraction
        t1_features = self.feature_extractor(t1_features)
        t2_features = self.feature_extractor(t2_features)
        t1c_features = self.feature_extractor(t1c_features)

        # Reshape features to (B, N, C) format for cross-attention
        batch_size, _, depth, height, width = t1_features.shape
        flattened_size = int(depth * height * width)
        t1_flat = t1_features.reshape(batch_size, flattened_size, -1)
        t2_flat = t2_features.reshape(batch_size, flattened_size, -1)
        t1c_flat = t1c_features.reshape(batch_size, flattened_size, -1)

        # Cross-modal attention
        t1_t2_attn_output = self.cross_modal_attn_block(t1_flat, context=t2_flat)
        t2_t1c_attn_output = self.cross_modal_attn_block(t2_flat, context=t1c_flat)
        t1c_t1_attn_output = self.cross_modal_attn_block(t1c_flat, context=t1_flat)

        # Concatenate cross-attention outputs for fusion
        combined_features = torch.cat(
            [t1_t2_attn_output, t2_t1c_attn_output, t1c_t1_attn_output], dim=1
        )

        # Mamba Fusion Block
        fused_features = self.mamba_fusion_block(combined_features)

        # Classification Head
        output = self.cls_head(fused_features)

        return output


if __name__ == "__main__":
    add_torch_shape_forvs()

    # Set parameters for the test
    batch_size = 2  # Number of samples in a batch
    in_channels = 1  # Single channel for MRI inputs (grayscale)
    depth, height, width = 256, 256, 32  # Example dimensions for 3D MRI scans
    num_classes = 2  # Number of output classes (e.g., tumor vs. non-tumor)
    stem_channels = 16

    # Create sample inputs for T1, T2, and T1C (simulating 3D MRI sequences)
    # Each tensor represents a batch of images with shape [batch_size, channels, depth, height, width]
    t1_sample = torch.randn(batch_size, in_channels, depth, height, width)
    t2_sample = torch.randn(batch_size, in_channels, depth, height, width)
    t1c_sample = torch.randn(batch_size, in_channels, depth, height, width)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t1_sample = t1_sample.to(device)
    t2_sample = t2_sample.to(device)
    t1c_sample = t1c_sample.to(device)

    conv_stem = ConvStem(in_channels=1, out_channels=stem_channels).to(device)
    conv_stem_output = conv_stem(t1_sample)

    feature_extractor = FeatureExtractor(in_channels=stem_channels).to(device)
    feature_extractor_output = feature_extractor(conv_stem_output)

    cross_modal_attn_block = CrossModalAttention(
        hidden_size=stem_channels * 2**4,
        num_heads=16,
        dropout_rate=0.1,
    ).to(device)
    # Reshape tensors to (B, N, C) format for cross-attention
    tensor1_flat = feature_extractor_output.reshape(
        batch_size,
        int(depth * height * width / 2**5 / 2**5 / 2**5),
        stem_channels * 2**4,
    )
    tensor2_flat = feature_extractor_output.reshape(
        batch_size,
        int(depth * height * width / 2**5 / 2**5 / 2**5),
        stem_channels * 2**4,
    )
    cross_modal_attn_block_output = cross_modal_attn_block(
        tensor1_flat, context=tensor2_flat
    )

    mamba_fusion_block = MambaFusionBlock(
        dim=cross_modal_attn_block_output.shape[-1]
    ).to(device)
    mamba_fusion_block_output = mamba_fusion_block(
        torch.repeat_interleave(cross_modal_attn_block_output, repeats=3, dim=1)
    )

    cls_head = ClsHead(
        input_dim=(stem_channels * 2**4),
        num_classes=2,
    ).to(device)
    cls_head_output = cls_head(mamba_fusion_block_output)

    # pooling = nn.AdaptiveAvgPool3d((1, 1, 1))
    # pooling_output = pooling(feature_extractor_output)
    # h_feature = torch.cat(
    #     (
    #         pooling_output.reshape(pooling_output.shape[0], pooling_output.shape[1], 1),
    #         pooling_output.reshape(pooling_output.shape[0], pooling_output.shape[1], 1),
    #     ),
    #     dim=2,
    # )
    # cls_head = ClsHead(
    #     input_dim=h_feature.shape[-1],
    #     num_classes=2,
    # ).to(device)
    # cls_head_output = cls_head(h_feature)

    # msf_encoder1 = MSFEncoder(in_channels=16, out_channels=32, scales=[3, 5, 7]).to(
    #     device
    # )
    # msf_encoder_output1 = msf_encoder1(conv_stem_output)

    # msf_encoder2 = MSFEncoder(in_channels=32, out_channels=64, scales=[3, 5, 7]).to(
    #     device
    # )
    # msf_encoder_output2 = msf_encoder2(msf_encoder_output1)

    # msf_encoder3 = MSFEncoder(in_channels=64, out_channels=128, scales=[3, 5, 7]).to(
    #     device
    # )
    # msf_encoder_output3 = msf_encoder3(msf_encoder_output2)

    # msf_encoder4 = MSFEncoder(in_channels=128, out_channels=256, scales=[3, 5, 7]).to(
    #     device
    # )
    # msf_encoder_output4 = msf_encoder4(msf_encoder_output3)

    # Instantiate the model
    model = MultiSeqMambaModel(
        in_channels=in_channels, num_classes=num_classes, stem_channels=stem_channels
    ).to(device)
    # Set model to evaluation mode (important for inference)
    model.eval()
    summary(
        model,
        input_size=[(2, 1, 256, 256, 32), (2, 1, 256, 256, 32), (2, 1, 256, 256, 32)],
    )
    # Forward pass through the model
    with torch.no_grad():  # Disable gradient computation for testing
        output = model(t1_sample, t2_sample, t1c_sample)

    # Display the output predictions
    print("Output predictions:", output)
    print("Predicted class labels:", torch.argmax(output, dim=1))
