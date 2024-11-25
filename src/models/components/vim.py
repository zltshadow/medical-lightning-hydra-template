# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
from functools import partial
from torch import Tensor
from typing import Optional
from timm.models.vision_transformer import _cfg
from timm.models import register_model
from timm.layers import trunc_normal_, lecun_normal_
from timm.layers import DropPath, to_2tuple, to_3tuple
from timm.models.vision_transformer import _load_weights
import math

from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
import random
from timm.models import create_model

# Copyright (c) 2023, Tri Dao, Albert Gu.

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, repeat
from torchinfo import summary
import yaml

from src.utils.utils import add_torch_shape_forvs

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None

# try:
#     from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn, bimamba_inner_fn, mamba_inner_fn_no_out_proj
# except ImportError:
#     selective_scan_fn, mamba_inner_fn, bimamba_inner_fn, mamba_inner_fn_no_out_proj = None, None, None, None, None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

# Copyright (c) 2023, Tri Dao, Albert Gu.

import torch
import torch.nn.functional as F

# from torch.cuda.amp import custom_bwd, custom_fwd
from torch.amp import custom_bwd, custom_fwd

custom_bwd = custom_bwd(device_type="cuda")
custom_fwd = custom_fwd(device_type="cuda")
from einops import rearrange, repeat
from causal_conv1d import causal_conv1d_fn
import causal_conv1d_cuda
import selective_scan_cuda


class SelectiveScanFn(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        u,
        delta,
        A,
        B,
        C,
        D=None,
        z=None,
        delta_bias=None,
        delta_softplus=False,
        return_last_state=False,
    ):
        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if D is not None:
            D = D.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if C.stride(-1) != 1:
            C = C.contiguous()
        if z is not None and z.stride(-1) != 1:
            z = z.contiguous()
        if B.dim() == 3:
            B = rearrange(B, "b dstate l -> b 1 dstate l")
            ctx.squeeze_B = True
        if C.dim() == 3:
            C = rearrange(C, "b dstate l -> b 1 dstate l")
            ctx.squeeze_C = True
        out, x, *rest = selective_scan_cuda.fwd(
            u, delta, A, B, C, D, z, delta_bias, delta_softplus
        )
        ctx.delta_softplus = delta_softplus
        ctx.has_z = z is not None
        last_state = x[:, :, -1, 1::2]  # (batch, dim, dstate)
        if not ctx.has_z:
            ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
            return out if not return_last_state else (out, last_state)
        else:
            ctx.save_for_backward(u, delta, A, B, C, D, z, delta_bias, x, out)
            out_z = rest[0]
            return out_z if not return_last_state else (out_z, last_state)

    @staticmethod
    def backward(ctx, dout, *args):
        if not ctx.has_z:
            u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
            z = None
            out = None
        else:
            u, delta, A, B, C, D, z, delta_bias, x, out = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        # The kernel supports passing in a pre-allocated dz (e.g., in case we want to fuse the
        # backward of selective_scan_cuda with the backward of chunk).
        # Here we just pass in None and dz will be allocated in the C++ code.
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
            u,
            delta,
            A,
            B,
            C,
            D,
            z,
            delta_bias,
            dout,
            x,
            out,
            None,
            ctx.delta_softplus,
            False,  # option to recompute out_z, not used here
        )
        dz = rest[0] if ctx.has_z else None
        dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
        dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
        return (
            du,
            ddelta,
            dA,
            dB,
            dC,
            dD if D is not None else None,
            dz,
            ddelta_bias if delta_bias is not None else None,
            None,
            None,
        )


def selective_scan_fn(
    u,
    delta,
    A,
    B,
    C,
    D=None,
    z=None,
    delta_bias=None,
    delta_softplus=False,
    return_last_state=False,
):
    """if return_last_state is True, returns (out, last_state)
    last_state has shape (batch, dim, dstate). Note that the gradient of the last state is
    not considered in the backward pass.
    """
    return SelectiveScanFn.apply(
        u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state
    )


def selective_scan_ref(
    u,
    delta,
    A,
    B,
    C,
    D=None,
    z=None,
    delta_bias=None,
    delta_softplus=False,
    return_last_state=False,
):
    """
    u: r(B D L)
    delta: r(B D L)
    A: c(D N) or r(D N)
    B: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
    C: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    out: r(B D L)
    last_state (optional): r(B D dstate) or c(B D dstate)
    """
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)
    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    is_variable_B = B.dim() >= 3
    is_variable_C = C.dim() >= 3
    if A.is_complex():
        if is_variable_B:
            B = torch.view_as_complex(
                rearrange(B.float(), "... (L two) -> ... L two", two=2)
            )
        if is_variable_C:
            C = torch.view_as_complex(
                rearrange(C.float(), "... (L two) -> ... L two", two=2)
            )
    else:
        B = B.float()
        C = C.float()
    x = A.new_zeros((batch, dim, dstate))
    ys = []
    deltaA = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))
    if not is_variable_B:
        deltaB_u = torch.einsum("bdl,dn,bdl->bdln", delta, B, u)
    else:
        if B.dim() == 3:
            deltaB_u = torch.einsum("bdl,bnl,bdl->bdln", delta, B, u)
        else:
            B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
            deltaB_u = torch.einsum("bdl,bdnl,bdl->bdln", delta, B, u)
    if is_variable_C and C.dim() == 4:
        C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
    last_state = None
    for i in range(u.shape[2]):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        if not is_variable_C:
            y = torch.einsum("bdn,dn->bd", x, C)
        else:
            if C.dim() == 3:
                y = torch.einsum("bdn,bn->bd", x, C[:, :, i])
            else:
                y = torch.einsum("bdn,bdn->bd", x, C[:, :, :, i])
        if i == u.shape[2] - 1:
            last_state = x
        if y.is_complex():
            y = y.real * 2
        ys.append(y)
    y = torch.stack(ys, dim=2)  # (batch dim L)
    out = y if D is None else y + u * rearrange(D, "d -> d 1")
    if z is not None:
        out = out * F.silu(z)
    out = out.to(dtype=dtype_in)
    return out if not return_last_state else (out, last_state)


class MambaInnerFnNoOutProj(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        xz,
        conv1d_weight,
        conv1d_bias,
        x_proj_weight,
        delta_proj_weight,
        A,
        B=None,
        C=None,
        D=None,
        delta_bias=None,
        B_proj_bias=None,
        C_proj_bias=None,
        delta_softplus=True,
        checkpoint_lvl=1,
    ):
        """
        xz: (batch, dim, seqlen)
        """
        assert checkpoint_lvl in [0, 1]
        L = xz.shape[-1]
        delta_rank = delta_proj_weight.shape[1]
        d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
        if torch.is_autocast_enabled():
            x_proj_weight = x_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
            delta_proj_weight = delta_proj_weight.to(
                dtype=torch.get_autocast_gpu_dtype()
            )
        if xz.stride(-1) != 1:
            xz = xz.contiguous()
        conv1d_weight = rearrange(conv1d_weight, "d 1 w -> d w")
        x, z = xz.chunk(2, dim=1)
        conv1d_bias = conv1d_bias.contiguous() if conv1d_bias is not None else None
        conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(
            x, conv1d_weight, conv1d_bias, None, None, None, True
        )
        # We're being very careful here about the layout, to avoid extra transposes.
        # We want delta to have d as the slowest moving dimension
        # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
        x_dbl = F.linear(
            rearrange(conv1d_out, "b d l -> (b l) d"), x_proj_weight
        )  # (bl d)
        delta = rearrange(
            delta_proj_weight @ x_dbl[:, :delta_rank].t(), "d (b l) -> b d l", l=L
        )
        ctx.is_variable_B = B is None
        ctx.is_variable_C = C is None
        ctx.B_proj_bias_is_None = B_proj_bias is None
        ctx.C_proj_bias_is_None = C_proj_bias is None
        if B is None:  # variable B
            B = x_dbl[:, delta_rank : delta_rank + d_state]  # (bl dstate)
            if B_proj_bias is not None:
                B = B + B_proj_bias.to(dtype=B.dtype)
            if not A.is_complex():
                # B = rearrange(B, "(b l) dstate -> b dstate l", l=L).contiguous()
                B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
            else:
                B = rearrange(
                    B, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2
                ).contiguous()
        else:
            if B.stride(-1) != 1:
                B = B.contiguous()
        if C is None:  # variable C
            C = x_dbl[:, -d_state:]  # (bl dstate)
            if C_proj_bias is not None:
                C = C + C_proj_bias.to(dtype=C.dtype)
            if not A.is_complex():
                # C = rearrange(C, "(b l) dstate -> b dstate l", l=L).contiguous()
                C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
            else:
                C = rearrange(
                    C, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2
                ).contiguous()
        else:
            if C.stride(-1) != 1:
                C = C.contiguous()
        if D is not None:
            D = D.contiguous()
        out, scan_intermediates, out_z = selective_scan_cuda.fwd(
            conv1d_out, delta, A, B, C, D, z, delta_bias, delta_softplus
        )
        ctx.delta_softplus = delta_softplus
        ctx.checkpoint_lvl = checkpoint_lvl
        if (
            checkpoint_lvl >= 1
        ):  # Will recompute conv1d_out and delta in the backward pass
            conv1d_out, delta = None, None
        ctx.save_for_backward(
            xz,
            conv1d_weight,
            conv1d_bias,
            x_dbl,
            x_proj_weight,
            delta_proj_weight,
            conv1d_out,
            delta,
            A,
            B,
            C,
            D,
            delta_bias,
            scan_intermediates,
            out,
        )
        # return rearrange(out_z, "b d l -> b l d")
        return out_z

    @staticmethod
    @custom_bwd
    def backward(ctx, dout):
        # dout: (batch, seqlen, dim)
        (
            xz,
            conv1d_weight,
            conv1d_bias,
            x_dbl,
            x_proj_weight,
            delta_proj_weight,
            conv1d_out,
            delta,
            A,
            B,
            C,
            D,
            delta_bias,
            scan_intermediates,
            out,
        ) = ctx.saved_tensors
        L = xz.shape[-1]
        delta_rank = delta_proj_weight.shape[1]
        d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
        x, z = xz.chunk(2, dim=1)
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        if ctx.checkpoint_lvl == 1:
            conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(
                x, conv1d_weight, conv1d_bias, None, None, None, True
            )
            delta = rearrange(
                delta_proj_weight @ x_dbl[:, :delta_rank].t(), "d (b l) -> b d l", l=L
            )
        # The kernel supports passing in a pre-allocated dz (e.g., in case we want to fuse the
        # backward of selective_scan_cuda with the backward of chunk).
        dxz = torch.empty_like(xz)  # (batch, dim, seqlen)
        dx, dz = dxz.chunk(2, dim=1)
        # dout_y = rearrange(dout, "b l d -> b d l") # because no arrange at end of forward, so dout shape is b d l
        dconv1d_out, ddelta, dA, dB, dC, dD, ddelta_bias, dz, out_z = (
            selective_scan_cuda.bwd(
                conv1d_out,
                delta,
                A,
                B,
                C,
                D,
                z,
                delta_bias,
                dout,
                scan_intermediates,
                out,
                dz,
                ctx.delta_softplus,
                True,  # option to recompute out_z
            )
        )
        dD = dD if D is not None else None
        dx_dbl = torch.empty_like(x_dbl)
        dB_proj_bias = None
        if ctx.is_variable_B:
            if not A.is_complex():
                dB = rearrange(dB, "b 1 dstate l -> (b l) dstate").contiguous()
            else:
                dB = rearrange(
                    dB, "b 1 dstate (l two) -> (b l) (dstate two)", two=2
                ).contiguous()
            dB_proj_bias = dB.sum(0) if not ctx.B_proj_bias_is_None else None
            dx_dbl[:, delta_rank : delta_rank + d_state] = dB  # (bl d)
            dB = None
        dC_proj_bias = None
        if ctx.is_variable_C:
            if not A.is_complex():
                dC = rearrange(dC, "b 1 dstate l -> (b l) dstate").contiguous()
            else:
                dC = rearrange(
                    dC, "b 1 dstate (l two) -> (b l) (dstate two)", two=2
                ).contiguous()
            dC_proj_bias = dC.sum(0) if not ctx.C_proj_bias_is_None else None
            dx_dbl[:, -d_state:] = dC  # (bl d)
            dC = None
        ddelta = rearrange(ddelta, "b d l -> d (b l)")
        ddelta_proj_weight = torch.einsum("dB,Br->dr", ddelta, x_dbl[:, :delta_rank])
        dx_dbl[:, :delta_rank] = torch.einsum("dB,dr->Br", ddelta, delta_proj_weight)
        dconv1d_out = rearrange(dconv1d_out, "b d l -> d (b l)")
        dx_proj_weight = torch.einsum(
            "Br,Bd->rd", dx_dbl, rearrange(conv1d_out, "b d l -> (b l) d")
        )
        dconv1d_out = torch.addmm(
            dconv1d_out, x_proj_weight.t(), dx_dbl.t(), out=dconv1d_out
        )
        dconv1d_out = rearrange(
            dconv1d_out, "d (b l) -> b d l", b=x.shape[0], l=x.shape[-1]
        )
        # The kernel supports passing in a pre-allocated dx (e.g., in case we want to fuse the
        # backward of conv1d with the backward of chunk).
        dx, dconv1d_weight, dconv1d_bias, _ = causal_conv1d_cuda.causal_conv1d_bwd(
            x, conv1d_weight, conv1d_bias, dconv1d_out, None, None, None, dx, True, True
        )
        dconv1d_bias = dconv1d_bias if conv1d_bias is not None else None
        dconv1d_weight = rearrange(dconv1d_weight, "d w -> d 1 w")
        return (
            dxz,
            dconv1d_weight,
            dconv1d_bias,
            dx_proj_weight,
            ddelta_proj_weight,
            dA,
            dB,
            dC,
            dD,
            ddelta_bias if delta_bias is not None else None,
            dB_proj_bias,
            dC_proj_bias,
            None,
        )


class MambaInnerFn(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        xz,
        conv1d_weight,
        conv1d_bias,
        x_proj_weight,
        delta_proj_weight,
        out_proj_weight,
        out_proj_bias,
        A,
        B=None,
        C=None,
        D=None,
        delta_bias=None,
        B_proj_bias=None,
        C_proj_bias=None,
        delta_softplus=True,
        checkpoint_lvl=1,
    ):
        """
        xz: (batch, dim, seqlen)
        """
        assert checkpoint_lvl in [0, 1]
        L = xz.shape[-1]
        delta_rank = delta_proj_weight.shape[1]
        d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
        if torch.is_autocast_enabled():
            x_proj_weight = x_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
            delta_proj_weight = delta_proj_weight.to(
                dtype=torch.get_autocast_gpu_dtype()
            )
            out_proj_weight = out_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
            out_proj_bias = (
                out_proj_bias.to(dtype=torch.get_autocast_gpu_dtype())
                if out_proj_bias is not None
                else None
            )
        if xz.stride(-1) != 1:
            xz = xz.contiguous()
        conv1d_weight = rearrange(conv1d_weight, "d 1 w -> d w")
        x, z = xz.chunk(2, dim=1)
        conv1d_bias = conv1d_bias.contiguous() if conv1d_bias is not None else None
        conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(
            x, conv1d_weight, conv1d_bias, None, None, None, True
        )
        # We're being very careful here about the layout, to avoid extra transposes.
        # We want delta to have d as the slowest moving dimension
        # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
        x_dbl = F.linear(
            rearrange(conv1d_out, "b d l -> (b l) d"), x_proj_weight
        )  # (bl d)
        delta = rearrange(
            delta_proj_weight @ x_dbl[:, :delta_rank].t(), "d (b l) -> b d l", l=L
        )
        ctx.is_variable_B = B is None
        ctx.is_variable_C = C is None
        ctx.B_proj_bias_is_None = B_proj_bias is None
        ctx.C_proj_bias_is_None = C_proj_bias is None
        if B is None:  # variable B
            B = x_dbl[:, delta_rank : delta_rank + d_state]  # (bl dstate)
            if B_proj_bias is not None:
                B = B + B_proj_bias.to(dtype=B.dtype)
            if not A.is_complex():
                # B = rearrange(B, "(b l) dstate -> b dstate l", l=L).contiguous()
                B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
            else:
                B = rearrange(
                    B, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2
                ).contiguous()
        else:
            if B.stride(-1) != 1:
                B = B.contiguous()
        if C is None:  # variable C
            C = x_dbl[:, -d_state:]  # (bl dstate)
            if C_proj_bias is not None:
                C = C + C_proj_bias.to(dtype=C.dtype)
            if not A.is_complex():
                # C = rearrange(C, "(b l) dstate -> b dstate l", l=L).contiguous()
                C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
            else:
                C = rearrange(
                    C, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2
                ).contiguous()
        else:
            if C.stride(-1) != 1:
                C = C.contiguous()
        if D is not None:
            D = D.contiguous()
        out, scan_intermediates, out_z = selective_scan_cuda.fwd(
            conv1d_out, delta, A, B, C, D, z, delta_bias, delta_softplus
        )
        ctx.delta_softplus = delta_softplus
        ctx.out_proj_bias_is_None = out_proj_bias is None
        ctx.checkpoint_lvl = checkpoint_lvl
        if (
            checkpoint_lvl >= 1
        ):  # Will recompute conv1d_out and delta in the backward pass
            conv1d_out, delta = None, None
        ctx.save_for_backward(
            xz,
            conv1d_weight,
            conv1d_bias,
            x_dbl,
            x_proj_weight,
            delta_proj_weight,
            out_proj_weight,
            conv1d_out,
            delta,
            A,
            B,
            C,
            D,
            delta_bias,
            scan_intermediates,
            out,
        )
        return F.linear(
            rearrange(out_z, "b d l -> b l d"), out_proj_weight, out_proj_bias
        )

    @staticmethod
    @custom_bwd
    def backward(ctx, dout):
        # dout: (batch, seqlen, dim)
        (
            xz,
            conv1d_weight,
            conv1d_bias,
            x_dbl,
            x_proj_weight,
            delta_proj_weight,
            out_proj_weight,
            conv1d_out,
            delta,
            A,
            B,
            C,
            D,
            delta_bias,
            scan_intermediates,
            out,
        ) = ctx.saved_tensors
        L = xz.shape[-1]
        delta_rank = delta_proj_weight.shape[1]
        d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
        x, z = xz.chunk(2, dim=1)
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        if ctx.checkpoint_lvl == 1:
            conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(
                x, conv1d_weight, conv1d_bias, None, None, None, True
            )
            delta = rearrange(
                delta_proj_weight @ x_dbl[:, :delta_rank].t(), "d (b l) -> b d l", l=L
            )
        # The kernel supports passing in a pre-allocated dz (e.g., in case we want to fuse the
        # backward of selective_scan_cuda with the backward of chunk).
        dxz = torch.empty_like(xz)  # (batch, dim, seqlen)
        dx, dz = dxz.chunk(2, dim=1)
        dout = rearrange(dout, "b l e -> e (b l)")
        dout_y = rearrange(out_proj_weight.t() @ dout, "d (b l) -> b d l", l=L)
        dconv1d_out, ddelta, dA, dB, dC, dD, ddelta_bias, dz, out_z = (
            selective_scan_cuda.bwd(
                conv1d_out,
                delta,
                A,
                B,
                C,
                D,
                z,
                delta_bias,
                dout_y,
                scan_intermediates,
                out,
                dz,
                ctx.delta_softplus,
                True,  # option to recompute out_z
            )
        )
        dout_proj_weight = torch.einsum(
            "eB,dB->ed", dout, rearrange(out_z, "b d l -> d (b l)")
        )
        dout_proj_bias = dout.sum(dim=(0, 1)) if not ctx.out_proj_bias_is_None else None
        dD = dD if D is not None else None
        dx_dbl = torch.empty_like(x_dbl)
        dB_proj_bias = None
        if ctx.is_variable_B:
            if not A.is_complex():
                dB = rearrange(dB, "b 1 dstate l -> (b l) dstate").contiguous()
            else:
                dB = rearrange(
                    dB, "b 1 dstate (l two) -> (b l) (dstate two)", two=2
                ).contiguous()
            dB_proj_bias = dB.sum(0) if not ctx.B_proj_bias_is_None else None
            dx_dbl[:, delta_rank : delta_rank + d_state] = dB  # (bl d)
            dB = None
        dC_proj_bias = None
        if ctx.is_variable_C:
            if not A.is_complex():
                dC = rearrange(dC, "b 1 dstate l -> (b l) dstate").contiguous()
            else:
                dC = rearrange(
                    dC, "b 1 dstate (l two) -> (b l) (dstate two)", two=2
                ).contiguous()
            dC_proj_bias = dC.sum(0) if not ctx.C_proj_bias_is_None else None
            dx_dbl[:, -d_state:] = dC  # (bl d)
            dC = None
        ddelta = rearrange(ddelta, "b d l -> d (b l)")
        ddelta_proj_weight = torch.einsum("dB,Br->dr", ddelta, x_dbl[:, :delta_rank])
        dx_dbl[:, :delta_rank] = torch.einsum("dB,dr->Br", ddelta, delta_proj_weight)
        dconv1d_out = rearrange(dconv1d_out, "b d l -> d (b l)")
        dx_proj_weight = torch.einsum(
            "Br,Bd->rd", dx_dbl, rearrange(conv1d_out, "b d l -> (b l) d")
        )
        dconv1d_out = torch.addmm(
            dconv1d_out, x_proj_weight.t(), dx_dbl.t(), out=dconv1d_out
        )
        dconv1d_out = rearrange(
            dconv1d_out, "d (b l) -> b d l", b=x.shape[0], l=x.shape[-1]
        )
        # The kernel supports passing in a pre-allocated dx (e.g., in case we want to fuse the
        # backward of conv1d with the backward of chunk).
        dx, dconv1d_weight, dconv1d_bias, _ = causal_conv1d_cuda.causal_conv1d_bwd(
            x, conv1d_weight, conv1d_bias, dconv1d_out, None, None, None, dx, True, True
        )
        dconv1d_bias = dconv1d_bias if conv1d_bias is not None else None
        dconv1d_weight = rearrange(dconv1d_weight, "d w -> d 1 w")
        return (
            dxz,
            dconv1d_weight,
            dconv1d_bias,
            dx_proj_weight,
            ddelta_proj_weight,
            dout_proj_weight,
            dout_proj_bias,
            dA,
            dB,
            dC,
            dD,
            ddelta_bias if delta_bias is not None else None,
            dB_proj_bias,
            dC_proj_bias,
            None,
        )


class BiMambaInnerFn(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        xz,
        conv1d_weight,
        conv1d_bias,
        x_proj_weight,
        delta_proj_weight,
        out_proj_weight,
        out_proj_bias,
        A,
        A_b,
        B=None,
        C=None,
        D=None,
        delta_bias=None,
        B_proj_bias=None,
        C_proj_bias=None,
        delta_softplus=True,
        checkpoint_lvl=1,
    ):
        """
        xz: (batch, dim, seqlen)
        """
        assert checkpoint_lvl in [0, 1]
        L = xz.shape[-1]
        delta_rank = delta_proj_weight.shape[1]
        d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
        if torch.is_autocast_enabled():
            x_proj_weight = x_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
            delta_proj_weight = delta_proj_weight.to(
                dtype=torch.get_autocast_gpu_dtype()
            )
            out_proj_weight = out_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
            out_proj_bias = (
                out_proj_bias.to(dtype=torch.get_autocast_gpu_dtype())
                if out_proj_bias is not None
                else None
            )
        if xz.stride(-1) != 1:
            xz = xz.contiguous()
        conv1d_weight = rearrange(conv1d_weight, "d 1 w -> d w")
        x, z = xz.chunk(2, dim=1)
        conv1d_bias = conv1d_bias.contiguous() if conv1d_bias is not None else None
        conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(
            x, conv1d_weight, conv1d_bias, None, None, None, True
        )
        # We're being very careful here about the layout, to avoid extra transposes.
        # We want delta to have d as the slowest moving dimension
        # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
        x_dbl = F.linear(
            rearrange(conv1d_out, "b d l -> (b l) d"), x_proj_weight
        )  # (bl d)
        delta = rearrange(
            delta_proj_weight @ x_dbl[:, :delta_rank].t(), "d (b l) -> b d l", l=L
        )
        ctx.is_variable_B = B is None
        ctx.is_variable_C = C is None
        ctx.B_proj_bias_is_None = B_proj_bias is None
        ctx.C_proj_bias_is_None = C_proj_bias is None
        if B is None:  # variable B
            B = x_dbl[:, delta_rank : delta_rank + d_state]  # (bl dstate)
            if B_proj_bias is not None:
                B = B + B_proj_bias.to(dtype=B.dtype)
            if not A.is_complex():
                # B = rearrange(B, "(b l) dstate -> b dstate l", l=L).contiguous()
                B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
            else:
                B = rearrange(
                    B, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2
                ).contiguous()
        else:
            if B.stride(-1) != 1:
                B = B.contiguous()
        if C is None:  # variable C
            C = x_dbl[:, -d_state:]  # (bl dstate)
            if C_proj_bias is not None:
                C = C + C_proj_bias.to(dtype=C.dtype)
            if not A.is_complex():
                # C = rearrange(C, "(b l) dstate -> b dstate l", l=L).contiguous()
                C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
            else:
                C = rearrange(
                    C, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2
                ).contiguous()
        else:
            if C.stride(-1) != 1:
                C = C.contiguous()
        if D is not None:
            D = D.contiguous()
        out_f, scan_intermediates_f, out_z_f = selective_scan_cuda.fwd(
            conv1d_out, delta, A, B, C, D, z, delta_bias, delta_softplus
        )
        assert not A_b.is_complex(), "A should not be complex!!"
        out_b, scan_intermediates_b, out_z_b = selective_scan_cuda.fwd(
            conv1d_out.flip([-1]),
            delta.flip([-1]),
            A_b,
            B.flip([-1]),
            C.flip([-1]),
            D,
            z.flip([-1]),
            delta_bias,
            delta_softplus,
        )

        out_z = out_z_f + out_z_b.flip([-1])

        ctx.delta_softplus = delta_softplus
        ctx.out_proj_bias_is_None = out_proj_bias is None
        ctx.checkpoint_lvl = checkpoint_lvl
        if (
            checkpoint_lvl >= 1
        ):  # Will recompute conv1d_out and delta in the backward pass
            conv1d_out, delta = None, None
        ctx.save_for_backward(
            xz,
            conv1d_weight,
            conv1d_bias,
            x_dbl,
            x_proj_weight,
            delta_proj_weight,
            out_proj_weight,
            conv1d_out,
            delta,
            A,
            A_b,
            B,
            C,
            D,
            delta_bias,
            scan_intermediates_f,
            scan_intermediates_b,
            out_f,
            out_b,
        )
        return F.linear(
            rearrange(out_z, "b d l -> b l d"), out_proj_weight, out_proj_bias
        )

    @staticmethod
    @custom_bwd
    def backward(ctx, dout):
        # dout: (batch, seqlen, dim)
        (
            xz,
            conv1d_weight,
            conv1d_bias,
            x_dbl,
            x_proj_weight,
            delta_proj_weight,
            out_proj_weight,
            conv1d_out,
            delta,
            A,
            A_b,
            B,
            C,
            D,
            delta_bias,
            scan_intermediates_f,
            scan_intermediates_b,
            out_f,
            out_b,
        ) = ctx.saved_tensors
        L = xz.shape[-1]
        delta_rank = delta_proj_weight.shape[1]
        d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
        x, z = xz.chunk(2, dim=1)
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        if ctx.checkpoint_lvl == 1:
            conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(
                x, conv1d_weight, conv1d_bias, None, None, None, True
            )
            delta = rearrange(
                delta_proj_weight @ x_dbl[:, :delta_rank].t(), "d (b l) -> b d l", l=L
            )
        # The kernel supports passing in a pre-allocated dz (e.g., in case we want to fuse the
        # backward of selective_scan_cuda with the backward of chunk).
        dxz = torch.empty_like(xz)  # (batch, dim, seqlen)
        dx, dz = dxz.chunk(2, dim=1)
        dout = rearrange(dout, "b l e -> e (b l)")
        dout_y = rearrange(out_proj_weight.t() @ dout, "d (b l) -> b d l", l=L)
        dconv1d_out, ddelta, dA, dB, dC, dD, ddelta_bias, dz, out_z_f = (
            selective_scan_cuda.bwd(
                conv1d_out,
                delta,
                A,
                B,
                C,
                D,
                z,
                delta_bias,
                dout_y,
                scan_intermediates_f,
                out_f,
                dz,
                ctx.delta_softplus,
                True,  # option to recompute out_z
            )
        )
        # flip one
        dz_b = torch.empty_like(dz)
        (
            dconv1d_out_f_b,
            ddelta_f_b,
            dA_b,
            dB_f_b,
            dC_f_b,
            dD_b,
            ddelta_bias_b,
            dz_b,
            out_z_b,
        ) = selective_scan_cuda.bwd(
            conv1d_out.flip([-1]),
            delta.flip([-1]),
            A_b,
            B.flip([-1]),
            C.flip([-1]),
            D,
            z.flip([-1]),
            delta_bias,
            dout_y.flip([-1]),
            scan_intermediates_b,
            out_b,
            dz_b,
            ctx.delta_softplus,
            True,  # option to recompute out_z
        )

        dconv1d_out = dconv1d_out + dconv1d_out_f_b.flip([-1])
        ddelta = ddelta + ddelta_f_b.flip([-1])
        dB = dB + dB_f_b.flip([-1])
        dC = dC + dC_f_b.flip([-1])
        dD = dD + dD_b
        ddelta_bias = ddelta_bias + ddelta_bias_b
        dz = dz + dz_b.flip([-1])
        out_z = out_z_f + out_z_b.flip([-1])

        dout_proj_weight = torch.einsum(
            "eB,dB->ed", dout, rearrange(out_z, "b d l -> d (b l)")
        )
        dout_proj_bias = dout.sum(dim=(0, 1)) if not ctx.out_proj_bias_is_None else None
        dD = dD if D is not None else None
        dx_dbl = torch.empty_like(x_dbl)
        dB_proj_bias = None
        if ctx.is_variable_B:
            if not A.is_complex():
                dB = rearrange(dB, "b 1 dstate l -> (b l) dstate").contiguous()
            else:
                dB = rearrange(
                    dB, "b 1 dstate (l two) -> (b l) (dstate two)", two=2
                ).contiguous()
            dB_proj_bias = dB.sum(0) if not ctx.B_proj_bias_is_None else None
            dx_dbl[:, delta_rank : delta_rank + d_state] = dB  # (bl d)
            dB = None
        dC_proj_bias = None
        if ctx.is_variable_C:
            if not A.is_complex():
                dC = rearrange(dC, "b 1 dstate l -> (b l) dstate").contiguous()
            else:
                dC = rearrange(
                    dC, "b 1 dstate (l two) -> (b l) (dstate two)", two=2
                ).contiguous()
            dC_proj_bias = dC.sum(0) if not ctx.C_proj_bias_is_None else None
            dx_dbl[:, -d_state:] = dC  # (bl d)
            dC = None
        ddelta = rearrange(ddelta, "b d l -> d (b l)")
        ddelta_proj_weight = torch.einsum("dB,Br->dr", ddelta, x_dbl[:, :delta_rank])
        dx_dbl[:, :delta_rank] = torch.einsum("dB,dr->Br", ddelta, delta_proj_weight)
        dconv1d_out = rearrange(dconv1d_out, "b d l -> d (b l)")
        dx_proj_weight = torch.einsum(
            "Br,Bd->rd", dx_dbl, rearrange(conv1d_out, "b d l -> (b l) d")
        )
        dconv1d_out = torch.addmm(
            dconv1d_out, x_proj_weight.t(), dx_dbl.t(), out=dconv1d_out
        )
        dconv1d_out = rearrange(
            dconv1d_out, "d (b l) -> b d l", b=x.shape[0], l=x.shape[-1]
        )
        # The kernel supports passing in a pre-allocated dx (e.g., in case we want to fuse the
        # backward of conv1d with the backward of chunk).
        dx, dconv1d_weight, dconv1d_bias, _ = causal_conv1d_cuda.causal_conv1d_bwd(
            x, conv1d_weight, conv1d_bias, dconv1d_out, None, None, None, dx, True, True
        )
        dconv1d_bias = dconv1d_bias if conv1d_bias is not None else None
        dconv1d_weight = rearrange(dconv1d_weight, "d w -> d 1 w")
        return (
            dxz,
            dconv1d_weight,
            dconv1d_bias,
            dx_proj_weight,
            ddelta_proj_weight,
            dout_proj_weight,
            dout_proj_bias,
            dA,
            dA_b,
            dB,
            dC,
            dD,
            ddelta_bias if delta_bias is not None else None,
            dB_proj_bias,
            dC_proj_bias,
            None,
        )


def mamba_inner_fn(
    xz,
    conv1d_weight,
    conv1d_bias,
    x_proj_weight,
    delta_proj_weight,
    out_proj_weight,
    out_proj_bias,
    A,
    B=None,
    C=None,
    D=None,
    delta_bias=None,
    B_proj_bias=None,
    C_proj_bias=None,
    delta_softplus=True,
):
    return MambaInnerFn.apply(
        xz,
        conv1d_weight,
        conv1d_bias,
        x_proj_weight,
        delta_proj_weight,
        out_proj_weight,
        out_proj_bias,
        A,
        B,
        C,
        D,
        delta_bias,
        B_proj_bias,
        C_proj_bias,
        delta_softplus,
    )


def bimamba_inner_fn(
    xz,
    conv1d_weight,
    conv1d_bias,
    x_proj_weight,
    delta_proj_weight,
    out_proj_weight,
    out_proj_bias,
    A,
    A_b,
    B=None,
    C=None,
    D=None,
    delta_bias=None,
    B_proj_bias=None,
    C_proj_bias=None,
    delta_softplus=True,
):
    return BiMambaInnerFn.apply(
        xz,
        conv1d_weight,
        conv1d_bias,
        x_proj_weight,
        delta_proj_weight,
        out_proj_weight,
        out_proj_bias,
        A,
        A_b,
        B,
        C,
        D,
        delta_bias,
        B_proj_bias,
        C_proj_bias,
        delta_softplus,
    )


def mamba_inner_fn_no_out_proj(
    xz,
    conv1d_weight,
    conv1d_bias,
    x_proj_weight,
    delta_proj_weight,
    A,
    B=None,
    C=None,
    D=None,
    delta_bias=None,
    B_proj_bias=None,
    C_proj_bias=None,
    delta_softplus=True,
):
    return MambaInnerFnNoOutProj.apply(
        xz,
        conv1d_weight,
        conv1d_bias,
        x_proj_weight,
        delta_proj_weight,
        A,
        B,
        C,
        D,
        delta_bias,
        B_proj_bias,
        C_proj_bias,
        delta_softplus,
    )


def mamba_inner_ref(
    xz,
    conv1d_weight,
    conv1d_bias,
    x_proj_weight,
    delta_proj_weight,
    out_proj_weight,
    out_proj_bias,
    A,
    B=None,
    C=None,
    D=None,
    delta_bias=None,
    B_proj_bias=None,
    C_proj_bias=None,
    delta_softplus=True,
):
    L = xz.shape[-1]
    delta_rank = delta_proj_weight.shape[1]
    d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
    x, z = xz.chunk(2, dim=1)
    x = causal_conv1d_fn(
        x, rearrange(conv1d_weight, "d 1 w -> d w"), conv1d_bias, "silu"
    )
    # We're being very careful here about the layout, to avoid extra transposes.
    # We want delta to have d as the slowest moving dimension
    # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
    x_dbl = F.linear(rearrange(x, "b d l -> (b l) d"), x_proj_weight)  # (bl d)
    delta = delta_proj_weight @ x_dbl[:, :delta_rank].t()
    delta = rearrange(delta, "d (b l) -> b d l", l=L)
    if B is None:  # variable B
        B = x_dbl[:, delta_rank : delta_rank + d_state]  # (bl d)
        if B_proj_bias is not None:
            B = B + B_proj_bias.to(dtype=B.dtype)
        if not A.is_complex():
            B = rearrange(B, "(b l) dstate -> b dstate l", l=L).contiguous()
        else:
            B = rearrange(
                B, "(b l) (dstate two) -> b dstate (l two)", l=L, two=2
            ).contiguous()
    if C is None:  # variable B
        C = x_dbl[:, -d_state:]  # (bl d)
        if C_proj_bias is not None:
            C = C + C_proj_bias.to(dtype=C.dtype)
        if not A.is_complex():
            C = rearrange(C, "(b l) dstate -> b dstate l", l=L).contiguous()
        else:
            C = rearrange(
                C, "(b l) (dstate two) -> b dstate (l two)", l=L, two=2
            ).contiguous()
    y = selective_scan_fn(
        x, delta, A, B, C, D, z=z, delta_bias=delta_bias, delta_softplus=True
    )
    return F.linear(rearrange(y, "b d l -> b l d"), out_proj_weight, out_proj_bias)


def bimamba_inner_ref(
    xz,
    conv1d_weight,
    conv1d_bias,
    x_proj_weight,
    delta_proj_weight,
    out_proj_weight,
    out_proj_bias,
    A,
    A_b,
    B=None,
    C=None,
    D=None,
    delta_bias=None,
    B_proj_bias=None,
    C_proj_bias=None,
    delta_softplus=True,
):
    L = xz.shape[-1]
    delta_rank = delta_proj_weight.shape[1]
    d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
    x, z = xz.chunk(2, dim=1)
    x = causal_conv1d_fn(
        x, rearrange(conv1d_weight, "d 1 w -> d w"), conv1d_bias, "silu"
    )
    # We're being very careful here about the layout, to avoid extra transposes.
    # We want delta to have d as the slowest moving dimension
    # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
    x_dbl = F.linear(rearrange(x, "b d l -> (b l) d"), x_proj_weight)  # (bl d)
    delta = delta_proj_weight @ x_dbl[:, :delta_rank].t()
    delta = rearrange(delta, "d (b l) -> b d l", l=L)
    if B is None:  # variable B
        B = x_dbl[:, delta_rank : delta_rank + d_state]  # (bl d)
        if B_proj_bias is not None:
            B = B + B_proj_bias.to(dtype=B.dtype)
        if not A.is_complex():
            B = rearrange(B, "(b l) dstate -> b dstate l", l=L).contiguous()
        else:
            B = rearrange(
                B, "(b l) (dstate two) -> b dstate (l two)", l=L, two=2
            ).contiguous()
    if C is None:  # variable B
        C = x_dbl[:, -d_state:]  # (bl d)
        if C_proj_bias is not None:
            C = C + C_proj_bias.to(dtype=C.dtype)
        if not A.is_complex():
            C = rearrange(C, "(b l) dstate -> b dstate l", l=L).contiguous()
        else:
            C = rearrange(
                C, "(b l) (dstate two) -> b dstate (l two)", l=L, two=2
            ).contiguous()
    y = selective_scan_fn(
        x, delta, A, B, C, D, z=z, delta_bias=delta_bias, delta_softplus=True
    )
    y_b = selective_scan_fn(
        x.flip([-1]),
        delta.flip([-1]),
        A_b,
        B.flip([-1]),
        C.flip([-1]),
        D,
        z.flip([-1]),
        delta_bias,
        delta_softplus=True,
    )
    y = y + y_b.flip([-1])
    return F.linear(rearrange(y, "b d l -> b l d"), out_proj_weight, out_proj_bias)


class Mamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
        bimamba_type="none",
        if_divide_out=False,
        init_layer_scale=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.bimamba_type = bimamba_type
        self.if_divide_out = if_divide_out

        self.init_layer_scale = init_layer_scale
        if init_layer_scale is not None:
            self.gamma = nn.Parameter(
                init_layer_scale * torch.ones((d_model)), requires_grad=True
            )

        self.in_proj = nn.Linear(
            self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs
        )

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(
            self.dt_rank, self.d_inner, bias=True, **factory_kwargs
        )

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        # bidirectional
        if bimamba_type == "v1":
            A_b = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=self.d_inner,
            ).contiguous()
            A_b_log = torch.log(A_b)  # Keep A_b_log in fp32
            self.A_b_log = nn.Parameter(A_b_log)
            self.A_b_log._no_weight_decay = True
        elif bimamba_type == "v2":
            A_b = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=self.d_inner,
            ).contiguous()
            A_b_log = torch.log(A_b)  # Keep A_b_log in fp32
            self.A_b_log = nn.Parameter(A_b_log)
            self.A_b_log._no_weight_decay = True

            self.conv1d_b = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                **factory_kwargs,
            )

            self.x_proj_b = nn.Linear(
                self.d_inner,
                self.dt_rank + self.d_state * 2,
                bias=False,
                **factory_kwargs,
            )
            self.dt_proj_b = nn.Linear(
                self.dt_rank, self.d_inner, bias=True, **factory_kwargs
            )

            self.D_b = nn.Parameter(
                torch.ones(self.d_inner, device=device)
            )  # Keep in fp32
            self.D_b._no_weight_decay = True

        self.out_proj = nn.Linear(
            self.d_inner, self.d_model, bias=bias, **factory_kwargs
        )

    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        if (
            self.use_fast_path and inference_params is None
        ):  # Doesn't support outputting the states
            if self.bimamba_type == "v1":
                A_b = -torch.exp(self.A_b_log.float())
                out = bimamba_inner_fn(
                    xz,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    self.out_proj.weight,
                    self.out_proj.bias,
                    A,
                    A_b,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                )
            elif self.bimamba_type == "v2":
                A_b = -torch.exp(self.A_b_log.float())
                out = mamba_inner_fn_no_out_proj(
                    xz,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    A,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                )
                out_b = mamba_inner_fn_no_out_proj(
                    xz.flip([-1]),
                    self.conv1d_b.weight,
                    self.conv1d_b.bias,
                    self.x_proj_b.weight,
                    self.dt_proj_b.weight,
                    A_b,
                    None,
                    None,
                    self.D_b.float(),
                    delta_bias=self.dt_proj_b.bias.float(),
                    delta_softplus=True,
                )
                # F.linear(rearrange(out_z, "b d l -> b l d"), out_proj_weight, out_proj_bias)
                if not self.if_divide_out:
                    out = F.linear(
                        rearrange(out + out_b.flip([-1]), "b d l -> b l d"),
                        self.out_proj.weight,
                        self.out_proj.bias,
                    )
                else:
                    out = F.linear(
                        rearrange(out + out_b.flip([-1]), "b d l -> b l d") / 2,
                        self.out_proj.weight,
                        self.out_proj.bias,
                    )

            else:
                out = mamba_inner_fn(
                    xz,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    self.out_proj.weight,
                    self.out_proj.bias,
                    A,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                )
        else:
            x, z = xz.chunk(2, dim=1)
            # Compute short convolution
            if conv_state is not None:
                # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                conv_state.copy_(
                    F.pad(x, (self.d_conv - x.shape[-1], 0))
                )  # Update state (B D W)
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x=x,
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                )

            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
            dt, B, C = torch.split(
                x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
            )
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )
            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)
            y = rearrange(y, "b d l -> b l d")
            out = self.out_proj(y)
        if self.init_layer_scale is not None:
            out = out * self.gamma
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert (
            hidden_states.shape[1] == 1
        ), "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(
                torch.roll(conv_state, shifts=-1, dims=-1)
            )  # Update state (B D W)
            conv_state[:, :, -1] = x
            x = torch.sum(
                conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1
            )  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        if selective_state_update is None:
            # Discretize A and B
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)  # (B D)
        else:
            y = selective_state_update(
                ssm_state,
                x,
                dt,
                A,
                B,
                C,
                self.D,
                z=z,
                dt_bias=self.dt_proj.bias,
                dt_softplus=True,
            )

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size,
            self.d_model * self.expand,
            self.d_conv,
            device=device,
            dtype=conv_dtype,
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size,
            self.d_model * self.expand,
            self.d_state,
            device=device,
            dtype=ssm_dtype,
        )
        return conv_state, ssm_state

    def _get_states_from_cache(
        self, inference_params, batch_size, initialize_states=False
    ):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (
                conv_state,
                ssm_state,
            )
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[
                self.layer_idx
            ]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state


class Block(nn.Module):
    def __init__(
        self,
        dim,
        mixer_cls,
        norm_cls=nn.LayerNorm,
        fused_add_norm=False,
        residual_in_fp32=False,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self,
        hidden_states: Tensor,
        residual: Optional[Tensor] = None,
        inference_params=None,
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (
                (hidden_states + residual) if residual is not None else hidden_states
            )
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = (
                rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            )
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(
            batch_size, max_seqlen, dtype=dtype, **kwargs
        )


# --------------------------------------------------------
# EVA-02: A Visual Representation for Neon Genesis
# Github source: https://github.com/baaivision/EVA/EVA02
# Copyright (c) 2023 Beijing Academy of Artificial Intelligence (BAAI)
# Licensed under The MIT License [see LICENSE for details]
# By Yuxin Fang
#
# Based on https://github.com/lucidrains/rotary-embedding-torch
# --------------------------------------------------------'

from math import pi

import torch
from torch import nn

from einops import rearrange, repeat


def broadcat(tensors, dim=-1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, "tensors must all have the same number of dimensions"
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all(
        [*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]
    ), "invalid dimensions for broadcastable concatentation"
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim=dim)


def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


class VisionRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        pt_seq_len,
        ft_seq_len=None,
        custom_freqs=None,
        freqs_for="lang",
        theta=10000,
        max_freq=10,
        num_freqs=1,
    ):
        super().__init__()
        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == "lang":
            freqs = 1.0 / (
                theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
            )
        elif freqs_for == "pixel":
            freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * pi
        elif freqs_for == "constant":
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f"unknown modality {freqs_for}")

        if ft_seq_len is None:
            ft_seq_len = pt_seq_len
        t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len

        freqs_h = torch.einsum("..., f -> ... f", t, freqs)
        freqs_h = repeat(freqs_h, "... n -> ... (n r)", r=2)

        freqs_w = torch.einsum("..., f -> ... f", t, freqs)
        freqs_w = repeat(freqs_w, "... n -> ... (n r)", r=2)

        freqs = broadcat((freqs_h[:, None, :], freqs_w[None, :, :]), dim=-1)

        self.register_buffer("freqs_cos", freqs.cos())
        self.register_buffer("freqs_sin", freqs.sin())

        print("======== shape of rope freq", self.freqs_cos.shape, "========")

    def forward(self, t, start_index=0):
        rot_dim = self.freqs_cos.shape[-1]
        end_index = start_index + rot_dim
        assert (
            rot_dim <= t.shape[-1]
        ), f"feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}"
        t_left, t, t_right = (
            t[..., :start_index],
            t[..., start_index:end_index],
            t[..., end_index:],
        )
        t = (t * self.freqs_cos) + (rotate_half(t) * self.freqs_sin)
        return torch.cat((t_left, t, t_right), dim=-1)


class VisionRotaryEmbeddingFast(nn.Module):
    def __init__(
        self,
        dim,
        pt_seq_len=16,
        ft_seq_len=None,
        custom_freqs=None,
        freqs_for="lang",
        theta=10000,
        max_freq=10,
        num_freqs=1,
    ):
        super().__init__()
        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == "lang":
            freqs = 1.0 / (
                theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
            )
        elif freqs_for == "pixel":
            freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * pi
        elif freqs_for == "constant":
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f"unknown modality {freqs_for}")

        if ft_seq_len is None:
            ft_seq_len = pt_seq_len
        t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len

        freqs = torch.einsum("..., f -> ... f", t, freqs)
        freqs = repeat(freqs, "... n -> ... (n r)", r=2)
        freqs = broadcat((freqs[:, None, :], freqs[None, :, :]), dim=-1)

        freqs_cos = freqs.cos().view(-1, freqs.shape[-1])
        freqs_sin = freqs.sin().view(-1, freqs.shape[-1])

        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)

        print("======== shape of rope freq", self.freqs_cos.shape, "========")

    def forward(self, t):
        if t.shape[1] % 2 != 0:
            t_spatial = t[:, 1:, :]
            t_spatial = (
                t_spatial * self.freqs_cos + rotate_half(t_spatial) * self.freqs_sin
            )
            return torch.cat((t[:, :1, :], t_spatial), dim=1)
        else:
            return t * self.freqs_cos + rotate_half(t) * self.freqs_sin


__all__ = [
    "vim_tiny_patch16_224",
    "vim_small_patch16_224",
    "vim_base_patch16_224",
    "vim_tiny_patch16_384",
    "vim_small_patch16_384",
    "vim_base_patch16_384",
]


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        stride=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
    ):
        super().__init__()
        # img_size = to_3tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (
            (img_size[0] - patch_size[0]) // stride + 1,
            (img_size[1] - patch_size[1]) // stride + 1,
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=stride
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHWD -> BNC
        x = self.norm(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        mixer_cls,
        norm_cls=nn.LayerNorm,
        fused_add_norm=False,
        residual_in_fp32=False,
        drop_path=0.0,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        # import ipdb; ipdb.set_trace()
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self,
        hidden_states: Tensor,
        residual: Optional[Tensor] = None,
        inference_params=None,
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)

            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = (
                rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            )
            if residual is None:
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            else:
                hidden_states, residual = fused_add_norm_fn(
                    self.drop_path(hidden_states),
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(
            batch_size, max_seqlen, dtype=dtype, **kwargs
        )


def create_block(
    d_model,
    d_state=16,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    drop_path=0.0,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
    if_bimamba=False,
    bimamba_type="none",
    if_divide_out=False,
    init_layer_scale=None,
):
    if if_bimamba:
        bimamba_type = "v1"
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    # import ipdb; ipdb.set_trace()
    mixer_cls = partial(
        Mamba,
        d_state=d_state,
        layer_idx=layer_idx,
        bimamba_type=bimamba_type,
        if_divide_out=if_divide_out,
        init_layer_scale=init_layer_scale,
        **ssm_cfg,
        **factory_kwargs,
    )
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


class VisionMamba(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        stride=16,
        depth=24,
        embed_dim=192,
        d_state=16,
        channels=3,
        num_classes=1000,
        ssm_cfg=None,
        drop_rate=0.0,
        drop_path_rate=0.1,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = True,
        initializer_cfg=None,
        fused_add_norm=True,
        residual_in_fp32=True,
        device=None,
        dtype=None,
        ft_seq_len=None,
        pt_hw_seq_len=14,
        if_bidirectional=False,
        final_pool_type="none",
        if_abs_pos_embed=True,
        if_rope=False,
        if_rope_residual=False,
        flip_img_sequences_ratio=-1.0,
        if_bimamba=False,
        bimamba_type="v2",
        if_cls_token=True,
        if_divide_out=True,
        init_layer_scale=None,
        use_double_cls_token=False,
        use_middle_cls_token=True,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        # add factory_kwargs into kwargs
        kwargs.update(factory_kwargs)
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.if_bidirectional = if_bidirectional
        self.final_pool_type = final_pool_type
        self.if_abs_pos_embed = if_abs_pos_embed
        self.if_rope = if_rope
        self.if_rope_residual = if_rope_residual
        self.flip_img_sequences_ratio = flip_img_sequences_ratio
        self.if_cls_token = if_cls_token
        self.use_double_cls_token = use_double_cls_token
        self.use_middle_cls_token = use_middle_cls_token
        self.num_tokens = 1 if if_cls_token else 0

        # pretrain parameters
        self.num_classes = num_classes
        self.d_model = self.num_features = self.embed_dim = (
            embed_dim  # num_features for consistency with other models
        )

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            stride=stride,
            in_chans=channels,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        if if_cls_token:
            if use_double_cls_token:
                self.cls_token_head = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
                self.cls_token_tail = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
                self.num_tokens = 2
            else:
                self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
                # self.num_tokens = 1

        if if_abs_pos_embed:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches + self.num_tokens, self.embed_dim)
            )
            self.pos_drop = nn.Dropout(p=drop_rate)

        if if_rope:
            half_head_dim = embed_dim // 2
            hw_seq_len = img_size // patch_size
            self.rope = VisionRotaryEmbeddingFast(
                dim=half_head_dim, pt_seq_len=pt_hw_seq_len, ft_seq_len=hw_seq_len
            )
        self.head = (
            nn.Linear(self.num_features, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

        # TODO: release this comment
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        # import ipdb;ipdb.set_trace()
        inter_dpr = [0.0] + dpr
        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )
        # transformer blocks
        self.layers = nn.ModuleList(
            [
                create_block(
                    embed_dim,
                    d_state=d_state,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    if_bimamba=if_bimamba,
                    bimamba_type=bimamba_type,
                    drop_path=inter_dpr[i],
                    if_divide_out=if_divide_out,
                    init_layer_scale=init_layer_scale,
                    **factory_kwargs,
                )
                for i in range(depth)
            ]
        )

        # output head
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            embed_dim, eps=norm_epsilon, **factory_kwargs
        )

        # self.pre_logits = nn.Identity()

        # original init
        self.patch_embed.apply(segm_init_weights)
        self.head.apply(segm_init_weights)
        if if_abs_pos_embed:
            trunc_normal_(self.pos_embed, std=0.02)
        if if_cls_token:
            if use_double_cls_token:
                trunc_normal_(self.cls_token_head, std=0.02)
                trunc_normal_(self.cls_token_tail, std=0.02)
            else:
                trunc_normal_(self.cls_token, std=0.02)

        # mamba init
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(
                batch_size, max_seqlen, dtype=dtype, **kwargs
            )
            for i, layer in enumerate(self.layers)
        }

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            "pos_embed",
            "cls_token",
            "dist_token",
            "cls_token_head",
            "cls_token_tail",
        }

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)

    def forward_features(
        self,
        x,
        inference_params=None,
        if_random_cls_token_position=False,
        if_random_token_rank=False,
    ):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        x = self.patch_embed(x)
        B, M, _ = x.shape

        if self.if_cls_token:
            if self.use_double_cls_token:
                cls_token_head = self.cls_token_head.expand(B, -1, -1)
                cls_token_tail = self.cls_token_tail.expand(B, -1, -1)
                token_position = [0, M + 1]
                x = torch.cat((cls_token_head, x, cls_token_tail), dim=1)
                M = x.shape[1]
            else:
                if self.use_middle_cls_token:
                    cls_token = self.cls_token.expand(B, -1, -1)
                    token_position = M // 2
                    # add cls token in the middle
                    x = torch.cat(
                        (x[:, :token_position, :], cls_token, x[:, token_position:, :]),
                        dim=1,
                    )
                elif if_random_cls_token_position:
                    cls_token = self.cls_token.expand(B, -1, -1)
                    token_position = random.randint(0, M)
                    x = torch.cat(
                        (x[:, :token_position, :], cls_token, x[:, token_position:, :]),
                        dim=1,
                    )
                    print("token_position: ", token_position)
                else:
                    cls_token = self.cls_token.expand(
                        B, -1, -1
                    )  # stole cls_tokens impl from Phil Wang, thanks
                    token_position = 0
                    x = torch.cat((cls_token, x), dim=1)
                M = x.shape[1]

        if self.if_abs_pos_embed:
            # if new_grid_size[0] == self.patch_embed.grid_size[0] and new_grid_size[1] == self.patch_embed.grid_size[1]:
            #     x = x + self.pos_embed
            # else:
            #     pos_embed = interpolate_pos_embed_online(
            #                 self.pos_embed, self.patch_embed.grid_size, new_grid_size,0
            #             )
            x = x + self.pos_embed
            x = self.pos_drop(x)

        if if_random_token_rank:

            # 生成随机 shuffle 索引
            shuffle_indices = torch.randperm(M)

            if isinstance(token_position, list):
                print(
                    "original value: ",
                    x[0, token_position[0], 0],
                    x[0, token_position[1], 0],
                )
            else:
                print("original value: ", x[0, token_position, 0])
            print("original token_position: ", token_position)

            # 执行 shuffle
            x = x[:, shuffle_indices, :]

            if isinstance(token_position, list):
                # 找到 cls token 在 shuffle 之后的新位置
                new_token_position = [
                    torch.where(shuffle_indices == token_position[i])[0].item()
                    for i in range(len(token_position))
                ]
                token_position = new_token_position
            else:
                # 找到 cls token 在 shuffle 之后的新位置
                token_position = torch.where(shuffle_indices == token_position)[
                    0
                ].item()

            if isinstance(token_position, list):
                print(
                    "new value: ",
                    x[0, token_position[0], 0],
                    x[0, token_position[1], 0],
                )
            else:
                print("new value: ", x[0, token_position, 0])
            print("new token_position: ", token_position)

        if_flip_img_sequences = False
        if (
            self.flip_img_sequences_ratio > 0
            and (self.flip_img_sequences_ratio - random.random()) > 1e-5
        ):
            x = x.flip([1])
            if_flip_img_sequences = True

        # mamba impl
        residual = None
        hidden_states = x
        if not self.if_bidirectional:
            for layer in self.layers:

                if if_flip_img_sequences and self.if_rope:
                    hidden_states = hidden_states.flip([1])
                    if residual is not None:
                        residual = residual.flip([1])

                # rope about
                if self.if_rope:
                    hidden_states = self.rope(hidden_states)
                    if residual is not None and self.if_rope_residual:
                        residual = self.rope(residual)

                if if_flip_img_sequences and self.if_rope:
                    hidden_states = hidden_states.flip([1])
                    if residual is not None:
                        residual = residual.flip([1])

                hidden_states, residual = layer(
                    hidden_states, residual, inference_params=inference_params
                )
        else:
            # get two layers in a single for-loop
            for i in range(len(self.layers) // 2):
                if self.if_rope:
                    hidden_states = self.rope(hidden_states)
                    if residual is not None and self.if_rope_residual:
                        residual = self.rope(residual)

                hidden_states_f, residual_f = self.layers[i * 2](
                    hidden_states, residual, inference_params=inference_params
                )
                hidden_states_b, residual_b = self.layers[i * 2 + 1](
                    hidden_states.flip([1]),
                    None if residual == None else residual.flip([1]),
                    inference_params=inference_params,
                )
                hidden_states = hidden_states_f + hidden_states_b.flip([1])
                residual = residual_f + residual_b.flip([1])

        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = (
                rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            )
            hidden_states = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        # return only cls token if it exists
        if self.if_cls_token:
            if self.use_double_cls_token:
                return (
                    hidden_states[:, token_position[0], :]
                    + hidden_states[:, token_position[1], :]
                ) / 2
            else:
                if self.use_middle_cls_token:
                    return hidden_states[:, token_position, :]
                elif if_random_cls_token_position:
                    return hidden_states[:, token_position, :]
                else:
                    return hidden_states[:, token_position, :]

        if self.final_pool_type == "none":
            return hidden_states[:, -1, :]
        elif self.final_pool_type == "mean":
            return hidden_states.mean(dim=1)
        elif self.final_pool_type == "max":
            return hidden_states
        elif self.final_pool_type == "all":
            return hidden_states
        else:
            raise NotImplementedError

    def forward(
        self,
        x,
        return_features=False,
        inference_params=None,
        if_random_cls_token_position=False,
        if_random_token_rank=False,
    ):
        x = self.forward_features(
            x,
            inference_params,
            if_random_cls_token_position=if_random_cls_token_position,
            if_random_token_rank=if_random_token_rank,
        )
        if return_features:
            return x
        x = self.head(x)
        if self.final_pool_type == "max":
            x = x.max(dim=1)[0]
        return x


@register_model
def vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2(
    pretrained=False, **kwargs
):
    model = VisionMamba(
        patch_size=16,
        embed_dim=192,
        depth=24,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        final_pool_type="mean",
        if_abs_pos_embed=True,
        if_rope=False,
        if_rope_residual=False,
        bimamba_type="v2",
        if_cls_token=True,
        if_divide_out=True,
        use_middle_cls_token=True,
        **kwargs,
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="to.do", map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def vim_tiny_patch16_stride8_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2(
    pretrained=False, **kwargs
):
    model = VisionMamba(
        patch_size=16,
        stride=8,
        embed_dim=192,
        depth=24,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        final_pool_type="mean",
        if_abs_pos_embed=True,
        if_rope=False,
        if_rope_residual=False,
        bimamba_type="v2",
        if_cls_token=True,
        if_divide_out=True,
        use_middle_cls_token=True,
        **kwargs,
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="to.do", map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2(
    pretrained=False, **kwargs
):
    model = VisionMamba(
        patch_size=16,
        embed_dim=384,
        depth=24,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        final_pool_type="mean",
        if_abs_pos_embed=True,
        if_rope=False,
        if_rope_residual=False,
        bimamba_type="v2",
        if_cls_token=True,
        if_divide_out=True,
        use_middle_cls_token=True,
        **kwargs,
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="to.do", map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def vim_small_patch16_stride8_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2(
    pretrained=False, **kwargs
):
    model = VisionMamba(
        patch_size=16,
        stride=8,
        embed_dim=384,
        depth=24,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        final_pool_type="mean",
        if_abs_pos_embed=True,
        if_rope=False,
        if_rope_residual=False,
        bimamba_type="v2",
        if_cls_token=True,
        if_divide_out=True,
        use_middle_cls_token=True,
        **kwargs,
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="to.do", map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def vim_base_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_middle_cls_token_div2(
    pretrained=False, **kwargs
):
    model = VisionMamba(
        patch_size=16,
        embed_dim=768,
        d_state=16,
        depth=24,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        final_pool_type="mean",
        if_abs_pos_embed=True,
        if_rope=False,
        if_rope_residual=False,
        bimamba_type="v2",
        if_cls_token=True,
        if_devide_out=True,
        use_middle_cls_token=True,
        **kwargs,
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="to.do", map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


if __name__ == "__main__":
    add_torch_shape_forvs()
    with open("configs/data/lbl.yaml", "r", encoding="utf-8") as f:
        data_config = yaml.load(f.read(), Loader=yaml.FullLoader)
    input_size = data_config["input_size"]
    batch_size = data_config["batch_size"]
    in_channels = data_config["in_channels"]
    num_classes = data_config["num_classes"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(
        model_name="vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2",
        pretrained=False,
        num_classes=num_classes,
        channels=in_channels,
        img_size=input_size,
    ).to(device)
    img = torch.randn(
        batch_size, in_channels, input_size[0], input_size[1]
    ).to(device)
    summary(model=model, input_size=img.shape)
    preds = model(img)
    print(preds, preds[0].shape)
