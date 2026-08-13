import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

if HAS_TRITON:

    # ==========================================================================
    # FORWARD KERNEL
    # ==========================================================================
    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_H': 4, 'BLOCK_W': 4}),
            triton.Config({'BLOCK_H': 4, 'BLOCK_W': 8}),
            triton.Config({'BLOCK_H': 8, 'BLOCK_W': 4}),
            triton.Config({'BLOCK_H': 8, 'BLOCK_W': 8}),
            triton.Config({'BLOCK_H': 8, 'BLOCK_W': 16}),
        ],
        key=['H', 'W']
    )
    @triton.jit
    def _fwd_kernel(
            density_ptr,
            out_ptr,
            dirs_ptr,
            origin_ptr,
            depths_ptr,
            scale_ptr,
            t_ptr,

            B,
            C,
            W,
            H,
            D,

            stride_db,
            stride_dc,
            stride_dd,
            stride_dh,
            stride_dw,

            stride_sb,
            stride_sh,
            stride_sw,

            stride_vb,
            stride_v_row,

            ACC_DTYPE: tl.constexpr,   # tl.float32 or tl.float64 — accumulation/compute precision
            BLOCK_H: tl.constexpr,
            BLOCK_W: tl.constexpr,
    ):

        pid_h = tl.program_id(0)
        pid_w = tl.program_id(1)
        pid_b = tl.program_id(2)

        h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
        w_offs = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)

        h_mask = h_offs < H
        w_mask = w_offs < W
        hw_mask = h_mask[:, None] & w_mask[None, :]

        # ----------------------------------------------------------------------
        # Load ray directions
        #
        # dirs_ptr/origin_ptr/depths_ptr/scale_ptr are produced on the Python
        # side already cast to ACC_DTYPE (they're tiny — cam params / a
        # handful of scalars — so upcasting them costs ~nothing but keeps the
        # position math at full compute precision). density_ptr is left in
        # its native storage dtype and only cast to ACC_DTYPE per-element
        # after loading, since that's the tensor that's actually large.
        # ----------------------------------------------------------------------
        b_dirs_offset = pid_b * stride_sb

        dir_base = (
                b_dirs_offset
                + h_offs[:, None] * stride_sh
                + w_offs[None, :] * stride_sw
        )

        dir_x = tl.load(dirs_ptr + dir_base + 0, mask=hw_mask, other=0.0)
        dir_y = tl.load(dirs_ptr + dir_base + 1, mask=hw_mask, other=0.0)
        dir_z = tl.load(dirs_ptr + dir_base + 2, mask=hw_mask, other=0.0)

        # ----------------------------------------------------------------------
        # Camera origin / scaling
        # ----------------------------------------------------------------------
        b_view_offset = pid_b * stride_vb

        orig_x = tl.load(origin_ptr + b_view_offset + 0 * stride_v_row)
        orig_y = tl.load(origin_ptr + b_view_offset + 1 * stride_v_row)
        orig_z = tl.load(origin_ptr + b_view_offset + 2 * stride_v_row)

        sc_x = tl.load(scale_ptr + b_view_offset + 0 * stride_v_row)
        sc_y = tl.load(scale_ptr + b_view_offset + 1 * stride_v_row)
        sc_z = tl.load(scale_ptr + b_view_offset + 2 * stride_v_row)

        W_max = tl.load(t_ptr + 0).to(tl.int32)
        H_max = tl.load(t_ptr + 1).to(tl.int32)
        D_max = tl.load(t_ptr + 2).to(tl.int32)

        Wf = W_max.to(ACC_DTYPE)
        Hf = H_max.to(ACC_DTYPE)
        Df = D_max.to(ACC_DTYPE)

        base_b_density = pid_b * stride_db
        b_depths_ptr = depths_ptr + pid_b * D

        # ----------------------------------------------------------------------
        # Loop over channels
        # ----------------------------------------------------------------------
        for c in range(C):

            accum_density = tl.zeros((BLOCK_H, BLOCK_W), dtype=ACC_DTYPE)

            base_c = base_b_density + c * stride_dc

            # ------------------------------------------------------------------
            # Single-step ray marching
            # ------------------------------------------------------------------
            for d in range(D):
                depth = tl.load(b_depths_ptr + d)

                # --------------------------------------------------------------
                # Compute positions
                # --------------------------------------------------------------
                ix = (orig_x + dir_x * depth) * (Wf / (Wf - 1)) - 0.5
                iy = (orig_y + dir_y * depth) * (Hf / (Hf - 1)) - 0.5
                iz = (orig_z + dir_z * depth) * (Df / (Df - 1)) - 0.5

                ix0 = tl.floor(ix).to(tl.int32)
                iy0 = tl.floor(iy).to(tl.int32)
                iz0 = tl.floor(iz).to(tl.int32)

                ix1 = ix0 + 1
                iy1 = iy0 + 1
                iz1 = iz0 + 1

                # --------------------------------------------------------------
                # Per-axis, per-corner validity (matches grid_sample zeros
                # padding exactly: each of the 8 corners is independently
                # in- or out-of-bounds, not gated by the floor corner alone)
                # --------------------------------------------------------------
                x0_valid = (ix0 >= 0) & (ix0 < W_max)
                x1_valid = (ix1 >= 0) & (ix1 < W_max)
                y0_valid = (iy0 >= 0) & (iy0 < H_max)
                y1_valid = (iy1 >= 0) & (iy1 < H_max)
                z0_valid = (iz0 >= 0) & (iz0 < D_max)
                z1_valid = (iz1 >= 0) & (iz1 < D_max)

                m000 = hw_mask & x0_valid & y0_valid & z0_valid
                m001 = hw_mask & x0_valid & y0_valid & z1_valid
                m010 = hw_mask & x0_valid & y1_valid & z0_valid
                m011 = hw_mask & x0_valid & y1_valid & z1_valid
                m100 = hw_mask & x1_valid & y0_valid & z0_valid
                m101 = hw_mask & x1_valid & y0_valid & z1_valid
                m110 = hw_mask & x1_valid & y1_valid & z0_valid
                m111 = hw_mask & x1_valid & y1_valid & z1_valid

                # --------------------------------------------------------------
                # Weights
                # --------------------------------------------------------------
                wx1 = ix - ix0.to(ACC_DTYPE)
                wy1 = iy - iy0.to(ACC_DTYPE)
                wz1 = iz - iz0.to(ACC_DTYPE)

                wx0 = 1.0 - wx1
                wy0 = 1.0 - wy1
                wz0 = 1.0 - wz1

                # --------------------------------------------------------------
                # Base pointers
                #
                # NOTE: each corner is now clamped defensively before use.
                # tl.where is needed because Triton still computes the pointer
                # address for masked-off lanes before the load's mask gates
                # the memory access — an unclamped OOB index (e.g. ix0 == -1)
                # can produce a pointer that wraps outside the tensor's
                # allocation entirely, which is UB even when mask=False.
                # Clamping keeps the address always in-bounds; the mask is
                # what actually zeros the *value*.
                # --------------------------------------------------------------
                ix0c = tl.minimum(tl.maximum(ix0, 0), W_max - 1)
                ix1c = tl.minimum(tl.maximum(ix1, 0), W_max - 1)
                iy0c = tl.minimum(tl.maximum(iy0, 0), H_max - 1)
                iy1c = tl.minimum(tl.maximum(iy1, 0), H_max - 1)
                iz0c = tl.minimum(tl.maximum(iz0, 0), D_max - 1)
                iz1c = tl.minimum(tl.maximum(iz1, 0), D_max - 1)

                ptr_y0x0z0 = density_ptr + base_c + iy0c * stride_dh + ix0c * stride_dw + iz0c * stride_dd
                ptr_y0x0z1 = density_ptr + base_c + iy0c * stride_dh + ix0c * stride_dw + iz1c * stride_dd
                ptr_y1x0z0 = density_ptr + base_c + iy1c * stride_dh + ix0c * stride_dw + iz0c * stride_dd
                ptr_y1x0z1 = density_ptr + base_c + iy1c * stride_dh + ix0c * stride_dw + iz1c * stride_dd
                ptr_y0x1z0 = density_ptr + base_c + iy0c * stride_dh + ix1c * stride_dw + iz0c * stride_dd
                ptr_y0x1z1 = density_ptr + base_c + iy0c * stride_dh + ix1c * stride_dw + iz1c * stride_dd
                ptr_y1x1z0 = density_ptr + base_c + iy1c * stride_dh + ix1c * stride_dw + iz0c * stride_dd
                ptr_y1x1z1 = density_ptr + base_c + iy1c * stride_dh + ix1c * stride_dw + iz1c * stride_dd

                # --------------------------------------------------------------
                # Loads — mask determines the zero-padding, clamped index
                # just keeps the address itself safe
                # --------------------------------------------------------------
                # Loaded in density's native (compact) storage dtype, then
                # upcast to ACC_DTYPE for the interpolation math below —
                # this is the only place native-dtype memory traffic happens.
                v000 = tl.load(ptr_y0x0z0, mask=m000, other=0.0).to(ACC_DTYPE)
                v001 = tl.load(ptr_y0x0z1, mask=m001, other=0.0).to(ACC_DTYPE)
                v010 = tl.load(ptr_y1x0z0, mask=m010, other=0.0).to(ACC_DTYPE)
                v011 = tl.load(ptr_y1x0z1, mask=m011, other=0.0).to(ACC_DTYPE)

                v100 = tl.load(ptr_y0x1z0, mask=m100, other=0.0).to(ACC_DTYPE)
                v101 = tl.load(ptr_y0x1z1, mask=m101, other=0.0).to(ACC_DTYPE)
                v110 = tl.load(ptr_y1x1z0, mask=m110, other=0.0).to(ACC_DTYPE)
                v111 = tl.load(ptr_y1x1z1, mask=m111, other=0.0).to(ACC_DTYPE)

                interp = (
                        v000 * wx0 * wy0 * wz0 +
                        v001 * wx0 * wy0 * wz1 +
                        v010 * wx0 * wy1 * wz0 +
                        v011 * wx0 * wy1 * wz1 +
                        v100 * wx1 * wy0 * wz0 +
                        v101 * wx1 * wy0 * wz1 +
                        v110 * wx1 * wy1 * wz0 +
                        v111 * wx1 * wy1 * wz1
                )

                accum_density += interp

            # ------------------------------------------------------------------
            # Store output
            # ------------------------------------------------------------------
            out_idx = (
                    pid_b * (C * H * W)
                    + c * (H * W)
                    + h_offs[:, None] * W
                    + w_offs[None, :]
            )

            tl.store(out_ptr + out_idx, accum_density, mask=hw_mask)


    # ==========================================================================
    # BACKWARD KERNEL
    # ==========================================================================
    #
    # NOTE:
    # Backward remains expensive due to atomics.
    #
    # Biggest future optimization:
    #   voxel-space accumulation tiles
    #
    # ==========================================================================
    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_H': 4, 'BLOCK_W': 4}),
            triton.Config({'BLOCK_H': 4, 'BLOCK_W': 8}),
            triton.Config({'BLOCK_H': 8, 'BLOCK_W': 4}),
            triton.Config({'BLOCK_H': 8, 'BLOCK_W': 8}),
            triton.Config({'BLOCK_H': 8, 'BLOCK_W': 16}),
            # smaller blocks worth including explicitly here since backward's
            # bottleneck is atomic contention, not memory bandwidth — the
            # optimum may sit at a different point than forward's
            triton.Config({'BLOCK_H': 2, 'BLOCK_W': 2}),
            triton.Config({'BLOCK_H': 2, 'BLOCK_W': 4}),
        ],
        key=['H', 'W']
    )
    @triton.jit
    def _bwd_kernel(
            grad_out_ptr,
            grad_density_ptr,
            dirs_ptr,
            origin_ptr,
            depths_ptr,
            scale_ptr,            # unused; kept for signature parity
            t_ptr,

            B, C, H, W, D,        # D = R_dim = depths.shape[1]

            stride_db,
            stride_dc,
            stride_dd,            # same order as forward
            stride_dh,
            stride_dw,

            stride_sb,
            stride_sh,
            stride_sw,

            stride_vb,
            stride_v_row,

            ACC_DTYPE: tl.constexpr,   # tl.float32 or tl.float64 — accumulation/compute precision
            BLOCK_H: tl.constexpr,
            BLOCK_W: tl.constexpr,
    ):

        pid_h = tl.program_id(0)
        pid_w = tl.program_id(1)
        pid_b = tl.program_id(2)

        h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
        w_offs = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)

        h_mask = h_offs < H
        w_mask = w_offs < W
        hw_mask = h_mask[:, None] & w_mask[None, :]

        b_dirs_offset = pid_b * stride_sb
        dir_base = (
                b_dirs_offset
                + h_offs[:, None] * stride_sh
                + w_offs[None, :] * stride_sw
        )

        dir_x = tl.load(dirs_ptr + dir_base + 0, mask=hw_mask, other=0.0)
        dir_y = tl.load(dirs_ptr + dir_base + 1, mask=hw_mask, other=0.0)
        dir_z = tl.load(dirs_ptr + dir_base + 2, mask=hw_mask, other=0.0)

        b_view_offset = pid_b * stride_vb

        orig_x = tl.load(origin_ptr + b_view_offset + 0 * stride_v_row)
        orig_y = tl.load(origin_ptr + b_view_offset + 1 * stride_v_row)
        orig_z = tl.load(origin_ptr + b_view_offset + 2 * stride_v_row)

        W_max = tl.load(t_ptr + 0).to(tl.int32)
        H_max = tl.load(t_ptr + 1).to(tl.int32)
        D_max = tl.load(t_ptr + 2).to(tl.int32)

        Wf = W_max.to(ACC_DTYPE)
        Hf = H_max.to(ACC_DTYPE)
        Df = D_max.to(ACC_DTYPE)

        base_b_density = pid_b * stride_db
        b_depths_ptr = depths_ptr + pid_b * D   # D == R_dim now

        for c in range(C):

            grad_out_idx = (
                    pid_b * (C * H * W)
                    + c * (H * W)
                    + h_offs[:, None] * W
                    + w_offs[None, :]
            )

            g = tl.load(grad_out_ptr + grad_out_idx, mask=hw_mask, other=0.0).to(ACC_DTYPE)

            base_c = base_b_density + c * stride_dc

            for d in range(D):
                depth = tl.load(b_depths_ptr + d)

                ix = (orig_x + dir_x * depth) * (Wf / (Wf - 1)) - 0.5
                iy = (orig_y + dir_y * depth) * (Hf / (Hf - 1)) - 0.5
                iz = (orig_z + dir_z * depth) * (Df / (Df - 1)) - 0.5

                ix0 = tl.floor(ix).to(tl.int32)
                iy0 = tl.floor(iy).to(tl.int32)
                iz0 = tl.floor(iz).to(tl.int32)

                ix1 = ix0 + 1
                iy1 = iy0 + 1
                iz1 = iz0 + 1

                x0_valid = (ix0 >= 0) & (ix0 < W_max)
                x1_valid = (ix1 >= 0) & (ix1 < W_max)
                y0_valid = (iy0 >= 0) & (iy0 < H_max)
                y1_valid = (iy1 >= 0) & (iy1 < H_max)
                z0_valid = (iz0 >= 0) & (iz0 < D_max)
                z1_valid = (iz1 >= 0) & (iz1 < D_max)

                m000 = hw_mask & x0_valid & y0_valid & z0_valid
                m001 = hw_mask & x0_valid & y0_valid & z1_valid
                m010 = hw_mask & x0_valid & y1_valid & z0_valid
                m011 = hw_mask & x0_valid & y1_valid & z1_valid
                m100 = hw_mask & x1_valid & y0_valid & z0_valid
                m101 = hw_mask & x1_valid & y0_valid & z1_valid
                m110 = hw_mask & x1_valid & y1_valid & z0_valid
                m111 = hw_mask & x1_valid & y1_valid & z1_valid

                wx1 = ix - ix0.to(ACC_DTYPE)
                wy1 = iy - iy0.to(ACC_DTYPE)
                wz1 = iz - iz0.to(ACC_DTYPE)

                wx0 = 1.0 - wx1
                wy0 = 1.0 - wy1
                wz0 = 1.0 - wz1

                ix0c = tl.minimum(tl.maximum(ix0, 0), W_max - 1)
                ix1c = tl.minimum(tl.maximum(ix1, 0), W_max - 1)
                iy0c = tl.minimum(tl.maximum(iy0, 0), H_max - 1)
                iy1c = tl.minimum(tl.maximum(iy1, 0), H_max - 1)
                iz0c = tl.minimum(tl.maximum(iz0, 0), D_max - 1)
                iz1c = tl.minimum(tl.maximum(iz1, 0), D_max - 1)

                ptr_y0x0z0 = grad_density_ptr + base_c + iy0c * stride_dh + ix0c * stride_dw + iz0c * stride_dd
                ptr_y0x0z1 = grad_density_ptr + base_c + iy0c * stride_dh + ix0c * stride_dw + iz1c * stride_dd
                ptr_y1x0z0 = grad_density_ptr + base_c + iy1c * stride_dh + ix0c * stride_dw + iz0c * stride_dd
                ptr_y1x0z1 = grad_density_ptr + base_c + iy1c * stride_dh + ix0c * stride_dw + iz1c * stride_dd
                ptr_y0x1z0 = grad_density_ptr + base_c + iy0c * stride_dh + ix1c * stride_dw + iz0c * stride_dd
                ptr_y0x1z1 = grad_density_ptr + base_c + iy0c * stride_dh + ix1c * stride_dw + iz1c * stride_dd
                ptr_y1x1z0 = grad_density_ptr + base_c + iy1c * stride_dh + ix1c * stride_dw + iz0c * stride_dd
                ptr_y1x1z1 = grad_density_ptr + base_c + iy1c * stride_dh + ix1c * stride_dw + iz1c * stride_dd

                tl.atomic_add(ptr_y0x0z0, g * wx0 * wy0 * wz0, mask=m000)
                tl.atomic_add(ptr_y0x0z1, g * wx0 * wy0 * wz1, mask=m001)
                tl.atomic_add(ptr_y1x0z0, g * wx0 * wy1 * wz0, mask=m010)
                tl.atomic_add(ptr_y1x0z1, g * wx0 * wy1 * wz1, mask=m011)

                tl.atomic_add(ptr_y0x1z0, g * wx1 * wy0 * wz0, mask=m100)
                tl.atomic_add(ptr_y0x1z1, g * wx1 * wy0 * wz1, mask=m101)
                tl.atomic_add(ptr_y1x1z0, g * wx1 * wy1 * wz0, mask=m110)
                tl.atomic_add(ptr_y1x1z1, g * wx1 * wy1 * wz1, mask=m111)

def _acc_dtype_for(t: torch.Tensor) -> torch.dtype:
    """Precision to accumulate/compute in.

    fp64 input -> fp64 accumulation (needed for gradcheck-grade precision).
    fp16/bf16/fp32 input -> fp32 accumulation. Summing across `ray_samples`
    (often hundreds) of trilinear-interpolated terms in fp16/bf16 loses
    real accuracy (10/7 mantissa bits), so we always upcast at least to
    fp32 for the math while leaving the density tensor itself in its
    native (smaller) storage dtype to keep memory traffic/footprint low.
    """
    return torch.float64 if t.dtype == torch.float64 else torch.float32


def _tl_dtype(torch_dtype: torch.dtype):
    return tl.float64 if torch_dtype == torch.float64 else tl.float32


# ==============================================================================
# PYTORCH AUTOGRAD WRAPPER
# ==============================================================================
class _FusedVolumeRenderFunction(torch.autograd.Function):

    @staticmethod
    def forward(
            ctx,
            density,
            dirs_ijk,
            origin_ijk,
            depths,
            scale,
            vol_shape
    ):
        # Accumulation precision is driven by the density tensor's dtype;
        # the small auxiliary tensors (ray dirs/origin/depths/scale/shape)
        # are upcast to match so all position/weight math happens at full
        # compute precision without materializing a second copy of the
        # (large) density volume.
        acc_dtype = _acc_dtype_for(density)
        tl_acc_dtype = _tl_dtype(acc_dtype)

        dirs_ijk = dirs_ijk.to(acc_dtype)
        origin_ijk = origin_ijk.to(acc_dtype)
        depths = depths.to(acc_dtype)
        scale = scale.to(acc_dtype)
        vol_shape = vol_shape.to(acc_dtype)

        ctx.save_for_backward(
            density,
            dirs_ijk,
            origin_ijk,
            depths,
            scale,
            vol_shape
        )
        ctx.acc_dtype = acc_dtype

        B, C, W, H, D_dim = density.shape
        _, H_out, W_out, _ = dirs_ijk.shape
        R_dim = depths.shape[1]

        # Output carries the accumulation precision (fp32, or fp64 to match
        # a fp64 density input) rather than the density's own storage dtype,
        # so the trilinear-sum result isn't truncated before the caller has
        # a chance to use it. The output tensor is comparatively tiny
        # (B,C,H,W) vs. the density volume, so this costs little memory.
        out = torch.empty(
            (B, C, H_out, W_out),
            device=density.device,
            dtype=acc_dtype
        )

        grid = lambda meta: (
            triton.cdiv(H_out, meta["BLOCK_H"]),
            triton.cdiv(W_out, meta["BLOCK_W"]),
            B
        )

        _fwd_kernel[grid](
            density,
            out,
            dirs_ijk,
            origin_ijk,
            depths,
            scale,
            vol_shape,

            B,
            C,
            H_out,
            W_out,
            R_dim,

            *density.stride(),
            *dirs_ijk.stride()[:3],
            origin_ijk.stride(0),
            origin_ijk.stride(1),

            ACC_DTYPE=tl_acc_dtype,

            # BLOCK_H=8,
            # BLOCK_W=8,
        )

        return out

    @staticmethod
    def backward(ctx, grad_output):
        density, dirs_ijk, origin_ijk, depths, scale, vol_shape = ctx.saved_tensors
        acc_dtype = ctx.acc_dtype
        tl_acc_dtype = _tl_dtype(acc_dtype)

        B, C, W, H, D_dim = density.shape
        _, H_out, W_out, _ = dirs_ijk.shape

        # Accumulate gradients in acc_dtype via atomic_add (fp16/bf16 atomics
        # are poorly supported and would silently truncate every add), then
        # cast down to density's native dtype once at the very end — a
        # single rounding step instead of one per ray sample.
        grad_density_acc = torch.zeros(density.shape, device=density.device, dtype=acc_dtype)

        grad_output = grad_output.contiguous()
        if grad_output.dtype != acc_dtype:
            grad_output = grad_output.to(acc_dtype)

        grid = lambda meta: (
            triton.cdiv(H_out, meta["BLOCK_H"]),
            triton.cdiv(W_out, meta["BLOCK_W"]),
            B
        )

        R_dim = depths.shape[1]
        _bwd_kernel[grid](
            grad_output,
            grad_density_acc,
            dirs_ijk, origin_ijk, depths, scale, vol_shape,
            B, C, H_out, W_out, R_dim,  # was D_dim
            *density.stride(),
            *dirs_ijk.stride()[:3],
            origin_ijk.stride(0),
            origin_ijk.stride(1),
            ACC_DTYPE=tl_acc_dtype,
            # BLOCK_H=4, BLOCK_W=4,
        )

        grad_density = grad_density_acc.to(density.dtype)

        return grad_density, None, None, None, None, None


# ==============================================================================
# USER MODULE
# ==============================================================================
class FusedVolumeRenderer(torch.nn.Module):

    def __init__(
            self,
            ray_samples: int,
            density_factor: float = 1.0,
            apply_poisson_fn=None
    ):
        super().__init__()

        self.ray_samples = ray_samples
        self.density_factor = density_factor

        self.apply_poisson = (
            apply_poisson_fn
            if apply_poisson_fn is not None
            else lambda x: x
        )

    def forward(
            self,
            density: torch.Tensor,
            view_mat: torch.Tensor,
            near: torch.Tensor,
            far: torch.Tensor,
            ras2ijk: torch.Tensor,
            vol_shape: torch.Tensor,
            dirs_cam: torch.Tensor,
    ) -> torch.Tensor:
        if not HAS_TRITON:
            raise ModuleNotFoundError("Requires triton.")

        with torch.autocast(device_type='cuda', enabled=False):
            B = density.shape[0]
            device = density.device

            orig_dtype = density.dtype
            # fp64 density (e.g. gradcheck) -> fp64 compute throughout the ray
            # setup math; everything else (fp16/bf16/fp32) computes in fp32.
            # These are all small tensors (camera params, per-ray directions),
            # so upcasting them is essentially free — the memory- and
            # bandwidth-heavy tensor is `density`, which stays in its native
            # dtype and is only upcast per-sample inside the kernel.
            compute_dtype = torch.float64 if orig_dtype == torch.float64 else torch.float32

            view_mat = view_mat.to(compute_dtype)
            near = near.to(compute_dtype)
            far = far.to(compute_dtype)
            ras2ijk = ras2ijk.to(device, dtype=compute_dtype)
            vol_shape = vol_shape.to(device, dtype=compute_dtype)
            dirs_cam = dirs_cam.to(compute_dtype)

            cam2world = view_mat[:, :3, :3]
            cam_pos = view_mat[:, :3, 3]

            R = ras2ijk[:3, :3]
            t = ras2ijk[:3, 3]


            dirs_world = torch.einsum(
                'bij,hwj->bhwi',
                cam2world,
                dirs_cam
            )

            dirs_ijk = torch.einsum('bhwi,ji->bhwj', dirs_world, R)

            origin_ijk = torch.einsum('bi,ji->bj', cam_pos, R) + t

            scale = (
                    2.0 / (vol_shape - 1)
            ).view(1, 1, 1, 1, 3)

            t_vals = torch.linspace(0, 1, self.ray_samples, device=device, dtype=compute_dtype)[None, :]
            depths = near[:, None] + (far - near)[:, None] * t_vals

            # What should be correct ijk coordinate math. Want to compute within kernel to avoid materializing large tensor
            #ijk = scale * origin_ijk[:, None, None, None, :] + dirs_ijk.unsqueeze(1) * depths[:, :, None, None, None] - 1

            fused_sum = _FusedVolumeRenderFunction.apply(
                density,
                dirs_ijk,
                origin_ijk,
                depths,
                scale,
                vol_shape
            )  # returned in compute_dtype (see _acc_dtype_for)

            step_size = (
                    0.1
                    * (far - near)
                    / self.ray_samples
            ).view(B, 1, 1, 1)

            scaled_extinction = torch.exp(
                -fused_sum * step_size
            )

            out = 1.0 - self.apply_poisson(scaled_extinction)

            # Restore the caller's dtype at the boundary — internal math stayed
            # at compute_dtype for accuracy, but callers shouldn't be surprised
            # by a fp16 volume producing an fp32 render.
            return out.to(orig_dtype)