# Very much inspired by the excellent code from Philipp Henzler on his PlatonicGAN, ICCV19
# PlatonicGAN: https://henzler.github.io/publication/platonicgan/
# See https://github.com/henzler/platonicgan/blob/master/scripts/renderer
#%%
import math
from typing import Tuple, Optional

import monai.data
import torch
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from monai.networks.blocks import Convolution
from torch import nn
import numpy as np

from torchvtk.utils import make_2d, apply_tf_torch, apply_tf_tex_torch

from timeit import default_timer as timer

from torchvtk.rendering.triton_raycast import FusedVolumeRenderer

__all__ = ['homogenize_mat', 'homogenize_vec', 'get_proj_mat', 'get_view_mat', 'get_vtk_view_mat', 'get_random_pos', 'VolumeRaycaster']

def homogenize_mat(mat):
    ''' Adds a row (bottom) and column (right) to the matrix `mat` with all zeros and a 1 in the lower right corner.
    Is used to make the matrix work as transformation for homogeneous coordinates. '''
    assert torch.is_tensor(mat)
    ret = torch.eye(mat.size(-1)+1, dtype=mat.dtype, device=mat.device)
    if mat.ndim > 2:
        flat_mat = mat.view(-1, mat.size(-2), mat.size(-1))
        num_mats = flat_mat.size(0)
        ret = ret[None].expand(num_mats, -1, -1)
    else: flat_mat = mat
    ret[..., :mat.size(-2), :mat.size(-1)] = flat_mat
    return ret.reshape(*mat.shape[:-2], mat.size(-2)+1, mat.size(-1)+1)

def homogenize_vec(vec, dim=None):
    ''' Adds an additional component to `vec` with value 1 to make it a homogeneous coordinate. '''
    assert torch.is_tensor(vec) and 3 in vec.shape
    if dim is None: dim = vec.ndim - list(reversed(vec.shape)).index(3) - 1
    ad_shape = list(vec.shape); ad_shape[dim] = 1
    nu = torch.ones(ad_shape, dtype=vec.dtype, device=vec.device)
    return torch.cat([vec, nu], dim=dim)

def get_proj_mat(fov, aspect, near=0.1, far=100, dtype=None, device=None):
    ''' Computes a projection matrix according to inputs
    Args:
        fov (float): Field of View in radians
        aspect (float): Aspect ratio width/height of the viewport
        near (float): Near plane distance
        far (float): Far plane distance
        dtype (torch.dtype): Torch type to cast matrix to
        device (torch.device): Device to put matrix on
    Returns:
        Perspective projection matrix as torch tensor of shape (4,4)
    '''
    q = 1 / math.tan(fov * 0.5)
    a = q / aspect
    # b = (far + near) / (near - far)
    # c = (2*far*near) / (near - far)
    b = -far / (far-near)
    c = - (far*near)/(far-near)

    return torch.tensor([[a, 0, 0, 0],
                         [0, q, 0, 0],
                         [0, 0, b,-1],
                         [0, 0, c, 0]]).to(dtype).to(device)

# import glm
# def get_random_view_mat(distance=(2,3)):
#     if isinstance(distance, (int, float)): distance = (distance, distance)
#     assert isinstance(distance, (list, tuple)) and len(distance) == 2
#     look_from = np.random.normal(0, 1, (3,)) # rand pos with distance in [2,3]
#     look_from = glm.vec3(look_from / np.linalg.norm(look_from) * np.random.uniform(distance[0], distance[1], (1,)))
#     look_to = glm.vec3(0.5)
#     up = glm.vec3(0.0, 1, 0)
#     view_dir = glm.normalize(look_to - look_from)
#     right = glm.cross(view_dir, up)
#     look_up = glm.cross(right, view_dir)
#     return torch.tensor(glm.lookAt(look_from, look_to, look_up).to_list())

def get_view_mat(look_from, look_to=None, look_up=None, dtype=None):
    ''' Computes a view matrix based on camera parameters.
    Args:
        look_from (Tensor (BS, 3)): Batch of camera positions
        look_up (Tensor (BS, 3)): Batch of up vectors
        look_to (Tensor (BS, 3)): Batch of position vectors of the object of interest
    Returns:
        View matrix as torch Tensor of shape (BS, 3, 3). Uses device of `look_from`
    '''
    look_from = make_2d(look_from)
    bs, dev = look_from.size(0), look_from.device
    if look_up is None: look_up = F.normalize(torch.eye(3)[1].expand(bs, 3), dim=1).to(dev)
    else:               look_up = make_2d(look_up)
    if look_to is None: look_to = F.normalize(torch.zeros(3).expand( bs, 3), dim=1).to(dev)
    else:               look_to = make_2d(look_to)

    z = F.normalize(look_from - look_to, dim=1).expand(bs, 3)
    x = F.normalize(torch.cross(look_up, z), dim=1)
    y = torch.cross(z, x, dim=1)
    ret = torch.eye(4)[None].expand(bs, -1, -1)
    ret[..., :3, :3] = torch.stack([x, y, z], dim=1)
    ret[...,  3, :3] = torch.stack([torch.bmm(x.unsqueeze(-2), look_from.unsqueeze(-1)),
                                    torch.bmm(y.unsqueeze(-2), look_from.unsqueeze(-1)),
                                    torch.bmm(z.unsqueeze(-2), look_from.unsqueeze(-1))]).squeeze()
    return ret.to(dtype)

def lookAt(look_from, look_up=None):
    view_dir = F.normalize(-look_from)
    if look_up is None:
        look_up = torch.tensor([1.0, .0, 0], dtype=look_from.dtype, device=look_from.device).expand(look_from.size(0), -1)
        right = torch.cross(look_up, view_dir)
        up = torch.cross(right, view_dir)
    else:
        up = F.normalize(look_up)
        right = torch.cross(look_up, view_dir)
    mat = torch.eye(4).expand(look_from.size(0), -1, -1)
    mat[:, 0, :3] = right
    mat[:, 1, :3] = up
    mat[:, 2, :3] = view_dir
    tmat = torch.eye(4).expand(look_from.size(0), -1, -1)
    # tmat[:, :3,  3] = -look_from
    fmat = torch.matmul(mat, tmat).permute(0, 2, 1)
    return fmat


def get_random_carm_views(n_views, sid_range, ap_range, lat_range, si_range, center):
    """
    Randomly sample C-arm view parameters.

    Parameters:
    -----------
    sid_range : tuple of 2 floats
        (min_sid, max_sid) in mm. Typical range: (900, 1200)
    ap_range : tuple of 2 floats
        (min_ap, max_ap) in degrees. Typical range: (-40, 40) for cranial/caudal
    lat_range : tuple of 2 floats
        (min_lat, max_lat) in degrees. Typical range: (-90, 90) for LAO/RAO
    si_range : tuple of 2 floats
        (min_si, max_si) in mm. Typical range: (-100, 100) for table translation

    Returns:
    --------
    sid : float
        Source-to-image distance in mm
    ap_angle : float
        AP angle in degrees
    lat_angle : float
        Lateral angle in degrees
    table_si : float
        Table superior-inferior translation in mm

    Notes:
    ------
    Distribution choice rationale:
    - SID: Uniform is appropriate - mechanical constraint, no preferred distance
    - AP angle: Uniform is reasonable for training, though clinical use shows
      bias toward AP (0°), cranial (15-30°), and steep cranial (>30°)
    - Lateral angle: Uniform works, but clinical practice favors AP (0°),
      RAO 30°, and LAO 30-45° views
    - Table SI: Uniform is appropriate - depends on anatomy region of interest

    For more realistic clinical distributions, consider:
    - Adding bias toward common views (0°, ±30°, ±45°)
    - Using a mixture of uniforms or peaked distributions
    """
    import random

    views = []
    for _ in range(n_views):
        sid = random.uniform(sid_range[0], sid_range[1])
        ap_angle = random.uniform(ap_range[0], ap_range[1])
        lat_angle = random.uniform(lat_range[0], lat_range[1])
        table_si = random.uniform(si_range[0], si_range[1])

        pos, focal, up = carm_to_camera_params(sid, ap_angle, lat_angle, center, table_si)
        views.append(get_vtk_view_mat(pos, focal, up))

    views = torch.stack(views)

    return views

def carm_to_camera_params(sid, ap_angle, lat_angle, center_ras, table_si=0.0):
    """
    Convert C-arm position parameters to camera parameters.

    Parameters:
    -----------
    sid : float
        Source-to-Image Distance in mm (distance from X-ray source to detector)
    ap_angle : float
        Anteroposterior (AP) angle in degrees
        - 0° = AP view (source in front, looking posterior)
        - Positive = cranial angulation
        - Negative = caudal angulation
    lat_angle : float
        Lateral angle in degrees
        - 0° = AP view
        - Positive = RAO (source moves to patient's right)
        - Negative = LAO (source moves to patient's left)
    center_ras : tuple or list of 3 floats
        Center point of the CT scan in RAS coordinates (mm)
        RAS = Right, Anterior, Superior
    table_si : float, optional
        Table translation in superior-inferior direction in mm (default: 0.0)
        - Positive = table moves superior (head up)
        - Negative = table moves inferior (head down)

    Returns:
    --------
    cam_pos : numpy array (3,)
        Camera/source position in RAS coordinates
    look_at : numpy array (3,)
        Look-at point (focal point) in RAS coordinates
    look_up : numpy array (3,)
        Up vector for camera orientation
    """

    # Convert angles to radians
    ap_rad = np.deg2rad(ap_angle)
    lat_rad = np.deg2rad(lat_angle)

    # Center point with table translation
    # Table moves in superior-inferior direction (Z-axis in RAS)
    center = np.array(center_ras) + np.array([0, 0, table_si])

    # Initial source position at SID distance along negative Y axis (anterior)
    source_local = np.array([0, -sid, 0])

    # Rotation around Z-axis (superior) for lateral angulation
    R_lat = np.array([
        [np.cos(lat_rad), -np.sin(lat_rad), 0],
        [np.sin(lat_rad), np.cos(lat_rad), 0],
        [0, 0, 1]
    ])

    # Rotation around X-axis (right) for cranial/caudal angulation
    R_ap = np.array([
        [1, 0, 0],
        [0, np.cos(ap_rad), -np.sin(ap_rad)],
        [0, np.sin(ap_rad), np.cos(ap_rad)]
    ])

    # Apply rotations: first lateral, then AP
    R_total = R_ap @ R_lat
    source_rotated = R_total @ source_local

    # Camera position in world coordinates
    cam_pos = source_rotated + center

    # Look-at point is the center
    look_at = center

    # Up vector: start with superior direction and apply same rotations
    up_local = np.array([0, 0, 1])
    look_up = R_total @ up_local

    return cam_pos, look_at, look_up



def get_vtk_view_mat(cam_pos: Tuple[float],  # (3,) camera center in RAS
                     cam_focal: Tuple[float],  # (3,) camera focal point in RAS
                     cam_viewup: Tuple[float],  # (3,) view-up vector in RAS)
                     device: str = 'cpu'):
    cam_pos = torch.as_tensor(cam_pos, dtype=torch.float32)
    cam_focal = torch.as_tensor(cam_focal, dtype=torch.float32)
    cam_viewup = torch.as_tensor(cam_viewup, dtype=torch.float32)

    # Construct VTK-compatible camera axes
    forward = torch.nn.functional.normalize(cam_focal - cam_pos, dim=0)  # +Z axis (direction of projection in VTK)
    right = torch.nn.functional.normalize(torch.linalg.cross(forward, cam_viewup), dim=0)
    up = torch.linalg.cross(right, forward)  # True up direction

    # View matrix (camera-to-world)
    view_mat = torch.zeros(4, 4, device=device)
    view_mat[:3, 0] = right
    view_mat[:3, 1] = up
    view_mat[:3, 2] = forward 
    view_mat[:3, 3] = cam_pos
    view_mat[3, 3] = 1.0

    return view_mat


def get_rot_mat(look_from, old_look_from=None):
    if old_look_from is None:
        old_look_from = torch.zeros_like(look_from)
        old_look_from[..., 2] = 1.0
    look_from     = F.normalize(look_from, dim=1)
    old_look_from = F.normalize(old_look_from, dim=1)
    v = torch.cross(old_look_from, look_from, dim=1)
    s = torch.norm(v, dim=1)
    c = torch.bmm(old_look_from.unsqueeze(-2), look_from.unsqueeze(-1))
    vx = torch.tensor([[0,       -v[:,2], v[:,1]], # skew symmetric cross product matrix
                       [ v[:,2],    0,   -v[:,0]],
                       [-v[:,1],  v[:,0],    0  ]])
    return torch.eye(3) + vx + vx**2 * (1/(1+c))

def get_random_pos(bs=1, distance=(1,5)):
    ''' Computes a vector of random positions.
    Args:
        bs (int): Batch size, number of positions to generate
        distance (float, tuple of floats): Either a fixed distance or a range from which the distance is sampled uniformly
    Returns:
        List / Batch of random camera positions as torch Tensor of shape (BS, 3)
    '''
    if   isinstance(distance, (tuple, list)): # Draw random in between
        d = torch.rand(bs, 1) * (distance[1] - distance[0]) + distance[0]
    elif isinstance(distance, (int, float)):
        d = distance
    return F.normalize(torch.randn(bs, 3)) * d

def piecewise_linear_channelwise(x, xp, yp):
    """
    Apply a per-channel piecewise linear function to input x.
    Handles extrapolation beyond xp bounds by extending the first/last segment.

    Args:
        x (Tensor): Input (B, C, H, W) or (B, C, D, H, W)
        xp (Tensor): (C, K) sorted x keypoints per channel
        yp (Tensor): (C, K) y values per channel at keypoints

    Returns:
        Tensor: Output of same shape as x
    """
    if xp.ndim != 2 or yp.ndim != 2 or xp.shape != yp.shape:
        print(xp.shape, yp.shape)
        raise ValueError("xp and yp must have shape (C, K)")
    B, C = x.shape[:2]
    x_flat = x.view(B, C, -1)  # (B, C, N)

    K = xp.shape[1]
    xp = xp.unsqueeze(0).expand(B, -1, -1)  # (B, C, K)
    yp = yp.unsqueeze(0).expand(B, -1, -1)

    x_unsq = x_flat.unsqueeze(-1)  # (B, C, N, 1)
    xp_left = xp.unsqueeze(2)[:, :, :, :-1]  # (B, C, 1, K-1)
    xp_right = xp.unsqueeze(2)[:, :, :, 1:]

    # Mask to identify correct segment
    mask = (x_unsq >= xp_left) & (x_unsq < xp_right)

    # If no valid segment (i.e. x >= xp[-1]), assign to last segment
    none_selected = ~mask.any(dim=-1)
    idx = mask.float().argmax(dim=-1)  # (B, C, N)
    idx[none_selected] = K - 2  # assign last interval

    # Safe gather
    gather_idx = idx.unsqueeze(-1)
    xp_l = torch.gather(xp, 2, idx)
    xp_r = torch.gather(xp, 2, idx + 1)
    yp_l = torch.gather(yp, 2, idx)
    yp_r = torch.gather(yp, 2, idx + 1)

    # Linear interpolation
    denom = xp_r - xp_l
    denom = torch.where(denom == 0, torch.ones_like(denom), denom)  # avoid div0
    xval = x_flat
    yval = yp_l + (yp_r - yp_l) * ((xval - xp_l) / denom)

    return yval.view_as(x)


class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch, rates=(1, 2, 4, 8)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv2d(in_ch, out_ch, 3, padding=r, dilation=r) for r in rates
        ])
        self.proj = nn.Conv2d(out_ch * len(rates), out_ch, 1)

        # init
        for m in self.branches:
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            nn.init.zeros_(m.bias)
        nn.init.kaiming_normal_(self.proj.weight, nonlinearity="relu")
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        feats = [F.relu(branch(x)) for branch in self.branches]
        return self.proj(torch.cat(feats, dim=1))

class DepthAwareScatter(nn.Module):
    def __init__(self, in_ch, base_ch=32, alpha_max=0.4):
        super().__init__()
        self.alpha_max = alpha_max

        # depth mixing: causal 1D conv over D
        self.depth_conv = nn.Conv1d(in_ch, in_ch, kernel_size=5, padding=4, dilation=2)
        nn.init.kaiming_normal_(self.depth_conv.weight, nonlinearity="relu")
        nn.init.zeros_(self.depth_conv.bias)

        # conditioning features → ASPP
        self.enc = nn.Conv2d(in_ch * 2, base_ch, 1)
        self.aspp = ASPP(base_ch, base_ch)
        self.alpha_head = nn.Conv2d(base_ch, 1, 1)
        nn.init.kaiming_normal_(self.alpha_head.weight, nonlinearity="relu")
        nn.init.constant_(self.alpha_head.bias, -3.0)  # α ≈ 0 at init

        # lateral scatter mixing
        self.scatter_conv = nn.Conv2d(in_ch + base_ch, 1, 3, padding=1)
        nn.init.kaiming_normal_(self.scatter_conv.weight, nonlinearity="relu")  # no scatter initially
        nn.init.zeros_(self.scatter_conv.bias)


    def forward(self, mu, dz=1.0):
        # mu: [B,C,D,H,W], I0 = 1
        B, C, D, H, W = mu.shape

        # integrate attenuation
        tau_z = torch.cumsum(mu * dz, dim=2)           # [B,C,D,H,W]
        I_z   = torch.exp(-tau_z)                           # [B,C,D,H,W]
        I_primary = I_z[:, :, -1]                           # [B,C,H,W]

        # scatter source per depth
        S_z = mu * dz * I_z                            # [B,C,D,H,W]

        # depth-aware weighting (causal conv along D)
        # reshape to [B*H*W, C, D]
        S_z_perm = S_z.permute(0, 3, 4, 1, 2).contiguous()  # [B,H,W,C,D]
        S_z_flat = S_z_perm.view(-1, C, D)                  # [B*H*W, C, D]
        S_w = self.depth_conv(S_z_flat)                     # [B*H*W, C, D]
        S_w = S_w.view(B, H, W, C, D).permute(0, 3, 4, 1, 2)  # [B,C,D,H,W]
        S = S_w.sum(dim=2)                                  # [B,C,H,W]

        # conditioning features
        tau_exit = tau_z[:, :, -1]                          # [B,C,H,W]
        feats = torch.cat([I_primary, tau_exit], dim=1)     # [B,2C,H,W]
        F0 = F.relu(self.enc(feats))                        # [B,base,H,W]

        # ASPP context
        F_aspp = self.aspp(F0)                              # [B,base,H,W]

        # scatter map
        scatter_in = torch.cat([S, F_aspp], dim=1)          # [B,C+base,H,W]
        scatter_map = F.relu(self.scatter_conv(scatter_in)) # [B,1,H,W]

        # alpha gate
        alpha = torch.sigmoid(self.alpha_head(F_aspp)) * self.alpha_max

        # output
        I_out = I_primary + alpha * scatter_map
        return I_out, I_primary, scatter_map, alpha

#%%
class VolumeRaycaster(nn.Module):
    def __init__(
            self,
            density_factor: float = 100.0,
            ray_samples: int = 256,
            resolution: Tuple[int, int] = (224, 224),
            use_checkpointing: bool = False,
            use_beer_lambert: bool = True,
            scatter: Optional[int] = None,  # or int | None if Python 3.10+
            i0: Optional[float] = None,
            fov: Optional[float] = 20.0
    ) -> None:
        ''' Initializes differentiable raycasting layer

        Args:
            density_factor (float): scales the overall density
            ray_samples (int): Number of samples along the rays
            resolution (int, (int, int)): Tuple describing width and height of the render. A single int produces a square image
            '''
        super().__init__()
        self.density_factor = density_factor
        self.ray_samples    = ray_samples
        self.use_checkpointing = use_checkpointing
        self.use_beer_lambert = use_beer_lambert
        self.scatter = None
        self.i0 = i0
        if isinstance(resolution, tuple):
              self.w, self.h = resolution
        else: self.w, self.h = resolution, resolution

        # Z = torch.linspace(-1, 1, ray_samples)
        # W = torch.linspace(-1, 1, self.w)
        # H = torch.linspace(-1, 1, self.h)
        # self.samples = self.get_coord_grid(Z, H, W, perspective=True)
        self.register_buffer('dirs_cam', torch.empty(self.h, self.w, 3), persistent=False)
        self.triton_raycaster = FusedVolumeRenderer(self.ray_samples, self.density_factor, self.apply_poisson)


        if scatter is not None:
            self.scatter_channels = scatter
            self.scatter = DepthAwareScatter(scatter)

        self.set_fov(fov)

    def set_fov(self, fov: float) -> None:
        # Get the current device from the buffer
        device = self.dirs_cam.device

        # Do computation on CPU
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, self.h),  # vertical: -1 (bottom) to 1 (top)
            torch.linspace(-1, 1, self.w),  # horizontal: -1 (left) to 1 (right)
            indexing='ij'
        )

        # Vertical FOV in radians
        fov_y = np.deg2rad(fov)
        aspect = self.w / self.h
        # Compute direction vectors in camera coordinates
        px = x * np.tan(fov_y / 2) * aspect
        py = y * np.tan(fov_y / 2)
        pz = torch.ones_like(px)
        dirs_cam = torch.stack([px, -py, pz], dim=-1)  # shape: (H, W, 3)

        # Normalize directions and move to device once at the end
        dirs_cam = F.normalize(dirs_cam, dim=-1).to(device)
        self.dirs_cam = dirs_cam

    def get_coord_grid(self, z, y, x, perspective=False, fovy=0.20, ar=1.0):
        ''' Computes the samples given linspaces of the correct sizes for each spatial dimension. '''
        z, y, x = torch.meshgrid(z, y, x)
        coords = torch.stack([x, y, z], dim=-1)

        if perspective:
            fovx = ar * fovy
            sins = torch.sin(torch.Tensor([fovx/2, fovy/2]))
            coords[...,[0,1]] *= 1 + sins * coords[...,[2]]
        return coords

    def get_perspective_coord_grid(self, view_mat):
        bs = view_mat.size(0)
        corners = torch.tensor([
            [-1,-1,-1,    1],
            [-1,-1, 1,    1],
            [-1, 1,-1,    1],
            [-1, 1, 1,    1],
            [ 1,-1,-1,    1],
            [ 1,-1, 1,    1],
            [ 1, 1,-1,    1],
            [ 1, 1, 1,    1.0]]).expand(bs, -1, -1)
        c_vs = torch.bmm(corners, view_mat) # Corners in View space
        dist = torch.norm(c_vs[..., :3], dim=2)
        nears, fars = torch.clamp(dist.min(dim=1).values - 2, 0), dist.max(dim=1).values + 2
        c_h = F.normalize(c_vs[...,[0,2]].reshape(-1, 2), dim=1)
        c_v = F.normalize(c_vs[...,[1,2]].reshape(-1, 2), dim=1)
        vd  = F.normalize(torch.mean(c_vs, dim=1), dim=1).unsqueeze(1).expand(-1, 8, -1).reshape(-1, 4)

        alphas_h = torch.acos(torch.bmm(c_h.unsqueeze(-2), vd[...,[0,2]].unsqueeze(-1)).reshape(bs, 8))
        alphas_v = torch.acos(torch.bmm(c_v.unsqueeze(-2), vd[...,[1,2]].unsqueeze(-1)).reshape(bs, 8))
        fovs_h   = torch.max(alphas, dim=1).values
        fovs_v   = torch.max(alphas, dim=1).values
        ars  = fovs_h / fovs_v
        fovs = fovs_v + 0.05 # About 2.8deg larger than necessary to fit the volume in the frustum

        proj_mat = torch.stack([get_proj_mat(fov, ar, near, far) for fov, ar, near, far in zip(fovs, ars, nears, fars)])

        inv_view, inv_proj = torch.inverse(view_mat), torch.inverse(proj_mat)
        inv_tfm = torch.bmm(view_mat, proj_mat)
        z, y, x = torch.meshgrid(torch.linspace(-1, 1, self.ray_samples),
                                 torch.linspace(-1, 1, self.h),
                                 torch.linspace(-1, 1, self.w))
        samples = torch.stack([x,y,z, torch.ones_like(x)], dim=-1).expand(bs,-1,-1,-1,-1)
        samples_poses = torch.bmm(samples.reshape(bs, -1, 4), inv_tfm).reshape(bs, self.ray_samples, self.h, self.w, 4)

    @staticmethod
    def compute_clipping_distances(camera_matrix, volume_shape, ijk2ras, margin=30.0):
        """
        Compute near and far clipping distances for ray sampling based on camera matrix
        and volume bounds in IJK space.

        Parameters:
        -----------
        camera_matrix : torch.Tensor (B, 4, 4) or (4, 4)
            Batched or single 4x4 camera view matrix in RAS space
            Can be world-to-camera or camera-to-world (will be auto-detected)
        volume_shape : tuple of 3 ints
            Volume dimensions in IJK space (I, J, K)
        ijk2ras : torch.Tensor (4, 4)
            Transformation matrix from IJK to RAS coordinates
        margin : float, optional
            Safety margin to add to near/far distances in mm (default: 0.0)
            Positive values expand the clipping range

        Returns:
        --------
        near : torch.Tensor (B,) or float
            Near clipping distance in mm (distance from camera to closest volume point)
        far : torch.Tensor (B,) or float
            Far clipping distance in mm (distance from camera to farthest volume point)
        """

        if not isinstance(camera_matrix, torch.Tensor):
            camera_matrix = torch.tensor(camera_matrix, dtype=torch.float32)
        if not isinstance(ijk2ras, torch.Tensor):
            ijk2ras = torch.tensor(ijk2ras, dtype=torch.float32)


        # Handle both batched and single camera matrices
        is_batched = camera_matrix.ndim == 3
        if not is_batched:
            camera_matrix = camera_matrix.unsqueeze(0)

        batch_size = camera_matrix.shape[0]
        device = camera_matrix.device

        ijk2ras = ijk2ras.to(dtype=camera_matrix.dtype, device=device)

        # Extract camera position from matrix
        # Check if this is camera-to-world (large translation) or world-to-camera (small translation)
        test_pos = camera_matrix[:, :3, 3]  # (B, 3)
        translation_norm = torch.norm(test_pos, dim=1)  # (B,)

        # For matrices with small translation, assume world-to-camera and invert
        needs_invert = translation_norm < 1.0

        cam_pos_ras = torch.zeros(batch_size, 3, device=device)
        view_dir = torch.zeros(batch_size, 3, device=device)

        for i in range(batch_size):
            if needs_invert[i]:
                cam_to_world = torch.inverse(camera_matrix[i])
                cam_pos_ras[i] = cam_to_world[:3, 3]
                view_dir[i] = cam_to_world[:3, 2]
            else:
                cam_pos_ras[i] = camera_matrix[i, :3, 3]
                view_dir[i] = camera_matrix[i, :3, 2]

        # Normalize view directions
        view_dir = view_dir / torch.norm(view_dir, dim=1, keepdim=True)

        # Generate all 8 corners of the volume bounding box in IJK space
        I, J, K = volume_shape
        corners_ijk = torch.tensor([
            [0, 0, 0],
            [I - 1, 0, 0],
            [0, J - 1, 0],
            [I - 1, J - 1, 0],
            [0, 0, K - 1],
            [I - 1, 0, K - 1],
            [0, J - 1, K - 1],
            [I - 1, J - 1, K - 1]
        ], dtype=torch.float32, device=device)  # (8, 3)

        # Transform corners to RAS space
        corners_homog = torch.cat([corners_ijk, torch.ones(8, 1, device=device)], dim=1)  # (8, 4)
        corners_ras = (ijk2ras @ corners_homog.T).T[:, :3]  # (8, 3)

        # Compute distances from camera to each corner projected along view direction
        # Broadcast: cam_pos_ras (B, 1, 3), corners_ras (1, 8, 3)
        to_corners = corners_ras.unsqueeze(0) - cam_pos_ras.unsqueeze(1)  # (B, 8, 3)
        distances = torch.sum(to_corners * view_dir.unsqueeze(1), dim=2)  # (B, 8)

        # Near is the minimum distance, far is the maximum distance
        near = torch.min(distances, dim=1).values - margin  # (B,)
        far = torch.max(distances, dim=1).values + margin  # (B,)

        # Ensure near is positive
        near = torch.clamp(near, min=0.1)

        # Return scalar if input was not batched
        if not is_batched:
            return near.item(), far.item()

        return near, far

    def generate_vtk_ray_samples_ijk(self,
            vol_shape,  # (W, H, D) volume shape in voxels
            ras2ijk: torch.Tensor,  # 4x4 RAS to IJK transform
            view_mat: torch.Tensor,
            fov_y_deg: float,  # vertical FOV in degrees (VTK's ViewAngle)
            img_size: tuple,  # (height_px, width_px) in pixels
            n_depth: int,  # number of samples along each ray
            near: torch.Tensor,  # near plane (distance along view dir, in mm or RAS units)
            far: torch.Tensor  # far plane (distance along view dir)
    ) -> torch.Tensor:
        """
        Generate 3D sample coordinates in IJK space, cast as rays from camera through pixels in the image plane,
        compatible with VTK conventions and an arbitrary RAS2IJK matrix.
        Returns: Tensor of shape (H, W, n_depth, 3), the IJK coordinates along each ray.
        """
        start= timer()
        device = view_mat.device if isinstance(view_mat, torch.Tensor) else "cpu"
        vol_shape = torch.as_tensor(vol_shape, device=device)
        H, W = img_size
        D = n_depth
        B = view_mat.shape[0]
        ras2ijk = ras2ijk.to(device, dtype=torch.float32)

        # World ray directions: rotate by camera orientation (camera-to-world 3x3)
        cam2world = view_mat[..., :3, :3]
        cam_pos = view_mat[..., :3, 3]

        dirs_world = torch.einsum('bij,bhwj->bhwi', cam2world, self.dirs_cam.unsqueeze(0))

        # Ray origin: all start at camera position
        # ray_origins = cam_pos.reshape(-1, 1, 1, 3).expand(-1, H, W, -1)
        ray_origins = cam_pos[:, None, None, None, :]  # (B,1,1,1,3)
        R = ras2ijk[:3, :3]
        t = ras2ijk[:3, 3]

        ijk_coords = torch.empty(B, D, H, W, 3, device=device, dtype=torch.float32)

        # Sample depths along ray (uniform, in world/RAS units)
        depths = near.unsqueeze(1) + (far - near).unsqueeze(1) * torch.linspace(
            0, 1, D, device=device
        ).unsqueeze(0)  # (B, D)
        # depths = torch.linspace(near, far, D, device=device)
        dirs_world = dirs_world.unsqueeze(1)

        # Process depth in batches to reduce memory footprint
        for start in range(0, D, 32):
            end = min(start + 32, D)
            depths_batch = depths[:, start:end]  # (d_batch,)
            depths_batch = depths_batch.view(B, -1, 1, 1, 1)  # (1, d_batch, 1, 1, 1)

            # Compute sample positions for this depth batch
            # Broadcasting: ray_origins (B,H,W,3) + dirs_world (B,H,W,3) * depths_batch (1,d,H,W)
            samples = ray_origins + dirs_world * depths_batch

            # Apply RAS->IJK affine
            ijk_batch = samples @ R.T + t  # (B,d,H,W,3)

            # Normalize to [-1,1]
            ijk_batch = ((2 * ijk_batch) / (vol_shape - 1)) - 1

            # Store in output
            ijk_coords[:, start:end] = ijk_batch

        return ijk_coords  # (B,D,H,W,3)


    def get_camera_matrix(self, look_from):
        nu  = F.normalize(look_from)
        old = torch.tensor([0, 0, 1.0], dtype=nu.dtype, device=nu.device).expand(nu.size(0),-1)
        k  = (old + nu) / 2
        kc = k.unsqueeze(-1) # Column vector
        kr = kc.permute(0,2,1)               # Row vector

        R = 2* (torch.matmul(kc, kr) / (k*k).sum(1).view(-1,1,1)) - torch.eye(3).expand(nu.size(0), -1,-1)
        R[torch.isnan(R).sum(dim=(1,2)).bool()] = torch.flip(torch.eye(3, dtype=nu.dtype, device=nu.device), [0])
        return R

    def apply_poisson(self, transmission: torch.Tensor) -> torch.Tensor:
        if self.i0 is None:
            return transmission

        # Differentiable Gaussian approximation of Poisson
        transmission = torch.clamp(transmission, min=1e-8)
        std = torch.sqrt(transmission / self.i0)
        epsilon = torch.randn_like(transmission)
        noisy_transmission = transmission + std * epsilon
        return torch.clamp(noisy_transmission, 0, 1)

    def forward(self, vol: torch.Tensor, view_mat: torch.Tensor, ras2ijk: torch.Tensor,
                tile_h: int = 256, tile_w: int = 256, triton: bool = False) -> torch.Tensor:
        """
        Args:
            vol:      (B, C, D, H, W)
            view_mat: (B, 4, 4) or (N, 4, 4) where N = B * views_per_vol
            ras2ijk:  (4, 4)
        Returns:
            (N, C, H, W)
        """
        bs = vol.shape[0]
        N = view_mat.shape[0]

        if N < bs:
            raise ValueError(f"view_mat has {N} entries but batch size is {bs}")
        if N > bs and N % bs != 0:
            raise ValueError(f"N ({N}) must be a multiple of batch size ({bs})")

        # Expand density to match number of views
        views_per_vol = N // bs
        density = vol.permute(0, 1, 4, 3, 2)  # (B, C, W, H, D)
        if views_per_vol > 1:
            density = density.repeat_interleave(views_per_vol, dim=0)  # (N, C, W, H, D)
        # N==1 broadcast: expand view_mat to bs
        if N == 1:
            view_mat = view_mat.expand(bs, -1, -1)

        near, far = self.compute_clipping_distances(view_mat, vol.shape[2:], torch.inverse(ras2ijk))

        use_tiling = self.h >= 2 * tile_h or self.w >= 2 * tile_w
        _tile_h = tile_h if use_tiling else None
        _tile_w = tile_w if use_tiling else None

        vol_shape = torch.as_tensor(vol.shape[2:], device=vol.device, dtype=torch.float32)
        if triton:
            render_fn = lambda d, v, n, f: self.triton_raycaster(d, v, n, f, ras2ijk, vol_shape, self.dirs_cam)
        else:
            render_fn = torch.vmap(
                lambda d, v, n, f: self._render_single(d, v, n, f, ras2ijk, vol_shape, _tile_h, _tile_w),
                in_dims=(0, 0, 0, 0),
                randomness='different',  # required if apply_poisson uses torch.randn_like
            )

        return render_fn(density, view_mat, near, far)  # (N, C, H, W)

    def _render_single(
            self,
            density: torch.Tensor,  # (C, W, H, D) — single volume, vmapped over batch
            view_single: torch.Tensor,  # (4, 4)
            near: torch.Tensor,  # scalar
            far: torch.Tensor,  # scalar
            ras2ijk: torch.Tensor,  # (4, 4) shared constant
            vol_shape: torch.Tensor,  # (3,) shared constant
            tile_h: int,
            tile_w: int,
    ) -> torch.Tensor:
        """Render a single (volume, view) pair. Vmapped over batch dim.
        Returns (C, H, W).
        """
        # density = density.contiguous()

        step_size = 0.1 * (far - near) / self.ray_samples

        # --- Ray generation (no loop, no B dim) ---
        cam2world = view_single[:3, :3]
        cam_pos = view_single[:3, 3]

        device = view_single.device if isinstance(view_single, torch.Tensor) else "cpu"
        ras2ijk = ras2ijk.to(device, dtype=torch.float32)

        dirs_world = torch.einsum('ij,hwj->hwi', cam2world, self.dirs_cam)  # (H, W, 3)

        R = ras2ijk[:3, :3]
        t = ras2ijk[:3, 3]

        scale = 2.0 / (vol_shape - 1)  # (3,)

        dirs_ijk = dirs_world @ R.T  # (H, W, 3) @ (3, 3) — small, not (D,H,W,3)
        origin_ijk = cam_pos @ R.T + t  # (3,) — scalar op

        depths = near + (far - near) * torch.linspace(0, 1, self.ray_samples, device=density.device)  # (D,)

        ijk = scale * (origin_ijk + dirs_ijk.unsqueeze(0) * depths[:, None, None, None]) - 1

        # (D, H, W, 3) — no loop, fits in memory per-item
        # samples = cam_pos + dirs_world.unsqueeze(0) * depths[:, None, None, None]
        # ijk = ((2 * (samples @ R.T + t)) / (vol_shape - 1)) - 1  # (D, H, W, 3)

        # grid_sample expects (C, W, H, D) input and (1, D, H, W, 3) grid
        density_b = density.unsqueeze(0)  # (1, C, W, H, D)

        def render_tile(coords_tile):
            # coords_tile: (1, D, h, w, 3)
            dens_tile = F.grid_sample(density_b, coords_tile, align_corners=False)  # (1, C, D, h, w)

            if self.use_beer_lambert:
                if self.scatter:
                    I_out_no_scatter = 1.0 - self.apply_poisson(
                        torch.exp(-torch.sum(dens_tile[:, :-self.scatter_channels] * step_size, dim=2))
                    )
                    I_out, _, _, _ = self.scatter(dens_tile[:, -self.scatter_channels:], step_size)
                    return torch.cat([I_out_no_scatter, 1.0 - I_out], dim=1).squeeze(0)  # (C, h, w)
                else:
                    return (1.0 - self.apply_poisson(
                        torch.exp(-torch.sum(dens_tile * step_size, dim=2))
                    )).squeeze(0)
            else:
                d = self.density_factor * dens_tile / self.ray_samples
                transmission = torch.cumprod(1.0 - d, dim=2)
                weight = d * transmission
                w_sum = weight.sum(dim=2)
                alpha = 1.0 - torch.prod(1.0 - d, dim=2)
                return (weight.sum(dim=2) / (w_sum + 1e-6) * alpha).squeeze(0)  # (C, h, w)

        if tile_h is None or (self.h < 2 * tile_h and self.w < 2 * tile_w):
            # No tiling — single call
            grid = ijk.permute(0, 1, 2, 3).unsqueeze(0)  # (1, D, H, W, 3)
            if self.use_checkpointing and self.training:
                out = cp.checkpoint(render_tile, grid, use_reentrant=False)
            else:
                out = render_tile(grid)
        else:
            # Tile over H and W
            row_outputs = []
            for h_start in range(0, self.h, tile_h):
                h_end = min(h_start + tile_h, self.h)
                col_outputs = []
                for w_start in range(0, self.w, tile_w):
                    w_end = min(w_start + tile_w, self.w)
                    grid = ijk[:, h_start:h_end, w_start:w_end, :].unsqueeze(0)  # (1, D, h, w, 3)
                    if self.use_checkpointing and self.training:
                        tile_out = cp.checkpoint(render_tile, grid, use_reentrant=False)
                    else:
                        tile_out = render_tile(grid)
                    col_outputs.append(tile_out)
                row_outputs.append(torch.cat(col_outputs, dim=-1))
            out = torch.cat(row_outputs, dim=-2)  # (C, H, W)

        return out


if __name__ == '__main__':
    from monai.transforms import LoadImage, EnsureType, EnsureChannelFirst, ScaleIntensity, Compose, Spacing, \
    NormalizeIntensity, CenterSpatialCrop, RandAffine, ScaleIntensityRange
    from matplotlib import pyplot as plt
    from torch.profiler import profile, record_function, ProfilerActivity
    from torch.autograd import gradcheck

    import torch._inductor.config as cfg
    cfg.cpp.vec_isa_ok = False

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


    load_tf = Compose([
        LoadImage(),
        EnsureChannelFirst(),
        Spacing([2.5, 2.5, 3.0]),
        # CenterSpatialCrop([152,152,140]),
        ScaleIntensityRange(-3024, 3024, 0, 1, clip=True),
        EnsureType(),
        # RandAffine(1, translate_range=((0, 0), (0, 0), (-15, 15)), padding_mode='zeros'),
    ])
    vol = load_tf('CTChest.nii.gz').unsqueeze(0)
    # vol.requires_grad_(True)

    tf = [torch.tensor([
        [-3500, 1, 1, 1, 0.0],
        [-200, 1, 1, 1, 0.0],
        [200, 1, 1, 1, 0.05],
        [1535, 1, 1, 1, 0.5],
        [3071, 1, 1, 1, 0.65],
    ]).cuda()]
    tf[0][:, 0] = (tf[0][:, 0] + 3500) / 7000

    xp = tf[0][:, 0]
    yp = tf[0][:, -1]
    a = piecewise_linear_channelwise(vol.cuda(), xp.unsqueeze(0), yp.unsqueeze(0))

    vol = vol.cuda()
    hu = vol * (3024 - (-3524)) + (-3524)
    mu = torch.clamp(0.05 * (1.0 + hu / 800.0), min=0.0)

    ijk2ras = vol.meta['affine']
    ras2ijk = torch.inverse(ijk2ras)

    print(ijk2ras, ras2ijk)

    center = torch.ones(4).double()
    center[:3] = torch.as_tensor(vol.shape[2:]) // 2
    center = ijk2ras @ center
    print(center[:3])

    ren = VolumeRaycaster(scatter=None, resolution=(1024,1024), i0=None, ray_samples=384).cuda().eval()
    # ren = torch.compile(ren)
    # vol = torch.rand(8, 1, 128, 128, 128).cuda()
    # vol.requires_grad_(True)


    view_mat = get_vtk_view_mat((0., 1000, -130.),
                                center[:3],
                                (0.0, 0.0, 1.), device='cuda').unsqueeze(0)

    # view_mat = get_vtk_view_mat((825.512239409456, -13.179125178309306, -150.8782530984467),
    #                             (0.7339149949892914, -69.45105638432082, -184.52283569821498),
    #                             (-0.0018398770598324777, -0.0012575743709173355, 0.9999975166764699))
    # print(view_mat.inverse())

    view_mat = view_mat.repeat(1, 1, 1)

    start = timer()
    out = ren(mu.expand(1, 8, -1, -1, -1), view_mat=view_mat, ras2ijk=ras2ijk)
    torch.cuda.synchronize()
    end = timer()
    print(out.shape, end-start)

    plt.figure()
    plt.imshow(out[0,0].detach().cpu().numpy(), cmap='gray')
    plt.show()

    start = timer()
    out = ren(mu.expand(1, 8, -1, -1, -1), view_mat=view_mat, ras2ijk=ras2ijk)
    torch.cuda.synchronize()
    end = timer()
    print(out.shape, end-start)

    for _ in range(10):
        with torch.no_grad():
            ren(mu.expand(1, 8, -1, -1, -1), view_mat=view_mat, ras2ijk=ras2ijk)
    torch.cuda.synchronize()

    with torch.no_grad():
        out_triton = ren(mu.expand(1, 8, -1, -1, -1), view_mat=view_mat, ras2ijk=ras2ijk, triton=True)
        out_old = ren(mu.expand(1, 8, -1, -1, -1), view_mat=view_mat, ras2ijk=ras2ijk, triton=False)
        print(f'Old and Triton match: {torch.allclose(out_old, out_triton, atol=1e-4)}')
        print(f'Old and Triton Max: {(out_old - out_triton).abs().max()}, Mean: {(out_old - out_triton).abs().mean()}')
        fig, ax = plt.subplots(1,3)
        ax[0].imshow(out_old[0, 0].detach().cpu().numpy(), cmap='gray')
        ax[1].imshow(out_triton[0, 0].detach().cpu().numpy(), cmap='gray')
        im = ax[2].imshow((out_old - out_triton)[0, 0].detach().cpu().numpy(), cmap='coolwarm')
        cbar = fig.colorbar(im, ax=ax[2])
        plt.show()

    mu_prof = mu.expand(1, 8, -1, -1, -1)

    ren.eval()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,  # True adds overhead, only enable for deep dives
    ) as prof:
        for _ in range(10):  # enough iterations to smooth noise
            with record_function("vmap_forward"):
                with torch.no_grad():
                    out = ren.forward(mu_prof, view_mat=view_mat, ras2ijk=ras2ijk)
            torch.cuda.synchronize()  # must sync or CUDA ops appear instant
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=20,
    ))

    # start = timer()
    # out.sum().backward()
    # torch.cuda.synchronize()
    # end = timer()
    # print(end-start)

    print(torch.cuda.memory_summary())

    # plt.figure()
    # plt.imshow(out[0,0].detach().cpu().numpy(), cmap='gray')
    # plt.show()
    #
    # plt.figure()
    # plt.imshow(out2[0,0].detach().cpu().numpy(), cmap='gray')
    # plt.show()

    # plt.figure()
    # plt.imshow(out[0,-1].detach().cpu().numpy() - out[0,0].detach().cpu().numpy(), cmap='gray')
    # plt.show()
    # print(torch.abs(out[0,-1] - out[0,0]).max())

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()

    # ren = VolumeRaycaster(scatter=None, ray_samples=512, resolution=(1024,1024), i0=1e2).cuda().eval()


    print("############ Triton Version ############")

    start = timer()
    out = ren(mu.expand(1, 8, -1, -1, -1), view_mat=view_mat, ras2ijk=ras2ijk, triton=True)
    torch.cuda.synchronize()
    end = timer()
    torch.cuda.synchronize()
    end = timer()
    print(out.shape, end-start)

    with torch.no_grad():
        _ = ren(mu.expand(1, 1, -1, -1, -1), view_mat=view_mat, ras2ijk=ras2ijk, triton=True)
        _ = ren(mu.expand(1, 1, -1, -1, -1), view_mat=view_mat, ras2ijk=ras2ijk, triton=True)
        start = timer()
        out = ren(mu.expand(1, 1, -1, -1, -1), view_mat=view_mat, ras2ijk=ras2ijk, triton=True)
        torch.cuda.synchronize()
        end = timer()
        print(out.shape, end-start)

    plt.figure()
    plt.imshow(out[0,0].detach().cpu().numpy(), cmap='gray')
    plt.show()

    for _ in range(10):
        with torch.no_grad():
            ren(mu.expand(1, 8, -1, -1, -1), view_mat=view_mat, ras2ijk=ras2ijk, triton=True)
    torch.cuda.synchronize()

    ren.eval()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,  # True adds overhead, only enable for deep dives
    ) as prof:
        for _ in range(10):  # enough iterations to smooth noise
            with record_function("triton_forward"):
                with torch.no_grad():
                    out = ren.forward(mu_prof, view_mat=view_mat, ras2ijk=ras2ijk, triton=True)
            torch.cuda.synchronize()  # must sync or CUDA ops appear instant
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=20,
    ))

    # start = timer()
    # out.sum().backward()
    # torch.cuda.synchronize()
    # end = timer()
    # print(end-start)


    print(torch.cuda.memory_summary())

    with torch.no_grad():
        with torch.autocast(device_type="cuda"):
            _ = ren(mu.expand(1, 1, -1, -1, -1), view_mat=view_mat, ras2ijk=ras2ijk, triton=True)
            _ = ren(mu.expand(1, 1, -1, -1, -1), view_mat=view_mat, ras2ijk=ras2ijk, triton=True)
            start = timer()
            out_triton = ren(mu.expand(1, 1, -1, -1, -1), view_mat=view_mat, ras2ijk=ras2ijk, triton=True).float()
            torch.cuda.synchronize()
            end = timer()
            print("Triton fp16", end - start)

            start = timer()
            out_old = ren(mu.expand(1, 8, -1, -1, -1), view_mat=view_mat, ras2ijk=ras2ijk, triton=False).float()
            torch.cuda.synchronize()
            end = timer()
            print("Orig fp16", end - start)

        print(f'Old and Triton match fp16: {torch.allclose(out_old, out_triton, atol=1e-4)}')
        print(
            f'Old and Triton Max fp16: {(out_old - out_triton).abs().max()}, Mean: {(out_old - out_triton).abs().mean()}')

        out_triton_fp32 = ren(mu.expand(1, 1, -1, -1, -1), view_mat=view_mat, ras2ijk=ras2ijk, triton=True).float()
        out_old_fp32 = ren(mu.expand(1, 8, -1, -1, -1), view_mat=view_mat, ras2ijk=ras2ijk, triton=False).float()

        print(f'Old match fp32/fp16: {torch.allclose(out_old, out_old_fp32, atol=1e-4)}')
        print(f'Triton match fp32/fp16: {torch.allclose(out_triton_fp32, out_triton, atol=1e-4)}')

        print(
            f'Triton Max fp16/32: {(out_triton_fp32 - out_triton).abs().max()}, Mean: {(out_triton_fp32 - out_triton).abs().mean()}')

    ren = VolumeRaycaster(scatter=None, resolution=(512,512), i0=None).cuda()

    torch.manual_seed(0)
    mu = torch.randn(2, 8, 24, 16, device='cuda', dtype=torch.float64, requires_grad=True)

    # isolate the triton Function; gradcheck the full expanded graph in a second pass if you want
    ok = gradcheck(
        lambda m: ren(
            m.expand(1, 2, -1, -1, -1),
            view_mat=view_mat, ras2ijk=ras2ijk, triton=True,
        ),
        (mu,),
        eps=1e-6,
        atol=1e-4,  # looser than the fp64 default 1e-5 to absorb kernel fp noise
        rtol=1e-3,
        nondet_tol=1e-6,  # tl.atomic_add is order-nondeterministic
        fast_mode=True,
    )
    print("gradcheck passed:", ok)


