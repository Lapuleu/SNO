import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import scipy
import matplotlib.pyplot as plt
import os
import time
from timeit import default_timer
from D1utilities3 import *
from D1Adam import Adam
import time
import math
import scipy.special as sp
import scipy.io as sio
import warnings
from mpl_toolkits.axes_grid1 import make_axes_locatable
warnings.simplefilter('ignore', np.exceptions.RankWarning)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# ====================================
# Data Gen
# ====================================

class Sumudu_Transform(nn.Module):
    def __init__(self, in_channels, out_channels, degree, width, s):
        super().__init__()
        self.degree = degree
        self.width = width
        self.s = s
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Random parameters
        self.flip = (-1)**torch.randint(0, 2, (in_channels, out_channels))
        self.scale = 1.0 / (in_channels * out_channels)
        self.weight1 = nn.Parameter(self.scale * torch.rand((in_channels, out_channels), dtype=torch.float64))
        self.weight2 = nn.Parameter(self.scale * torch.rand((in_channels, out_channels), dtype=torch.float64))

        dtype = torch.float64
        t = torch.linspace(0, 1, s, dtype=dtype)
        raw_fact = torch.exp(torch.lgamma(torch.arange(1, s+1, dtype=dtype)+1))
        fact = raw_fact / raw_fact.max()
        self.register_buffer('factorial', fact.flip(0))
        vander = torch.vander(t, N=degree, increasing=False)
        pinv = torch.linalg.pinv(vander, rcond=1e-4)
        self.register_buffer('x_grid', t)
        self.register_buffer('factorial', fact.flip(0))
        self.register_buffer('vander_pinv', pinv)
        self.register_buffer('vander', vander)

    def coefficient_training(self, input, degree):
        # reshape to [batch*nx*ny, s]
        B,N,M,S = input.shape
        y = input.reshape(-1, S).double()  # [B*N*M, s]
        # polyfit using precomputed pinv (degree-1 is already handled)
        coef = (self.vander_pinv @ y.double().T).T   # [B*N*M, degree]
        return coef.reshape(B,N,M,degree)

    def transform(self, input):
        return input * self.factorial[:input.shape[3]]

    def inverse_transform(self, input):
        return input / self.factorial[:input.shape[3]]

    def approximate_sum(self, width, input):
        # Evaluate polynomial at x_grid using torch.vander
        B,N,M,D = input.shape
        V = self.vander[:, :D]           # [s, degree]
        # [B,N,M,s] = einsum over degree
        return torch.einsum('b n m d, s d -> b n m s', input, V)

    def weight_mul(self, input):
        # input: [B, in_channels, width, s]
        # einsum over last axis, same as original loops
        t1 = torch.einsum("b i x s, i o -> b o x s", input, self.weight1)
        t2 = torch.einsum("b i t s, i o -> b o t s", input, self.weight2)
        return t1 + t1 * t2

    def forward(self, x):
        B, C, Nx, Ny = x.shape

        # reshape to [B*Nx*Ny, C]
        x = x.permute(0, 2, 3, 1).contiguous()   # [B, Nx, Ny, C]

        # >>> IMPORTANT: if C != s, don't call coefficient_training here <<<
        if x.shape[-1] == self.vander.shape[0]:  # only when last dim == s
            x = self.coefficient_training(x, self.degree)
            x = self.transform(x)
            x = self.approximate_sum(self.width, x)
            x = self.weight_mul(x)
            x = self.coefficient_training(x, self.degree)
            x = self.inverse_transform(x)
            x = self.approximate_sum(self.width, x)

        # back to [B, C, Nx, Ny]
        return x.permute(0, 3, 1, 2).float()


class SNO2d(nn.Module):
    def __init__(self, degree, width, s, activation):
        super().__init__()
        self.width = width
        self.degree = degree
        self.activation = activation
        self.fc0 = nn.Linear(3, width)
        self.conv0 = Sumudu_Transform(width, width, degree, width, s)
        self.w0 = nn.Conv2d(width, width, 1)
        self.norm = nn.InstanceNorm2d(width)
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # x: [B, Nx, Ny] → [B, Nx, Ny, 1]
        
        if x.dim() == 3:
            x = x.unsqueeze(-1)
        spatial_shape = x.shape[1:-1]  # (Nx, Ny)
        grid = self.get_grid(*spatial_shape, device=x.device)
        grid = grid.repeat(x.shape[0], *[1 for _ in range(grid.dim() - 1)])  # repeat batch dimension
        print("x.shape:", x.shape)
        print("grid.shape:", grid.shape)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x).permute(0, 3, 1, 2)
        x1 = self.norm(self.conv0(self.norm(x)))
        x2 = self.w0(x)
        x = x1 + x2
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, *shape, device):

        dim = len(shape)

        if dim == 1:
            # 1D grid: x in [0,1]
            Nx = shape[0]
            gridx = torch.linspace(0, 1, Nx, device=device)
            grid = gridx.unsqueeze(-1).unsqueeze(0)  # [1, Nx, 1]
            return grid

        elif dim == 2:
            # 2D grid: (x,y) in [0,1]^2
            Nx, Ny = shape
            gridx = torch.linspace(0, 1, Nx, device=device)
            gridy = torch.linspace(0, 1, Ny, device=device)
            gridx, gridy = torch.meshgrid(gridx, gridy, indexing='ij')
            grid = torch.stack((gridx, gridy), dim=-1).unsqueeze(0)  # [1, Nx, Ny, 2]
            return grid

        elif dim == 3:
    # 3D grid: (x,y,z) in [0,1]^3
            Nx, Ny, Nz = shape
            gridx = torch.linspace(0, 1, Nx, device=device)
            gridy = torch.linspace(0, 1, Ny, device=device)
            gridz = torch.linspace(0, 1, Nz, device=device)
            gridx, gridy, gridz = torch.meshgrid(gridx, gridy, gridz, indexing='ij')
            grid = torch.stack((gridx, gridy, gridz), dim=-1).unsqueeze(0)  # [1, Nx, Ny, Nz, 3]
            return grid
        else:
            raise ValueError(f"Unsupported grid dimensionality: {dim}")

# ====================================
#  Define parameters and Load data
# ====================================

class SNO2dmain():
    def __init__(self):
        pass

    def supRes(self):
        save_index = 1
        current_directory = os.getcwd()
        case = "Case_SReDiffSR_"
        folder_index = str(save_index)
        results_dir = "/" + case + folder_index + "/"
        save_results_to = current_directory + results_dir
        os.makedirs(save_results_to, exist_ok=True)

        # --- Load data ---
        reader = MatReader('/workspace/Data/ReacDiff/data.mat')
        x_test = reader.read_field('f_test')
        y_test = reader.read_field('u_test')

        x_test = x_test.reshape(130,x_test.shape[1],x_test.shape[2],1)
        x_test = torch.tensor(x_test, dtype=torch.float32, device=device)
        y_test = torch.tensor(y_test, dtype=torch.float32, device=device)

        # --- Load trained model ---
        model = torch.load("/workspace/Data/ReacDiff/Wave_states", map_location=device, weights_only=False)
        model.eval()

        s = 40     # base visualization resolution
        new_s = 160  # super-res inference resolution
        B = y_test.shape[0]

        print(f"[Reconfiguring model for {new_s}x{new_s/2}]")
        print("[Running zero-shot super-resolution]...")

        print("[Running base resolution inference]...")

        # Create 2D grid for base resolution
        model.conv0.s = s
        with torch.no_grad():
            y_pred_base = model(x_test)  # shape [B, s, s, 1]
        y_pred_base = y_pred_base.squeeze(-1).cpu()

        model.conv0.s = new_s
        with torch.no_grad():
            # Predict at high resolution
            y_pred_hr = model(x_test).permute(0, 3, 1, 2)  # [B, 1, Nx, Ny]
            y_pred_hr = F.interpolate(
                y_pred_hr, size=(new_s, 20), mode='bilinear', align_corners=False
            ).squeeze(1)
            y_true_hr = y_test#F.interpolate(
                #y_test.unsqueeze(1), size=(new_s, new_s), mode='bilinear', align_corners=False
            #).squeeze(1)
            y_pred_lr = F.interpolate(y_pred_hr.unsqueeze(1), size=(s, 20), mode='bilinear', align_corners=False).squeeze(1)
        mse_val = F.mse_loss(y_pred_lr, y_true_hr).item()
        l2_val = torch.norm(y_pred_lr - y_true_hr).item()
        print(f"MSE={mse_val:.4e}, L2={l2_val:.4e}")

        # --- Downscale all fields to base resolution for visualization ---
        #y_true_lr = F.interpolate(y_true_hr.unsqueeze(1), size=(s, s), mode='bilinear', align_corners=False).squeeze(1)
        #y_base_lr = F.interpolate(y_pred_base.unsqueeze(1), size=(s, s), mode='bilinear', align_corners=False).squeeze(1)

        y_true = y_true_hr.cpu().numpy() #y_true_lr.cpu().numpy()
        y_pred = y_pred_lr.cpu().numpy()
        y_base = y_pred_base.cpu().numpy()#y_base_lr.cpu().numpy()

        # --- Compute errors at 200x200 (for quantitative metrics) ---
        err_super_hr = torch.abs(y_pred_lr - y_true_hr)
        y_true_hr = y_true_hr.cpu()
        err_base_hr = torch.abs(y_pred_base - y_true_hr)

        # --- Downscale errors for visualization ---
        err_super = err_super_hr#F.interpolate(err_super_hr.unsqueeze(1), size=(s, s), mode='bilinear', align_corners=False).squeeze(1).cpu().numpy()
        err_base = err_base_hr#F.interpolate(err_base_hr.unsqueeze(1), size=(s, s), mode='bilinear', align_corners=False).squeeze(1).cpu().numpy()

        # === Visualization block ===
        print("[Generating visualizations...]")

        def to_numpy_safe(t):
            """Convert PyTorch tensor or NumPy array to a 2D NumPy array."""
            if isinstance(t, torch.Tensor):
                t = t.detach().cpu().squeeze().numpy()
            elif isinstance(t, np.ndarray):
                t = np.squeeze(t)
            else:
                raise TypeError(f"Unsupported type: {type(t)}")
            return t

        y_true = to_numpy_safe(y_true_hr[0])
        y_base_pred = to_numpy_safe(y_base[0])
        y_super_pred = to_numpy_safe(y_pred_hr[0])

        # Downscale everything to 50×50 for visualization
        def downscale_to_50(tensor):
            if isinstance(tensor, np.ndarray):
                tensor = torch.tensor(tensor, dtype=torch.float32)
            t = tensor.unsqueeze(0).unsqueeze(0)  # [1,1,Nx,Ny]
            t = F.interpolate(t, size=(s, 20), mode='bilinear', align_corners=False)
            return t.squeeze().cpu().numpy()

        y_true_50 = downscale_to_50(y_true)
        y_base_50 = downscale_to_50(y_base_pred)
        y_super_50 = downscale_to_50(y_super_pred)

        # Compute error maps
        err_base = np.abs(y_base_50 - y_true_50)
        err_super = np.abs(y_super_50 - y_true_50)

        # Shared value and error scales
        vmin = min(y_true_50.min(), y_base_50.min(), y_super_50.min())
        vmax = max(y_true_50.max(), y_base_50.max(), y_super_50.max())
        errmax = max(err_base.max(), err_super.max())

        y_true_sample = y_true_50
        y_base_sample = y_base_50
        y_super_sample = y_super_50
        err_base_sample = np.abs(y_base_sample - y_true_sample)
        err_super_sample = np.abs(y_super_sample - y_true_sample)

        nx, ny = y_true_sample.shape
        extent = [0, nx, 0, ny]  # left, right, bottom, top

        # === Create 2x3 heatmap figure ===
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))

        # Use interpolation='bicubic' or 'bilinear' to avoid disconnected squares
        interp_method = 'nearest'

        # Row 1: Fields
        im0 = axes[0,0].imshow(y_true_sample, cmap='viridis', vmin=vmin, vmax=vmax,
                                origin='lower', extent=extent, aspect='auto', interpolation=interp_method)
        axes[0,0].set_title("Ground Truth")

        im1 = axes[0,1].imshow(y_base_sample, cmap='viridis', vmin=vmin, vmax=vmax,
                                origin='lower', extent=extent, aspect='auto', interpolation=interp_method)
        axes[0,1].set_title("Base Prediction")

        im2 = axes[0,2].imshow(y_super_sample, cmap='viridis', vmin=vmin, vmax=vmax,
                                origin='lower', extent=extent, aspect='auto', interpolation=interp_method)
        axes[0,2].set_title("Super-Resolution Prediction")

        # Row 2: Errors
        im3 = axes[1,0].imshow(err_base_sample, cmap='inferno', vmin=0, vmax=errmax,
                                origin='lower', extent=extent, aspect='auto', interpolation=interp_method)
        axes[1,0].set_title("Base Error")

        im4 = axes[1,1].imshow(err_super_sample, cmap='inferno', vmin=0, vmax=errmax,
                                origin='lower', extent=extent, aspect='auto', interpolation=interp_method)
        axes[1,1].set_title("Super-Resolution Error")

        axes[1,2].axis("off")  # empty subplot

        # Add vertical colorbars (as before)
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        for ax, im, label in zip([axes[0,0], axes[0,1], axes[0,2]], [im0, im1, im2], ["Field Value"]*3):
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            plt.colorbar(im, cax=cax, label=label)

        for ax, im, label in zip([axes[1,0], axes[1,1]], [im3, im4], ["Absolute Error"]*2):
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            plt.colorbar(im, cax=cax, label=label)

        # Gridlines and ticks
        for ax in axes.flatten():
            ax.set_xticks(np.linspace(0, nx, 6))
            ax.set_yticks(np.linspace(0, ny, 6))
            ax.grid(True, color='white', alpha=0.3)
            ax.set_xlabel("Time(s)")
            ax.set_ylabel("Location(m)")

        plt.tight_layout()
        plt.savefig(os.path.join(save_results_to, 'heatmaps_2d.png'), dpi=300)
        plt.close(fig)

        # === Line plots ===
        center_idx = 25

        # Center line
        plt.figure(figsize=(8, 5))
        plt.plot(y_true_50[center_idx, :], label="Ground Truth", lw=2)
        plt.plot(y_base_50[center_idx, :], label="Base Prediction", lw=2)
        plt.plot(y_super_50[center_idx, :], label="Super-Resolution", lw=2)
        plt.xlabel("X Index")
        plt.ylabel("Amplitude")
        plt.title("Center Line (Row 25)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_results_to, f'line_center.png'), dpi=300)
        plt.close()

        # Mean line
        plt.figure(figsize=(8, 5))
        plt.plot(y_true_50.mean(axis=0), label="Ground Truth", lw=2)
        plt.plot(y_base_50.mean(axis=0), label="Base Prediction", lw=2)
        plt.plot(y_super_50.mean(axis=0), label="Super-Resolution", lw=2)
        plt.xlabel("X Index")
        plt.ylabel("Amplitude (Row Mean)")
        plt.title("Mean Line Across All Rows")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_results_to, f'line_mean.png'), dpi=300)
        plt.close()

        print("[Saved visualizations: heatmaps_2d.png, line_center.png, line_mean.png]")
        # --- Save metrics and outputs ---
        np.savetxt(os.path.join(save_results_to, "superres_metrics.txt"),
                   np.array([[mse_val, l2_val]]), header="MSE   L2")
        sio.savemat(os.path.join(save_results_to, "wave_states_superres_2d.mat"), {
            'y_pred_hr': y_pred_hr.cpu().numpy(),
            'y_true_hr': y_true_hr.cpu().numpy(),
            'y_pred_lr': y_pred,
            'y_true_lr': y_true
        })

        print(f"\n✅ Zero-Shot 2D Super-Resolution Complete")
        print(f"   - Inference resolution: {new_s}×{new_s}")
        print(f"   - Visualization resolution: {s}×{s}")
        print(f"   - Results saved in: {save_results_to}")

SNO2dmain().supRes()
