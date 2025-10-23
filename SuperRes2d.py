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
        case = "Case_SBeamSR_"
        folder_index = str(save_index)
        results_dir = "/" + case + folder_index + "/"
        save_results_to = current_directory + results_dir
        os.makedirs(save_results_to, exist_ok=True)

        # --- Load data ---
        reader = MatReader('/workspace/Data/Beam/data.mat')
        x_test = reader.read_field('f_test')  # [B, Nx, Ny]
        y_test = reader.read_field('u_test')  # [B, Nx, Ny]

        x_test = x_test.to(torch.float32).to(device)
        y_test = y_test.to(torch.float32).to(device)

        # --- Load trained model ---
        model = torch.load("/workspace/Data/Beam/Wave_states", map_location=device, weights_only=False)
        model.eval()

        s = 50     # base visualization resolution
        new_s = 200  # super-res inference resolution
        B = y_test.shape[0]
        x_vals = np.linspace(0, 1, s)
        y_vals = np.linspace(0, 1, s)

        print(f"[Reconfiguring model for {new_s}x{new_s}]")
        print("[Running zero-shot super-resolution]...")

        with torch.no_grad():
            # Predict at high resolution
            y_pred_hr = model(x_test).permute(0, 3, 1, 2)  # [B, 1, Nx, Ny]
            y_pred_hr = F.interpolate(
                y_pred_hr, size=(new_s, new_s), mode='bilinear', align_corners=False
            ).squeeze(1)
            y_true_hr = F.interpolate(
                y_test.unsqueeze(1), size=(new_s, new_s), mode='bilinear', align_corners=False
            ).squeeze(1)

        mse_val = F.mse_loss(y_pred_hr, y_true_hr).item()
        l2_val = torch.norm(y_pred_hr - y_true_hr).item()
        print(f"MSE={mse_val:.4e}, L2={l2_val:.4e}")

        # --- Downscale all fields to base resolution for visualization ---
        y_true_lr = F.interpolate(y_true_hr.unsqueeze(1), size=(s, s), mode='bilinear', align_corners=False).squeeze(1)
        y_pred_lr = F.interpolate(y_pred_hr.unsqueeze(1), size=(s, s), mode='bilinear', align_corners=False).squeeze(1)
        y_base_lr = F.interpolate(y_test.unsqueeze(1), size=(s, s), mode='bilinear', align_corners=False).squeeze(1)

        y_true = y_true_lr.cpu().numpy()
        y_pred = y_pred_lr.cpu().numpy()
        y_base = y_base_lr.cpu().numpy()

        # --- Compute errors at 200x200 (for quantitative metrics) ---
        err_super_hr = torch.abs(y_pred_hr - y_true_hr)
        err_base_hr = torch.abs(
            F.interpolate(y_test.unsqueeze(1), size=(new_s, new_s), mode='bilinear', align_corners=False).squeeze(1)
            - y_true_hr
        )

        # --- Downscale errors for visualization ---
        err_super = F.interpolate(err_super_hr.unsqueeze(1), size=(s, s), mode='bilinear', align_corners=False).squeeze(1).cpu().numpy()
        err_base = F.interpolate(err_base_hr.unsqueeze(1), size=(s, s), mode='bilinear', align_corners=False).squeeze(1).cpu().numpy()

        # --- Visualization Loop ---
        for idx in range(1):
            mean_true = y_true[idx].mean(axis=0)
            mean_base = y_base[idx].mean(axis=0)
            mean_super = y_pred[idx].mean(axis=0)
            mean_err_base = err_base[idx].mean(axis=0)
            mean_err_super = err_super[idx].mean(axis=0)

            # Line plot comparison
            plt.figure(figsize=(10, 6))
            plt.plot(x_vals, mean_true, 'k--', label='Ground Truth')
            plt.plot(x_vals, mean_base, 'b', label='Base (50x50)')
            plt.plot(x_vals, mean_super, 'r', label='Super-Res (200x200 → 50x50)')
            plt.title(f"Sample #{idx} — Mean Field Comparison")
            plt.xlabel("X-axis")
            plt.ylabel("Mean Field Value")
            plt.legend()
            plt.grid(alpha=0.4, linestyle='--')
            plt.xticks(np.linspace(0, 1, 6))
            plt.tight_layout()
            plt.savefig(os.path.join(save_results_to, f'sample_{idx}_mean_line.png'), dpi=300)
            plt.close()

            # Error line plot
            plt.figure(figsize=(10, 4))
            plt.plot(x_vals, mean_err_base, 'b', label='|Error| Base vs GT')
            plt.plot(x_vals, mean_err_super, 'r', label='|Error| Super-Res vs GT')
            plt.title(f"Sample #{idx} — Mean Absolute Error")
            plt.xlabel("X-axis")
            plt.ylabel("|Error|")
            plt.legend()
            plt.grid(alpha=0.4, linestyle='--')
            plt.xticks(np.linspace(0, 1, 6))
            plt.tight_layout()
            plt.savefig(os.path.join(save_results_to, f'sample_{idx}_mean_error.png'), dpi=300)
            plt.close()

            # --- Heatmaps (same color scale) ---
            vmin_field = min(y_true[idx].min(), y_base[idx].min(), y_pred[idx].min())
            vmax_field = max(y_true[idx].max(), y_base[idx].max(), y_pred[idx].max())
            err_vmax = max(err_base[idx].max(), err_super[idx].max())

            fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
            extent = [0, 1, 0, 1]

            im0 = axes[0].imshow(y_true[idx], origin='lower', extent=extent, cmap='viridis',
                                 vmin=vmin_field, vmax=vmax_field)
            axes[0].set_title('Ground Truth (Downscaled)')
            plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

            im1 = axes[1].imshow(y_base[idx], origin='lower', extent=extent, cmap='viridis',
                                 vmin=vmin_field, vmax=vmax_field)
            axes[1].set_title('Base Prediction (50x50)')
            plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

            im2 = axes[2].imshow(y_pred[idx], origin='lower', extent=extent, cmap='viridis',
                                 vmin=vmin_field, vmax=vmax_field)
            axes[2].set_title('Super-Res (200x200 → 50x50)')
            plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

            im3 = axes[3].imshow(err_super[idx], origin='lower', extent=extent, cmap='magma',
                                 vmin=0, vmax=err_vmax)
            axes[3].set_title('|Error| Super-Res vs GT (Downscaled)')
            plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

            for ax in axes:
                ax.set_xlabel("X-axis")
                ax.set_ylabel("Y-axis")
                ax.set_xticks(np.linspace(0, 1, 6))
                ax.set_yticks(np.linspace(0, 1, 6))
                ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)

            plt.tight_layout()
            plt.savefig(os.path.join(save_results_to, f'sample_{idx}_heatmaps.png'), dpi=300)
            plt.close()

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