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
    def __init__(self, in_channels, out_channels, degree, width, s, device=None):
        super().__init__()
        self.degree = degree
        self.width = width
        self.s = s
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.scale = 1.0 / (in_channels * out_channels)
        self.weight1 = nn.Parameter(self.scale * torch.rand((in_channels, out_channels), dtype  =torch.float64))
        self.flip = (-1) ** torch.randint(0, 2, (in_channels, out_channels))

        dtype = torch.float64
        x_grid = torch.linspace(0.0, 1.0, steps=s, dtype=dtype)  # [s]
        vander = torch.vander(x_grid, N=degree, increasing=False)  # [s, degree]
        vander_pinv = torch.linalg.pinv(vander, rcond=1e-6)
        idx = torch.arange(1, s + 1, dtype=dtype)
        fact = torch.exp(torch.lgamma(idx + 1.0))
        self.register_buffer('x_grid', x_grid)
        self.register_buffer('vander', vander)
        self.register_buffer('vander_pinv', vander_pinv)
        self.register_buffer('factorial', fact)  # factorial[n-1] == n!

    def coefficient_training(self, input):
        B, C, D = input.shape
        y = input.reshape(-1, D).double()  # [B*C, D]
        coef = (self.vander_pinv[:, :D] @ y.T).T   # use only first D cols of vander_pinv if D < s
        coef = coef.reshape(B, C, -1)  # [B, C, degree]
        return coef

    def transform(self, input):
        fact = self.factorial[:D].view(1, 1, -1)
        return input.double() * fact

    def inverse_transform(self, input):
        D = input.shape[2]
        fact = self.factorial[:D].view(1, 1, -1)
        return input.double() / fact

    def approximate_sum(self, width, input):
        B, C, degree = input.shape
        V = self.vander[:, :degree]   # [s, degree]
        out = input.reshape(-1, degree).double() @ V.T   # [B*C, s]
        out = out.reshape(B, C, self.s)
        return out

    def weight_mul(self, input):
        return torch.einsum("b i x, i o -> b o x", input.double(), self.weight1)

    def forward(self, x):
        B, C, D = x.shape
        x = self.coefficient_training(x)    
        x = self.transform(x)                       
        x = self.approximate_sum(self.width, x)  
        x = self.weight_mul(x)
        x = self.coefficient_training(x) 
        x = self.inverse_transform(x)
        x = self.approximate_sum(self.width, x)  

        return x.float()
    
class SNO1d(nn.Module):
    def __init__(self, degree, width, s, activation):
        super(SNO1d, self).__init__()
        self.degree = degree
        self.s = s
        self.width = width
        self.activation = activation

        self.fc0 = nn.Linear(1, self.width) 
        self.layer_norm = nn.LayerNorm(self.width)

        self.conv0 = Sumudu_Transform(self.width, self.width, self.degree, self.width, self.s)
        
        self.w0 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

        self.bn0 = torch.nn.BatchNorm1d(self.width)

    def forward(self,x):
        #grid = self.get_grid(x.shape, x.device)
        #x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        if self.s != x.shape[-1]:
            x = F.interpolate(x, size=self.s, mode='linear', align_corners=False)
        x1 = self.conv0(x)
        x2 = self.w0(x)
        if x1.shape[-1] != x2.shape[-1]:
            x2 = F.interpolate(x2, size=x1.shape[-1], mode='linear', align_corners=False)
        x = self.bn0(x1 + x2)
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.relu6(x)
        x = self.fc2(x)
        return x
  
    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.cfloat)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)

# ====================================
#  Define parameters and Load data
# ====================================

class SNO1dmain():
    def __init__(self):
        pass
    def supRes(self):
        # Configuration
        save_index = 1   
        current_directory = os.getcwd()
        case = "Case_SPen05SR_"
        folder_index = str(save_index)

        results_dir = "/" + case + folder_index +"/"
        save_results_to = current_directory + results_dir
        if not os.path.exists(save_results_to):
            os.makedirs(save_results_to)

        # Load test data (forcing term)
        reader = MatReader('/workspace/Data/Pen05/data.mat')
        x_test = reader.read_field('f_test')
        y_test = reader.read_field('u_test')
        grid_x_test = reader.read_field('x_test')

        x_test = x_test.reshape(x_test.shape[0], 2048, 1)
        x_test = x_test.detach().clone().to(torch.float32).to(device)
        x_low = torch.linspace(0, 1, 2048).view(1, 1, -1).to(device)
        # Load model
        model = torch.load("/workspace/Data/Pen05/Wave_states", map_location=device, weights_only=False)
        model.eval()
        with torch.no_grad():
            y_low = model(x_test)
            y_low = y_low.permute(0, 2, 1)
            y_base_up = F.interpolate(y_low, size=8192, mode='linear', align_corners=False)
            y_base_up = y_base_up.permute(0, 2, 1)
        # Define higher-resolution grid
        s = 2048
        new_s = 8192  # <-- super-res factor (e.g., 4× original 2048)
        dtype = torch.float64
        
        x_grid_hr = torch.linspace(0, 1, steps=new_s, dtype=dtype)
        vander_hr = torch.vander(x_grid_hr, N=model.conv0.degree, increasing=False)
        vander_pinv_hr = torch.linalg.pinv(vander_hr, rcond=1e-6)

        # Replace buffers in Sumudu_Transform
        with torch.no_grad():
            x_grid_hr = x_grid_hr.to(device)
            vander_hr = vander_hr.to(device)
            vander_pinv_hr = vander_pinv_hr.to(device)

            model.conv0.register_buffer('x_grid', x_grid_hr)
            model.conv0.register_buffer('vander', vander_hr)
            model.conv0.register_buffer('vander_pinv', vander_pinv_hr)
            model.conv0.s = new_s
            model.conv0.width = new_s

        print(f"[Model reconfigured] New spatial resolution: {new_s}")

        # Run zero-shot super resolution
        print(f"[Running zero-shot super resolution inference] ...")

        # Measure inference time
        start_time = time.time()
        with torch.no_grad():
            y_pred_hr = model(x_test).cpu()
        elapsed_time = time.time() - start_time

        myloss = LpLoss(size_average=False)

        # Interpolate y_test to match high-res grid (if necessary)
        y_test_torch = y_test.detach().clone().to(torch.float32)
        y_test_interp = F.interpolate(
            y_test_torch.unsqueeze(1),  # [B, 1, 2048]
            size=y_pred_hr.shape[1],    # match 8192
            mode="linear",
            align_corners=False
        ).squeeze(1)                    # [B, 8192]

        # Compute losses
        # Match dimensions
        y_pred_hr = y_pred_hr.squeeze(-1)          # [B, 8192]
        y_test_interp = y_test_interp.to(y_pred_hr.device)

        mse = F.mse_loss(y_pred_hr, y_test_interp)
        l2 = myloss(y_pred_hr, y_test_interp)


        print("\n=============================")
        print("Zero-Shot Super-Resolution Results")
        print("=============================") 
        print(f"High-res points: {y_pred_hr.shape[1]}")
        print(f"Time elapsed: {elapsed_time:.4f} s")
        print(f"MSE: {mse.item():.4e}")
        print(f"L2 error: {l2.item():.4e}")
        print("=============================\n")

        y_base_up = y_base_up.squeeze(-1)
        y_pred_hr = y_pred_hr.cpu()
        y_base_up = y_base_up.cpu()
        print("y_pred_hr:", y_pred_hr.shape)
        print("y_base_up:", y_base_up.shape)

        mse1 = F.mse_loss(y_pred_hr, y_base_up).item()
        rel_err = (torch.norm(y_pred_hr - y_base_up) / torch.norm(y_pred_hr)).item()

        print(f"[Metrics] MSE = {mse1:.4e}, Relative Error = {rel_err:.4e}")

        x_highres = x_grid_hr.cpu().numpy()
        y_pred_hr_np = y_pred_hr.cpu().numpy() if torch.is_tensor(y_pred_hr) else y_pred_hr
        y_test_interp_np = y_test_interp.cpu().numpy()

        plt.figure(figsize=(12, 5))

        # Super-res prediction (red)
        plt.subplot(1, 2, 1)
        plt.plot(
            torch.linspace(0, 1, y_pred_hr.shape[1]),
            y_pred_hr[0].cpu(),
            label=f'Super-res ({new_s})'
        )

        # Upscaled base (dashed)
        plt.plot(
            torch.linspace(0, 1, y_base_up.shape[1]),
            y_base_up[0].cpu(),
            '--',
            label=f'Upscaled base ({s})'
        )
        plt.legend()
        plt.xlabel("Spatial coordinate (x)")
        plt.ylabel("Field value u(x)")
        plt.title("Super-Resolution vs. Upscaled Base Output")

        plt.subplot(1, 2, 2)
        plt.plot(
            torch.linspace(0, 1, y_base_up.shape[1]),
            (y_pred_hr[0].cpu() - y_base_up[0].cpu()).abs(),
            '--',
        )
        plt.title("Absolute Error (|Super - Upscaled Base|)")
        plt.tight_layout()
        plt.savefig(os.path.join(save_results_to, f'basic_v_supres.png'), dpi=300)
        plt.show()

        # Plot one random test example
        idx = 0
        plt.figure(figsize=(8,4))
        plt.plot(x_highres, y_test_interp_np[idx], 'k--', label='Ground Truth (Interpolated)', linewidth=1.5)
        plt.plot(x_highres, y_pred_hr_np[idx], 'r', label='Model Prediction (Super-Res)', linewidth=1.2)
        plt.title(f'Zero-Shot Super-Resolution (Test sample #{idx})')
        plt.xlabel('time (t)')
        plt.ylabel('u(x)')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_results_to, f'superres_sample_{idx}.png'), dpi=300)
        plt.show()

        # (Optional) Compare several random samples
        """num_samples = min(3, y_pred_hr_np.shape[0])
        sample_indices = np.random.choice(y_pred_hr_np.shape[0], num_samples, replace=False)
        fig, axes = plt.subplots(num_samples, 1, figsize=(8, 3*num_samples), sharex=True)
        for i, ax in enumerate(axes):
            idx = sample_indices[i]
            ax.plot(x_highres, y_test_interp_np[idx], 'k--', label='True')
            ax.plot(x_highres, y_pred_hr_np[idx], 'r', label='Pred')
            ax.set_title(f'Sample #{idx}')
            ax.legend()
        plt.xlabel('x')
        plt.tight_layout()
        plt.savefig(os.path.join(save_results_to, 'superres_multi_samples.png'), dpi=300)
        plt.show()

        if y_pred_hr_np.shape != y_test_interp_np.shape:
            min_len = min(y_pred_hr_np.shape[1], y_test_interp_np.shape[1])
            y_pred_hr_np = y_pred_hr_np[:, :min_len]
            y_test_interp_np = y_test_interp_np[:, :min_len]
            x_highres = x_highres[:min_len]
        """
        #  Absolute Error Heatmap
        sample_idx = 0
        error_map = np.abs(y_pred_hr_np[sample_idx] - y_test_interp_np[sample_idx])
        error_map_sample = error_map[np.newaxis, :]
        vmin_all = min(y_pred_hr_np[sample_idx].min(), y_test_interp_np[sample_idx].min())
        vmax_all = max(y_pred_hr_np[sample_idx].max(), y_test_interp_np[sample_idx].max())

        plt.figure(figsize=(10, 2))  # make height small
        plt.imshow(error_map_sample, extent=[x_highres[0], x_highres[-1], 0, 1],
                aspect='auto', origin='lower', cmap='magma')
        plt.colorbar(label='|Prediction Error|')
        plt.xlabel('t (time)')
        plt.yticks([])  # no y-axis labels, just one sample
        plt.title(f'Absolute Error Heatmap: Test Sample #{sample_idx}')
        plt.tight_layout()
        plt.savefig(os.path.join(save_results_to, 'error_heatmap.png'), dpi=300)
        plt.show()

        # Predicted vs Ground Truth field comparison
        sample_idx = 0  # or set a fixed one, e.g., 0

        y_true_sample = y_test_interp_np[sample_idx]
        y_pred_sample = y_pred_hr_np[sample_idx]
        error_sample = np.abs(y_pred_sample - y_true_sample)

        # === Line plots for visual comparison ===
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        axes[0].plot(x_highres, y_true_sample, 'k--', label='Ground Truth', linewidth=1.5)
        axes[0].set_title(f'Test Sample #{sample_idx} — Ground Truth')
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        axes[1].plot(x_highres, y_pred_sample, 'r', label='Prediction', linewidth=1.2)
        axes[1].set_title('Model Prediction (Super-Res)')
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        axes[2].plot(x_highres, error_sample, 'm', label='|Error|', linewidth=1.0)
        axes[2].set_title('Absolute Error Over Time')
        axes[2].set_xlabel('t (time)')
        axes[2].legend()
        axes[2].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_results_to, f'sample_{sample_idx}_comparison_lineplots.png'), dpi=300)
        plt.show()

        y_true = y_test_interp_np[sample_idx]
        y_base = y_base_up[sample_idx].cpu().numpy()
        y_super = y_pred_hr_np[sample_idx]

        # Compute absolute errors
        err_base = np.abs(y_base - y_true)
        err_super = np.abs(y_super - y_true)
        err_vmin, err_vmax = 0, max(err_base.max(), err_super.max())

        # === Heatmaps of the same sample ===
        fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)

        # Expand each into [1, time] so imshow can render it
        y_true_heat = y_true_sample[np.newaxis, :]
        y_pred_heat = y_pred_sample[np.newaxis, :]
        error_heat = error_sample[np.newaxis, :]

        im0 = axes[0].imshow(y_true_heat, extent=[x_highres[0], x_highres[-1], 0, 1],
                            aspect='auto', origin='lower', cmap='viridis', vmin = vmin_all, vmax = vmax_all)
        axes[0].set_title(f'Ground Truth (Sample #{sample_idx})')
        axes[0].set_ylabel('')
        plt.colorbar(im0, ax=axes[0], orientation='vertical', fraction=0.046, pad=0.04)

        im1 = axes[1].imshow(y_pred_heat, extent=[x_highres[0], x_highres[-1], 0, 1],
                            aspect='auto', origin='lower', cmap='viridis', vmin = vmin_all, vmax = vmax_all)
        axes[1].set_title('Prediction (Super-Res)')
        axes[1].set_ylabel('')
        plt.colorbar(im1, ax=axes[1], orientation='vertical', fraction=0.046, pad=0.04)

        im2 = axes[2].imshow(error_heat, extent=[x_highres[0], x_highres[-1], 0, 1],
                            aspect='auto', origin='lower', cmap='magma', vmin = 0, vmax = err_vmax)
        axes[2].set_title('|Error| = |u_pred - u_true|')
        axes[2].set_xlabel('t (time)')
        axes[2].set_ylabel('')
        plt.colorbar(im2, ax=axes[2], orientation='vertical', fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig(os.path.join(save_results_to, f'sample_{sample_idx}_comparison_heatmaps.png'), dpi=300)
        plt.show()

        sample_idx = 0  # or choose any sample index
        x_vals = x_highres  # common grid
        y_true = y_test_interp_np[sample_idx]
        y_base = y_base_up[sample_idx].cpu().numpy()
        y_super = y_pred_hr_np[sample_idx]

        # Compute absolute errors
        err_base = np.abs(y_base - y_true)
        err_super = np.abs(y_super - y_true)
        vmin_all = min(y_true.min(), y_base.min(), y_super.min())
        vmax_all = max(y_true.max(), y_base.max(), y_super.max())
        err_vmin, err_vmax = 0, max(err_base.max(), err_super.max())

        # === Line Plot Comparison ===
        plt.figure(figsize=(10, 6))
        plt.plot(x_vals, y_true, 'k--', label='Ground Truth', linewidth=1.5)
        plt.plot(x_vals, y_base, 'b', label='Base Upscaled', linewidth=1.2)
        plt.plot(x_vals, y_super, 'r', label='Super-Res Prediction', linewidth=1.2)
        plt.title(f"Test Sample #{sample_idx}: Ground Truth vs Base vs Super-Res")
        plt.xlabel("Time (t)")
        plt.ylabel("Displacement u(t)")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_results_to, f'sample_{sample_idx}_base_vs_super_lineplot.png'), dpi=300)
        plt.show()

        # === Absolute Error Line Plot ===
        plt.figure(figsize=(10, 4))
        plt.plot(x_vals, err_base, 'b', label='|Error| Base vs GT')
        plt.plot(x_vals, err_super, 'r', label='|Error| Super-Res vs GT')
        plt.title(f"Absolute Errors Comparison (Sample #{sample_idx})")
        plt.xlabel("Time (t)")
        plt.ylabel("|u_pred - u_true|")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_results_to, f'sample_{sample_idx}_error_lineplots.png'), dpi=300)
        plt.show()

        # === Heatmap Comparison ===
        fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

        # GT
        im0 = axes[0].imshow(y_true[np.newaxis, :], extent=[x_vals[0], x_vals[-1], 0, 1],
                            aspect='auto', origin='lower', cmap='viridis', vmin=vmin_all, vmax=vmax_all)
        axes[0].set_title(f'Ground Truth (Sample #{sample_idx})')
        plt.colorbar(im0, ax=axes[0], orientation='vertical', fraction=0.046, pad=0.04)

        # Base Upscaled
        im1 = axes[1].imshow(y_base[np.newaxis, :], extent=[x_vals[0], x_vals[-1], 0, 1],
                            aspect='auto', origin='lower', cmap='viridis', vmin=vmin_all, vmax=vmax_all)
        axes[1].set_title('Base Upscaled Prediction')
        plt.colorbar(im1, ax=axes[1], orientation='vertical', fraction=0.046, pad=0.04)

        # Super-Res
        im2 = axes[2].imshow(y_super[np.newaxis, :], extent=[x_vals[0], x_vals[-1], 0, 1],
                            aspect='auto', origin='lower', cmap='viridis', vmin=vmin_all, vmax=vmax_all)
        axes[2].set_title('Super-Resolution Prediction')
        plt.colorbar(im2, ax=axes[2], orientation='vertical', fraction=0.046, pad=0.04)

        # Errors heatmap (difference of both)
        err_ratio = np.clip(err_base / (err_super + 1e-8), 0, 5)  # optional ratio comparison
        im3 = axes[3].imshow(np.stack([err_base, err_super]).T[np.newaxis, :, 0], extent=[x_vals[0], x_vals[-1], 0, 1],
                            aspect='auto', origin='lower', cmap='magma', vmin=err_vmin, vmax=err_vmax)
        axes[3].set_title('|Error| Heatmap (Base vs Super)')
        axes[3].set_xlabel("Time (t)")
        plt.colorbar(im3, ax=axes[3], orientation='vertical', fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig(os.path.join(save_results_to, f'sample_{sample_idx}_base_vs_super_heatmaps.png'), dpi=300)
        plt.show()
        
        # Save results
        np.savetxt(save_results_to + "/superres_metrics.txt", 
           np.array([[mse.item(), l2.item(), elapsed_time]]),
           header="MSE   L2   Time(s)")
        save_path = os.path.join(save_results_to, "wave_states_superres.mat")
        sio.savemat(save_path, {
            'x_highres': x_grid_hr.cpu().numpy(),
            'y_pred_highres': y_pred_hr,
            'y_test': y_test.numpy(),
        })

        print("\n=============================")
        print(f"✅ Zero-Shot Super Resolution Complete")
        print(f"Saved high-res predictions to: {save_path}")
        print("=============================\n")
SNO1dmain().supRes()
