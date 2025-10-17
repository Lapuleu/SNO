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

import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.special as sp

import torch
import torch.nn as nn

class Sumudu_Transform(nn.Module):
    def __init__(self, in_channels, out_channels, degree, width, s, device=None):
        """
        Vectorized, GPU-friendly Sumudu transform module.
        - in_channels: input channels (i)
        - out_channels: output channels (o)
        - degree: polynomial degree (number of coefficients)
        - width: number of evaluation points to approximate sum (s_grid size)
        - s: grid size (number of x points used for approximation; same as `width` in semantics)
        - device: optional (module buffers will still move with module.to(device))
        """
        super().__init__()
        self.degree = degree
        self.width = width
        self.s = s
        self.in_channels = in_channels
        self.out_channels = out_channels

        # learnable weights (use float64 for numerical stability)
        self.scale = 1.0 / (in_channels * out_channels)
        self.weight1 = nn.Parameter(self.scale * torch.rand((in_channels, out_channels), dtype=torch.float64))
        # keep the original flip if you want random +-1 behavior
        self.flip = (-1) ** torch.randint(0, 2, (in_channels, out_channels))

        dtype = torch.float64

        # precompute x grid on [0,1] (matches original code)
        x_grid = torch.linspace(0.0, 1.0, steps=s, dtype=dtype)  # [s]

        # Vandermonde matrix: [s, degree], highest power first (increasing=False)
        vander = torch.vander(x_grid, N=degree, increasing=False)  # [s, degree]

        # pseudo-inverse of vander for fast polyfit: shape [degree, s]
        # rcond chosen small to stabilize; you can tune
        vander_pinv = torch.linalg.pinv(vander, rcond=1e-6)

        # Precompute factorials up to max(s, degree) (float64)
        max_fact = max(s, degree)
        # factorial(n) = exp(lgamma(n+1)); create tensor [1..max_fact]
        idx = torch.arange(1, max_fact + 1, dtype=dtype)
        fact = torch.exp(torch.lgamma(idx + 1.0))  # [max_fact]
        # We'll store reversed like original code used .flip(0) sometimes; but keep direct ordering
        # register buffers so they move with the module (and get saved in state_dict)
        self.register_buffer('x_grid', x_grid)
        self.register_buffer('vander', vander)
        self.register_buffer('vander_pinv', vander_pinv)
        self.register_buffer('factorial', fact)  # factorial[n-1] == n!

    def coefficient_training(self, input):
        """
        Fit polynomial coefficients for each (batch, channel) using precomputed vander_pinv.
        input: [B, C, D] where D is number of samples (must match self.s used to build vander)
        returns: [B, C, degree]
        """
        B, C, D = input.shape
        # ensure double for numerical stability
        y = input.reshape(-1, D).double()  # [B*C, D]
        # coef = (vander_pinv @ y.T).T  => [B*C, degree]
        coef = (self.vander_pinv[:, :D] @ y.T).T   # use only first D cols of vander_pinv if D < s
        coef = coef.reshape(B, C, -1)  # [B, C, degree]
        return coef

    def transform(self, input):
        """
        Multiply coefficients by factorials (elementwise). Input shape [B, C, degree]
        """
        D = input.shape[2]
        # factorial indices: 0..D-1 correspond to 1..D so use factorial[:D]
        fact = self.factorial[:D].view(1, 1, -1)
        return input.double() * fact

    def inverse_transform(self, input):
        D = input.shape[2]
        fact = self.factorial[:D].view(1, 1, -1)
        return input.double() / fact

    def approximate_sum(self, width, input):
        """
        Evaluate polynomial (coefficients in `input`) on the stored x_grid.
        input: [B, C, degree]
        returns: [B, C, s]  (s == self.s)
        """
        B, C, degree = input.shape
        # use vander[:, :degree] which is [s, degree]
        V = self.vander[:, :degree]   # [s, degree]
        # compute output: (B*C, degree) @ (degree, s) -> (B*C, s) then reshape
        out = input.reshape(-1, degree).double() @ V.T   # [B*C, s]
        out = out.reshape(B, C, self.s)
        return out

    def weight_mul(self, input):
        """
        Multiply input [B, in_channels, s] by weight1 [in_channels, out_channels]
        producing [B, out_channels, s]
        """
        # einsum keeps things vectorized and fast on GPU
        return torch.einsum("b i x, i o -> b o x", input.double(), self.weight1)

    def forward(self, x):
        B, C, D = x.shape
        # Fit coefficients from raw samples -> [B, C, degree]
        x = self.coefficient_training(x)     # [B, C, degree]
        x = self.transform(x)                        # factorial scale
        # evaluate polynomial on x_grid -> [B, C, s]
        x = self.approximate_sum(self.width, x)  # [B, C, s]

        # apply weight multiply -> [B, out_channels, s]
        x = self.weight_mul(x)

        # Fit coefficients again on transformed channels (per original pipeline)
        x = self.coefficient_training(x)  # [B, out_channels, degree]
        x = self.inverse_transform(x)
        x = self.approximate_sum(self.width, x)  # [B, out_channels, s]

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
        x1 = self.conv0(x)
        x2 = self.w0(x)
        # Match resolutions if different (super-resolution case)
        if x1.shape[-1] != x2.shape[-1]:
            x2 = F.interpolate(x2, size=x1.shape[-1], mode='linear', align_corners=False)
        x = self.bn0(x1 + x2)

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        if self.activation == "relu":
            x = F.relu(x)
        elif self.activation == "leaky_relu":
            x = F.leaky_relu(x)
        elif self.activation == "relu6":
            x = F.relu6(x)
        elif self.activation == "gelu":
            x = F.gelu(x)
        elif self.activation == "tanh":
            x = torch.tanh(x)
        elif self.activation == "sin":
            x = torch.sin(x)
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
        case = "Case_SDuf0SR_"
        folder_index = str(save_index)

        results_dir = "/" + case + folder_index +"/"
        save_results_to = current_directory + results_dir
        if not os.path.exists(save_results_to):
            os.makedirs(save_results_to)

        # Load model
        model = torch.load("/workspace/Wave_states", map_location=device, weights_only=False)
        model.eval()

        # Define higher-resolution grid
        new_s = 8192  # <-- super-res factor (e.g., 4× original 2048)
        dtype = torch.float64
        x_grid_hr = torch.linspace(0.0, 1.0, steps=new_s, dtype=dtype)
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

        # Load test data (forcing term)
        reader = MatReader('/workspace/Data/Duff0/data.mat')
        x_test = reader.read_field('f_test')
        y_test = reader.read_field('u_test')
        grid_x_test = reader.read_field('x_test')

        x_test = x_test.reshape(x_test.shape[0], 2048, 1)
        x_test = x_test.detach().clone().to(torch.float32).to(device)

        # Run zero-shot super resolution
        print(f"[Running zero-shot super resolution inference] ...")
        with torch.no_grad():
            y_pred_hr = model(x_test).cpu().numpy()
        
        # Measure inference time
        start_time = time.time()
        with torch.no_grad():
            y_pred_hr = model(x_test).cpu()
        elapsed_time = time.time() - start_time

        # Compute loss metrics
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

        x_highres = x_grid_hr.cpu().numpy()
        y_pred_hr_np = y_pred_hr.cpu().numpy() if torch.is_tensor(y_pred_hr) else y_pred_hr
        y_test_interp_np = y_test_interp.cpu().numpy()

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

        """# Predicted vs Ground Truth field comparison
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

        # Ground truth
        im0 = axes[0].imshow(y_test_interp_np, extent=[x_highres[0], x_highres[-1], 0, y_test_interp_np.shape[0]],
                            aspect='auto', origin='lower', cmap='viridis')
        axes[0].set_title('Ground Truth (u_test)')
        axes[0].set_xlabel('x (space or time)')
        axes[0].set_ylabel('Test sample index')
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        # Prediction
        im1 = axes[1].imshow(y_pred_hr_np, extent=[x_highres[0], x_highres[-1], 0, y_pred_hr_np.shape[0]],
                            aspect='auto', origin='lower', cmap='viridis')
        axes[1].set_title('Predicted (Super-Res)')
        axes[1].set_xlabel('x (space or time)')
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        # Absolute error
        im2 = axes[2].imshow(error_map, extent=[x_highres[0], x_highres[-1], 0, error_map.shape[0]],
                            aspect='auto', origin='lower', cmap='magma')
        axes[2].set_title('|Error| = |u_pred - u_true|')
        axes[2].set_xlabel('x (space or time)')
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        plt.suptitle('Zero-Shot Super-Resolution: Field Comparison and Error Map', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(save_results_to, 'superres_field_comparison.png'), dpi=300)
        plt.show()
        """
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