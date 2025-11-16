
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
from utilities3 import *
from Adam import Adam
import time
import math
import scipy.special as sp
import warnings
warnings.simplefilter('ignore', np.exceptions.RankWarning)

torch.manual_seed(0)
np.random.seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
torch.backends.cudnn.benchmark = True

# ====================================
# Data Gen
# ====================================

class Sumudu_Transform(nn.Module):
    def __init__(self, in_channels, out_channels, degree, width, s):
        super().__init__()
        self.degree = degree
        self.width  = width
        self.s      = s
        self.in_channels  = in_channels
        self.out_channels = out_channels

        # random learnable parameters
        self.flip   = (-1)**torch.randint(0, 2, (in_channels, out_channels))
        self.scale  = 1.0 / (in_channels * out_channels)
        self.weight1 = nn.Parameter(self.scale * torch.rand((in_channels, out_channels), dtype=torch.float64))
        self.weight2 = nn.Parameter(self.scale * torch.rand((in_channels, out_channels), dtype=torch.float64))
        self.weight3 = nn.Parameter(self.scale * torch.rand((in_channels, out_channels), dtype=torch.float64))
        self.weight4 = nn.Parameter(self.scale * torch.rand((in_channels, out_channels), dtype=torch.float64))


        # constant grid and factorial (factorial can still be cached safely)
        dtype = torch.float64
        self.register_buffer('x_grid',torch.linspace(0, (s-1)*.02, s, dtype=dtype))
        fact = torch.exp(torch.lgamma(torch.linspace(0, s-1, s, dtype=dtype)+1))
        self.register_buffer('factorial', fact)

    def coefficient_training(self, input, degree):
        """
        Fit polynomial coefficients each call using a fresh Vandermonde and pinv.
        """
        B, N, A, M, S = input.shape
        y = input.reshape(-1, S).double()           # [B*N*A*M, s]
        V = torch.vander(self.x_grid, N=degree, increasing=False)  # [s, degree]
        pinv = torch.linalg.pinv(V, rcond=1e-4)                      # [degree, s]
        coef = (pinv @ y.T).T                                        # [B*N*M, degree]
        return coef.reshape(B, N, A, M, degree)

    def transform(self, input):
        return input * self.factorial[:input.shape[4]]

    def inverse_transform(self, input):
        return input / self.factorial[:input.shape[4]]

    def approximate_sum(self, width, input):
        """
        Evaluate polynomial each call using a fresh Vandermonde.
        """
        B, N, A, M, D = input.shape
        V = torch.vander(self.x_grid, N=D, increasing=False)   # [s, degree]
        return torch.einsum('b n a m d, s d -> b n a m s', input, V)

    def weight_mul(self, input):
        t1 = torch.einsum("b i t x s, i o -> b o t x s", input, self.weight1)
        t2 = torch.einsum("b i t y s, i o -> b o t y s", input, self.weight2)
        t3 = torch.einsum("b i x y s, i o -> b oxys ", input, self.weight3)
        t4 = torch.einsum("bixts, io-> boxts", input, self.weight4)
        return (t1+t2+t3+t4)**2

    def forward(self, x):
        x = x.double()
        x = self.coefficient_training(x, self.degree)
        x = self.transform(x)
        x = self.approximate_sum(self.width, x)
        x = self.weight_mul(x)
        x = self.coefficient_training(x, self.degree)
        x = self.inverse_transform(x)
        x = self.approximate_sum(self.width, x)
        return x.float()


class SNO2d(nn.Module):
    def __init__(self, degree, width, s):
        super().__init__()
        self.width = width
        self.degree = degree
        self.fc0 = nn.Linear(4, width)
        self.conv0 = Sumudu_Transform(width, width, degree, width, s)
        self.w0 = nn.Conv3d(width, width, 1)
        self.norm = nn.InstanceNorm3d(width)
        self.fc1 = nn.Linear(width, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        #grid = self.get_grid(x.shape, x.device)
        #x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)
        x1 = self.norm(self.conv0(self.norm(x)))
        x2 = self.w0(x)
        x = x1 + x2
        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        B, Nx, Ny = shape[0], shape[1], shape[2]
        gx = torch.linspace(0, 1, Nx, device=device).view(1, Nx, 1, 1).expand(B, Nx, Ny, 1)
        gy = torch.linspace(0, 1, Ny, device=device).view(1, 1, Ny, 1).expand(B, Nx, Ny, 1)
        return torch.cat((gx, gy), dim=-1)

class main():
    def __init__(self, degree):
        self.degree = degree

    def train(self):
        file = np.load('/workspace/Brus/Brusselator_force_train.npz')

        nt, nx, ny = 39, file['nx'], file['ny']
        num_train = file['n_samples1']
        num_test = file['n_samples2']
        inputs_train = file['inputs_train'].reshape(num_train, nt)
        inputs_test = file['inputs_test'].reshape(num_test, nt)
        outputs_train = np.array((file['outputs_train'])).reshape(num_train, nt, nx, ny)
        outputs_test = np.array((file['outputs_test'])).reshape(num_test, nt, nx, ny)
                
        batch_size = 50
        epochs = 300
        learning_rate = 0.005
        step_size = 100
        gamma = 0.5
        s = 50

        width = 16

        t = nt
        orig_r = 28
        r = 2
        h = int(((orig_r - 1) / r) + 1)
        s = h

        x = np.linspace(0, 1, orig_r)
        y = np.linspace(0, 1, orig_r)
        z = np.linspace(0, 1, t)
        tt_np, xx_np, yy_np = np.meshgrid(z, x, y, indexing='ij')

        # Put time/location coordinate tensors on device immediately
        T = torch.linspace(0, 19, nt, device=device).reshape(1, nt)    # used in PR3d
        X = torch.linspace(0, 1, steps=orig_r, device=device).reshape(1, orig_r)[:, :s]
        Y = torch.linspace(0, 1, steps=orig_r, device=device).reshape(1, orig_r)[:, :s]

        # Prepare training/test tensors and move to device only when needed (DataLoader handles batched transfer)
        # Convert numpy into torch tensors (stay on CPU for dataset creation) - they will be moved to GPU per-batch
        x_train = torch.tile(torch.tensor(inputs_train, dtype=torch.float32), (orig_r, orig_r, 1, 1)).permute(2, 3, 0, 1)[:, :, ::r, ::r][:, :, :s, :s]
        y_train = torch.tensor(outputs_train, dtype=torch.float32)[:, :, ::r, ::r][:, :, :s, :s]
        grid_x_train = torch.tile(torch.tensor(tt_np, dtype=torch.float32), (num_train, 1, 1, 1))[:, :, ::r, ::r][:, :, :s, :s]
        grid_y_train = torch.tile(torch.tensor(xx_np, dtype=torch.float32), (num_train, 1, 1, 1))[:, :, ::r, ::r][:, :, :s, :s]
        grid_z_train = torch.tile(torch.tensor(yy_np, dtype=torch.float32), (num_train, 1, 1, 1))[:, :, ::r, ::r][:, :, :s, :s]

        x_test = torch.tile(torch.tensor(inputs_test, dtype=torch.float32), (orig_r, orig_r, 1, 1)).permute(2, 3, 0, 1)[:, :, ::r, ::r][:, :, :s, :s]
        y_test = torch.tensor(outputs_test, dtype=torch.float32)[:, :, ::r, ::r][:, :, :s, :s]
        grid_x_test = torch.tile(torch.tensor(tt_np, dtype=torch.float32), (num_test, 1, 1, 1))[:, :, ::r, ::r][:, :, :s, :s]
        grid_y_test = torch.tile(torch.tensor(xx_np, dtype=torch.float32), (num_test, 1, 1, 1))[:, :, ::r, ::r][:, :, :s, :s]
        grid_z_test = torch.tile(torch.tensor(yy_np, dtype=torch.float32), (num_test, 1, 1, 1))[:, :, ::r, ::r][:, :, :s, :s]

        # Normalizers (they may contain buffers/params; move them to device when used - here we will move right away)
        x_normalizer = RangeNormalizer(x_train)
        x_train = x_normalizer.encode(x_train)
        x_test = x_normalizer.encode(x_test)

        y_normalizer = RangeNormalizer(y_train)
        y_train = y_normalizer.encode(y_train)

        # Reshape / require_grad as original; but do NOT set requires_grad on dataset tensors (unnecessary)
        grid_x_train = grid_x_train.reshape(num_train, t, s, s, 1)
        grid_y_train = grid_y_train.reshape(num_train, t, s, s, 1)
        grid_z_train = grid_z_train.reshape(num_train, t, s, s, 1)
        x_train = x_train.reshape(num_train, t, s, s, 1)
        x_train = torch.cat([x_train, grid_x_train, grid_y_train, grid_z_train], dim=-1)

        grid_x_test = grid_x_test.reshape(num_test, t, s, s, 1)
        grid_y_test = grid_y_test.reshape(num_test, t, s, s, 1)
        grid_z_test = grid_z_test.reshape(num_test, t, s, s, 1)
        x_test = x_test.reshape(num_test, t, s, s, 1)
        x_test = torch.cat([x_test, grid_x_test, grid_y_test, grid_z_test], dim=-1)

        y_train = y_train.reshape(num_train, t, s, s, 1)
        y_test = y_test.reshape(num_test, t, s, s, 1)

        # DataLoaders: pin_memory=True and set num_workers>0 to better feed the GPU
        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train),
                                                batch_size=batch_size, shuffle=True,
                                                pin_memory=True, num_workers=4)
        train_loaderL2 = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train),
                                                    batch_size=batch_size, shuffle=False,
                                                    pin_memory=True, num_workers=4)
        vali_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test),
                                                batch_size=batch_size, shuffle=False,
                                                pin_memory=True, num_workers=4)
        # model
        model = SNO2d(self.degree, width,s).to(device)

        # ====================================
        # Training 
        # ====================================
        optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

        start_time = time.time()
        myloss = LpLoss(size_average=False)

        # move normalizers to device if they have internal tensors (you used .cuda() previously)
        # I'm assuming your RangeNormalizer supports .cuda(); if not, ensure it moves internal tensors
        try:
            x_normalizer = x_normalizer.to(device)
            y_normalizer = y_normalizer.to(device)
        except Exception:
            # fallback: keep using as-is; we will move tensors passed through normalize/denormalize explicitly
            pass

        train_error = np.zeros((epochs, 1))
        train_loss = np.zeros((epochs, 1))
        vali_error = np.zeros((epochs, 1))
        vali_loss = np.zeros((epochs, 1))

        # training loop
        for ep in range(epochs):
            model.train()
            t1 = default_timer()
            train_mse = 0.0
            for x_batch, y_batch in train_loader:
                # Move batch to GPU non-blocking (DataLoader uses pin_memory)
                x_batch = x_batch.to(device=device, non_blocking=True)
                y_batch = y_batch.to(device=device, non_blocking=True)

                optimizer.zero_grad()
                out = model(x_batch.float())

                out = out[:, :, :, :, 0:1]
                yb = y_batch[:, :, :, :, 0:1]

                loss = myloss(out.view(out.size(0), -1), yb.view(yb.size(0), -1))
                loss.backward()

                optimizer.step()
                train_mse += loss.item()

            scheduler.step()

            # training L2 compute
            model.eval()
            train_l2 = 0.0
            with torch.no_grad():
                for x_batch, y_batch in train_loaderL2:
                    x_batch = x_batch.to(device=device, non_blocking=True)
                    y_batch = y_batch.to(device=device, non_blocking=True)
                    out = model(x_batch.float())

                    # decode on GPU (ensure y_normalizer has device support)
                    try:
                        out = y_normalizer.decode(out[:, :, :, :, 0:1])
                        ydec = y_normalizer.decode(y_batch[:, :, :, :, 0:1])
                    except Exception:
                        # If normalizer isn't on device, move tensors to CPU when decoding (minimize CPU transfers)
                        out = out[:, :, :, :, 0:1]
                        ydec = y_batch[:, :, :, :, 0:1]
                    train_l2 = myloss(out.view(out.size(0), -1), ydec.view(ydec.size(0), -1)).item()

            test_l2 = 0.0
            with torch.no_grad():
                for x_batch, y_batch in vali_loader:
                    x_batch = x_batch.to(device=device, non_blocking=True)
                    y_batch = y_batch.to(device=device, non_blocking=True)
                    out = model(x_batch.float())

                    try:
                        out = y_normalizer.decode(out[:, :, :, :, 0:1])
                    except Exception:
                        out = out[:, :, :, :, 0:1]
                    yb = y_batch[:, :, :, :, 0:1]
                    test_l2 = myloss(out.view(out.size(0), -1), yb.view(yb.size(0), -1)).item()

            train_mse /= len(train_loader)
            train_l2 /= num_train
            test_l2 /= num_test

            train_error[ep, 0] = train_l2
            vali_error[ep, 0] = test_l2
            t2 = default_timer()
            print("Epoch: %d, time: %.3f, Train l2: %.6f, Vali l2: %.6f" % (ep, t2 - t1, train_l2, test_l2))

        elapsed = time.time() - start_time
        print("\n=============================")
        print("Training done...")
        print('Training time: %.3f' % (elapsed))
        print("=============================\n")

        # ====================================
        # saving settings (save state_dict not whole model)
        # ====================================
        current_directory = os.getcwd()
        case = "Case_SBrus_"
        save_index = 1
        folder_index = str(save_index)

        results_dir = "/" + case + folder_index + "/"
        save_results_to = current_directory + results_dir
        if not os.path.exists(save_results_to):
            os.makedirs(save_results_to)

        x = np.linspace(0, epochs - 1, epochs)
        np.savetxt(save_results_to + '/epoch.txt', x)
        np.savetxt(save_results_to + '/train_error.txt', train_error)
        np.savetxt(save_results_to + '/vali_error.txt', vali_error)
        save_models_to = save_results_to + "model/"
        if not os.path.exists(save_models_to):
            os.makedirs(save_models_to)

        torch.save(model.state_dict(), save_models_to + 'Wave_states_state_dict.pt')

        # ====================================
        # Testing
        # ====================================
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test),
                                                batch_size=1, shuffle=False, pin_memory=True, num_workers=2)

        pred_u = torch.zeros(num_test, y_test.shape[1], y_test.shape[2], y_test.shape[3], dtype=torch.float32, device=device)
        index = 0
        test_l2 = 0.0
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(device=device, non_blocking=True)
                y_batch = y_batch.to(device=device, non_blocking=True)
                out = model(x_batch.float())
                try:
                    out = y_normalizer.decode(out[:, :, :, :, 0])
                except Exception:
                    out = out[:, :, :, :, 0]
                test_l2 += myloss(out, y_batch[:, :, :, :, 0]).item()
                pred_u[index, :, :, :] = out
                index += 1

        test_l2 /= index

        # move small items to CPU only at the end for saving
        scipy.io.savemat(save_results_to + 'wave_states_test.mat',
                        mdict={'test_err': test_l2,
                                'T': T.cpu().numpy(),
                                'X': X.cpu().numpy(),
                                'Y': Y.cpu().numpy(),
                                'y_test': y_test,  # still on CPU original tensor
                                'y_pred': pred_u.cpu().numpy(),
                                'Train_time': elapsed})

        print("\n=============================")
        print('Testing error: %.3e' % (test_l2))
        print("=============================\n")
main(degree=4).train()

