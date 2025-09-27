
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
from D2utilities3 import *
from D2Adam import Adam
import time
import math
import scipy.special as sp
import warnings
warnings.simplefilter('ignore', np.exceptions.RankWarning)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# ====================================
# saving settings
# ====================================
save_index = 1   
current_directory = os.getcwd()
case = "Case_SBeam_"
folder_index = str(save_index)

results_dir = "/" + case + folder_index +"/"
save_results_to = current_directory + results_dir
if not os.path.exists(save_results_to):
    os.makedirs(save_results_to)
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
        self.weights3 = nn.Parameter(self.scale * torch.rand((in_channels, out_channels), dtype=torch.float64))
        self.weights4 = nn.Parameter(self.scale * torch.rand((in_channels, out_channels), dtype=torch.float64))


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
        self.fc0 = nn.Linear(3, width)
        self.conv0 = Sumudu_Transform(width, width, degree, width, s)
        self.w0 = nn.Conv2d(width, width, 1)
        self.norm = nn.InstanceNorm2d(width)
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x).permute(0, 3, 1, 2)
        x1 = self.norm(self.conv0(self.norm(x)))
        x2 = self.w0(x)
        x = x1 + x2
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        B, Nx, Ny = shape[0], shape[1], shape[2]
        gx = torch.linspace(0, 1, Nx, device=device).view(1, Nx, 1, 1).expand(B, Nx, Ny, 1)
        gy = torch.linspace(0, 1, Ny, device=device).view(1, 1, Ny, 1).expand(B, Nx, Ny, 1)
        return torch.cat((gx, gy), dim=-1)

# ====================================
#  Define parameters and Load data
# ====================================

class main():
    def __init__(self, degree):
        self.degree = degree

    def train(self):
        s = 50
        ntrain = 200
        nvali = 50
        ntest=130

        batch_size_train = 50
        batch_size_vali = 50

        learning_rate = 0.002

        epochs = 1000
        step_size = 100
        gamma = 0.5

        modes1 = 4  
        modes2 = 4   
        width = 16

        # ====================================
        #  Define parameters and Load data
        # ====================================
        file = np.load('Data/Brusselator_force_train.npz')

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

        t = nt
        orig_r = 28
        r = 2
        h = int(((orig_r - 1)/r) + 1)
        s = h

        modes1 = 4
        modes2 = 4
        modes3 = 4
        width = 8
            
        x = np.linspace(0, 1, orig_r)
        y = np.linspace(0, 1, orig_r)
        z = np.linspace(0, 1, t)
        tt, xx, yy = np.meshgrid(z, x, y, indexing='ij')

        T=torch.linspace(0,19,nt).reshape(1,nt)
        X=torch.linspace(0,1,steps=orig_r).reshape(1,orig_r)[:,:s]
        Y=torch.linspace(0,1,steps=orig_r).reshape(1,orig_r)[:,:s]


        x_train = torch.tile(torch.tensor(inputs_train),(orig_r,orig_r,1,1)).permute(2,3,0,1)[:,:,::r,::r][:,:,:s,:s]
        y_train = torch.tensor(outputs_train)[:,:,::r,::r][:,:,:s,:s]
        grid_x_train = torch.tile(torch.tensor(tt),(num_train,1,1,1))[:,:,::r,::r][:,:,:s,:s]
        grid_y_train = torch.tile(torch.tensor(xx),(num_train,1,1,1))[:,:,::r,::r][:,:,:s,:s]
        grid_z_train = torch.tile(torch.tensor(yy),(num_train,1,1,1))[:,:,::r,::r][:,:,:s,:s]

        x_test = torch.tile(torch.tensor(inputs_test),(orig_r,orig_r,1,1)).permute(2,3,0,1)[:,:,::r,::r][:,:,:s,:s]
        y_test = torch.tensor(outputs_test)[:,:,::r,::r][:,:,:s,:s]
        grid_x_test = torch.tile(torch.tensor(tt),(num_test,1,1,1))[:,:,::r,::r][:,:,:s,:s]
        grid_y_test = torch.tile(torch.tensor(xx),(num_test,1,1,1))[:,:,::r,::r][:,:,:s,:s]
        grid_z_test = torch.tile(torch.tensor(yy),(num_test,1,1,1))[:,:,::r,::r][:,:,:s,:s]


        x_normalizer = RangeNormalizer(x_train)
        x_train = x_normalizer.encode(x_train)
        x_test = x_normalizer.encode(x_test)

        y_normalizer = RangeNormalizer(y_train)
        y_train = y_normalizer.encode(y_train)

        grid_x_train = grid_x_train.reshape(num_train, t, s, s, 1)  
        grid_x_train.requires_grad = True
        grid_y_train = grid_y_train.reshape(num_train, t, s, s, 1)
        grid_y_train.requires_grad = True
        grid_z_train = grid_z_train.reshape(num_train, t, s, s, 1)
        grid_z_train.requires_grad = True
        x_train = x_train.reshape(num_train, t, s, s, 1)
        x_train.requires_grad = True
        x_train = torch.cat([x_train, grid_x_train, grid_y_train, grid_z_train], dim = -1)

        grid_x_test = grid_x_test.reshape(num_test, t, s, s, 1)
        grid_y_test = grid_y_test.reshape(num_test, t, s, s, 1)
        grid_z_test = grid_z_test.reshape(num_test, t, s, s, 1)
        x_test = x_test.reshape(num_test, t, s, s, 1)
        x_test = torch.cat([x_test, grid_x_test, grid_y_test, grid_z_test], dim = -1)


        y_train = y_train.reshape(num_train, t, s, s, 1)
        y_test = y_test.reshape(num_test, t, s, s, 1)

        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=False)
        train_loaderL2 = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=False)
        vali_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

        device = torch.device('cuda') 
        # model
        model = SNO2d(self.degree, width,s).to(device)
        best_val = float("inf")
        patience = 2000   # stop after 20 epochs without improvement
        counter = 0

        # ====================================
        # Training 
        # ====================================
        optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        start_time = time.time()
        myloss = LpLoss(size_average=True)

        best_val = float("inf")
        counter = 0
        patience = 2000   # stop after 200 epochs without improvement

        train_error = np.zeros((epochs, 1))
        train_loss = np.zeros((epochs, 1))
        vali_error = np.zeros((epochs, 1))
        vali_loss = np.zeros((epochs, 1))
        for ep in range(epochs):
            model.train()
            t1 = default_timer()
            train_mse = 0
            train_l2 = 0
            n_train=0
            for x, y in train_loader:
                x, y = x.cuda(), y.cuda()

                optimizer.zero_grad()
                out = model(x)   
                mse = F.mse_loss(out.view(batch_size_train, -1), y.view(batch_size_train, -1), reduction='mean')
                l2 = myloss(out.view(-1,x_train.shape[1],x_train.shape[2]), y)
                l2.backward()

                optimizer.step()
                train_mse += mse.item()
                train_l2 += l2.item()
                n_train += 1

            scheduler.step()
            model.eval()
            vali_mse = 0.0
            vali_l2 = 0.0
            with torch.no_grad():
                n_vali=0
                for x, y in vali_loader:
                    x, y = x.cuda(), y.cuda()
                    out = model(x)
                    mse=F.mse_loss(out.view(-1,x_vali.shape[1],x_vali.shape[2]), y, reduction='mean')
                    vali_l2 += myloss(out.view(-1,x_vali.shape[1],x_vali.shape[2]), y).item()
                    vali_mse += mse.item()
                    n_vali += 1

            train_mse /= n_train
            vali_mse /= n_vali
            train_l2 /= n_train
            vali_l2 /= n_vali
            train_error[ep,0] = train_l2
            vali_error[ep,0] = vali_l2
            train_loss[ep,0] = train_mse
            vali_loss[ep,0] = vali_mse
            if vali_l2 < best_val:
                best_val = vali_l2
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping at epoch {ep}")
                    break
            t2 = default_timer()
            print("Epoch: %d, time: %.3f, Train Loss: %.3e,Vali Loss: %.3e, Train l2: %.4f, Vali l2: %.4f" % (ep, t2-t1, train_mse, vali_mse,train_l2, vali_l2))
        elapsed = time.time() - start_time
        print("\n=============================")
        print("Training done...")
        print('Training time: %.3f'%(elapsed))
        print("=============================\n")

        x = np.linspace(0, epochs-1, epochs)
        np.savetxt(save_results_to+'/epoch.txt', x)
        np.savetxt(save_results_to+'/train_loss.txt', train_loss)
        np.savetxt(save_results_to+'/vali_loss.txt', vali_loss)
        np.savetxt(save_results_to+'/train_error.txt', train_error)
        np.savetxt(save_results_to+'/vali_error.txt', vali_error)    
        save_models_to = save_results_to +"model/"
        if not os.path.exists(save_models_to):
            os.makedirs(save_models_to)
            
        torch.save(model, save_models_to+'Wave_states')

        # ====================================
        # testing
        # ====================================
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=1, shuffle=False)
        pred_u = torch.zeros(ntest,y_test.shape[1],y_test.shape[2])
        index = 0
        test_l2 = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.cuda(), y.cuda()
                out = model(x)
                test_l2 += myloss(out.view(-1,x_test.shape[1],x_test.shape[2]), y).item()
                pred_u[index,:,:] = out.view(-1,x_test.shape[1],x_test.shape[2])
                index = index + 1
        test_l2 /= index
        scipy.io.savemat(save_results_to+'wave_states_test.mat', 
                            mdict={ 'test_err': test_l2,
                                    'T': T.numpy(),
                                    'X': X.numpy(),
                                    'y_test': y_test.numpy(), 
                                    'y_pred': pred_u.cpu().numpy()})  
            
            
        print("\n=============================")
        print('Testing error: %.3e'%(test_l2))
        print("=============================\n")


        # Plotting the loss history
        num_epoch = epochs
        epoch = np.linspace(1, num_epoch, num_epoch)
        fig = plt.figure(constrained_layout=False, figsize=(7, 7))
        gs = fig.add_gridspec(1, 1)
        ax = fig.add_subplot(gs[0])
        ax.plot(epoch, train_loss[:,0], color='blue', label='Train Loss')
        ax.plot(epoch, vali_loss[:,0], color='red', label='Validation Loss')
        ax.set_yscale('log')
        ax.set_ylabel('Loss')
        ax.set_xlabel('Epochs')
        ax.legend(loc='upper left')
        fig.savefig(save_results_to+'loss_history.png')
        plt.show()
        return float(best_val)
main(degree=4).train()
