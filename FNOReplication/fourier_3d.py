import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from D3utilities3 import *
from D3Adam import Adam
import time
import operator
from functools import reduce
from functools import partial

from timeit import default_timer
import scipy.io

torch.manual_seed(0)
np.random.seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
torch.backends.cudnn.benchmark = True

activation = F.relu

################################################################
# 3d fourier layers
################################################################

import torch
import torch.nn as nn
import torch.fft


def compl_mul3d(a, b):
    # (batch, in_channel, x,y,t), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
    # a: complex tensor, b: complex tensor
    return torch.einsum("bixyz,ioxyz->boxyz", a, b)


class SpectralConv3d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d_fast, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = 1 / (in_channels * out_channels)

        # weights are now complex tensors
        self.weights1 = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat))

    def forward(self, x):
        batchsize = x.shape[0]

        # Fourier transform
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1], norm="ortho")

        # allocate output Fourier tensor
        out_ft = torch.zeros(batchsize,self.out_channels,x.size(-3),x.size(-2),x.size(-1) // 2 + 1,dtype=torch.cfloat,device=x.device,)

        # Apply spectral weights
        out_ft[:, :, : self.modes1, : self.modes2, : self.modes3] = compl_mul3d(x_ft[:, :, : self.modes1, : self.modes2, : self.modes3], self.weights1)
        out_ft[:, :, -self.modes1 :, : self.modes2, : self.modes3] = compl_mul3d(x_ft[:, :, -self.modes1 :, : self.modes2, : self.modes3], self.weights2)
        out_ft[:, :, : self.modes1, -self.modes2 :, : self.modes3] = compl_mul3d(x_ft[:, :, : self.modes1, -self.modes2 :, : self.modes3], self.weights3)
        out_ft[:, :, -self.modes1 :, -self.modes2 :, : self.modes3] = compl_mul3d(x_ft[:, :, -self.modes1 :, -self.modes2 :, : self.modes3], self.weights4)

        # Back to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)), dim=[-3, -2, -1], norm="ortho")
        return x

class SimpleBlock2d(nn.Module):
    def __init__(self, modes1, modes2, modes3, width):
        super(SimpleBlock2d, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.fc0 = nn.Linear(4, self.width)

        self.conv0 = SpectralConv3d_fast(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.bn0 = torch.nn.BatchNorm3d(self.width)

        self.fc1 = nn.Linear(self.width, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y, size_z = x.shape[1], x.shape[2], x.shape[3]

        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)

        x1 = self.conv0(x)
        x2 = self.w0(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x = self.bn0(x1 + x2)

        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

class Net2d(nn.Module):
    def __init__(self, modes, width):
        super(Net2d, self).__init__()

        self.conv1 = SimpleBlock2d(modes, modes, modes, width)


    def forward(self, x):
        x = self.conv1(x)
        return x


    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))

        return c

# ====================================
#  Define parameters and Load data
# ====================================
file = np.load('/workspace/Data/Brus/Brusselator_force_train.npz')

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

modes1 = 4
modes2 = 4
modes3 = 4
width = 8
    
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
model = Net2d(modes1,width).to(device)


# ====================================
# Training 
# ====================================
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
start_time = time.time()
myloss = LpLoss(size_average=False)
x_normalizer.cuda()
y_normalizer.cuda()

train_error = np.zeros((epochs, 1))
train_loss = np.zeros((epochs, 1))
vali_error = np.zeros((epochs, 1))
vali_loss = np.zeros((epochs, 1))
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_mse = 0
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()     
        optimizer.zero_grad()
        out = model(x.float())

        out = out[:,:,:,:,0:1]            
        y = y[:,:,:,:,0:1]

        loss = myloss(out.view(batch_size, -1), y.view(batch_size, -1)) 
        loss.backward()

        optimizer.step()
        train_mse += loss.item()  

    scheduler.step()
    model.eval()
    train_l2 = 0.0
    with torch.no_grad():
            for x, y in train_loaderL2:
                x, y = x.cuda(), y.cuda()
                out = model(x.float())
                
                out = y_normalizer.decode(out[:,:,:,:,0:1])               
                y = y_normalizer.decode(y[:,:,:,:,0:1])
                
                train_l2 = myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item() 

    test_l2 = 0.0
    with torch.no_grad():
        n_vali=0
        for x, y in vali_loader:
            x, y = x.cuda(), y.cuda()
            out = model(x.float())              
            out = y_normalizer.decode(out[:,:,:,:,0:1])
            y = y[:,:,:,:,0:1]
            test_l2 = myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item() 
            
    train_mse /= len(train_loader)
    train_l2 /= num_train
    test_l2 /= num_test


    train_error[ep,0] = train_l2
    vali_error[ep,0] = test_l2
    t2 = default_timer()
    print("Epoch: %d, time: %.3f, Train l2: %.4f, Vali l2: %.4f" % (ep, t2-t1, train_l2, test_l2))
elapsed = time.time() - start_time
print("\n=============================")
print("Training done...")
print('Training time: %.3f'%(elapsed))
print("=============================\n")


# ====================================
# saving settings
# ====================================
current_directory = os.getcwd()
case = "Case_FBrus_"
save_index = 1  
folder_index = str(save_index)

results_dir = "/" + case + folder_index +"/"
save_results_to = current_directory + results_dir
if not os.path.exists(save_results_to):
    os.makedirs(save_results_to)

x = np.linspace(0, epochs-1, epochs)
np.savetxt(save_results_to+'/epoch.txt', x)
np.savetxt(save_results_to+'/train_error.txt', train_error)
np.savetxt(save_results_to+'/vali_error.txt', vali_error)    
save_models_to = save_results_to +"model/"
if not os.path.exists(save_models_to):
    os.makedirs(save_models_to)
    
torch.save(model, save_models_to+'Wave_states')

# ====================================
# Testing 
# ====================================
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=1, shuffle=False)
pred_u = torch.zeros(num_test,y_test.shape[1],y_test.shape[2],y_test.shape[3])
index = 0
test_l2 = 0.0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.cuda(), y.cuda()
        out = model(x.float())
        out = y_normalizer.decode(out[:,:,:,:,0])    
        test_l2 += myloss(out, y[:,:,:,:,0]).item()
        pred_u[index,:,:,:] = out
        index = index + 1
test_l2 /= index
scipy.io.savemat(save_results_to+'wave_states_test.mat', 
                     mdict={ 'test_err': test_l2,
                            'T': T.numpy(),
                            'X': X.numpy(),
                            'Y': Y.numpy(),
                            'y_test': y_test.numpy(), 
                            'y_pred': pred_u.cpu().numpy(),
                            'Train_time':elapsed})  
    
    
print("\n=============================")
print('Testing error: %.3e'%(test_l2))
print("=============================\n")