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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# ====================================
# saving settings
# ====================================
save_index = 1   
current_directory = os.getcwd()
case = "Case_SDuf0_"
folder_index = str(save_index)

results_dir = "/" + case + folder_index +"/"
save_results_to = current_directory + results_dir
if not os.path.exists(save_results_to):
    os.makedirs(save_results_to)
# ====================================
# Data Gen
# ====================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.special as sp

class Sumudu_Transform(nn.Module):
    def __init__(self, in_channels, out_channels, degree, width, s, device=device):
        super(Sumudu_Transform, self).__init__()
        self.degree = degree
        self.width = width
        self.s = s
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.device = device

        self.flip = (-1)**torch.randint(0, 2, (in_channels, out_channels), device=device)
        self.scale = 1.0 / (in_channels * out_channels)
        self.weight1 = nn.Parameter(self.scale * torch.rand((in_channels, out_channels), dtype=torch.float64, device=device))

    def coefficient_training(self, input, degree):
        B, C, D = input.shape
        input = input.double() 
        x_lin = torch.linspace(0, 1, D, dtype=torch.float64, device=self.device)
        powers = torch.arange(degree-1, -1, -1, dtype=torch.float64, device=self.device)  # highest power first

        # Vandermonde matrix [D, degree]
        vander = x_lin.unsqueeze(1) ** powers.unsqueeze(0)  # [D, degree]

        # Solve least squares for all batches at once
        input_flat = input.reshape(B*C, D).T  # [D, B*C]
        coeffs = torch.linalg.lstsq(vander, input_flat).solution  # [degree, B*C]

        coeffs = coeffs.T.reshape(B, C, degree)  # [B, C, degree]
        return coeffs

    def transform(self, input):
        # Ensure factorial length matches input's last dimension
        fact = torch.tensor(sp.factorial(np.arange(input.shape[2], dtype=np.float64)), 
                            dtype=torch.float64, device=input.device)
        return input.double() * fact.view(1, 1, -1)

    def inverse_transform(self, input):
        fact = torch.tensor(sp.factorial(np.arange(input.shape[2], dtype=np.float64)), 
                            dtype=torch.float64, device=input.device)
        return input.double() / fact.view(1, 1, -1)

    def approximate_sum(self, width, input):
        B, C, degree = input.shape
        # Discretize [0,1] domain
        x_lin = torch.linspace(0, 1, self.s, dtype=torch.float64, device=self.device)  # [s]
        powers = torch.arange(degree-1, -1, -1, dtype=torch.float64, device=self.device)  # [degree]

        # x_lin^powers -> [s, degree]
        x_pow = x_lin.unsqueeze(1) ** powers.unsqueeze(0)  # [s, degree]

        # Multiply coefficients by x_pow
        input_flat = input.reshape(B*C, degree)  # [B*C, degree]
        output = input_flat @ x_pow.T  # [B*C, s]
        output = output.reshape(B, C, self.s)
        return output

    def weight_mul(self, weight1, input):
        return torch.einsum("bix,io->box", input, weight1.double())

    def forward(self, x):
        x = self.coefficient_training(x, self.degree)
        x = self.transform(x)
        x = self.approximate_sum(self.width, x)
        x = self.weight_mul(self.weight1, x)
        x = self.coefficient_training(x, self.degree)
        x = self.inverse_transform(x)
        x = self.approximate_sum(self.width, x).float()
        return x
    
class SNO1d(nn.Module):
    def __init__(self, degree, width, s):
        super(SNO1d, self).__init__()
        self.degree = degree
        self.s = s
        self.width = width

        self.fc0 = nn.Linear(1, self.width)

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
        x = self.bn0(x1 + x2)

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = torch.tanh(x)
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
s = 2048
batch_size_train = 20
batch_size_vali = 20
epochs = 1000
step_size = 100
gamma = 0.5
learning_rate = .001
width = 16
degree = 4

reader = MatReader('C:\\Users\\benze\\Downloads\\SNO main\\SNO-main\\LNOReplication\\1D_Duffing_c0\\Data\\data.mat')
x_train = reader.read_field('f_train')
y_train = reader.read_field('u_train')
grid_x_train = reader.read_field('x_train')

x_vali = reader.read_field('f_vali')
y_vali = reader.read_field('u_vali')
grid_x_vali = reader.read_field('x_vali')

x_test = reader.read_field('f_test')
y_test = reader.read_field('u_test')
grid_x_test = reader.read_field('x_test') 

x_train = x_train.reshape(x_train.shape[0],s,1)
x_vali = x_vali.reshape(x_vali.shape[0],s,1)
x_test = x_test.reshape(x_test.shape[0],s,1)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size_train, shuffle=True)
vali_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_vali, y_vali), batch_size=batch_size_vali, shuffle=True)

# model
model = SNO1d(degree, width,s).to(device)

# ====================================
# Training 
# ====================================
torch.autograd.set_detect_anomaly(True)
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
start_time = time.time()
myloss = LpLoss(size_average=True)

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
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        t=grid_x_train.to(device, non_blocking=True)
        optimizer.zero_grad()
        out = model(x)   
        mse = F.mse_loss(out.view(batch_size_train, -1), y.view(batch_size_train, -1), reduction='mean')
        l2 = myloss(out.view(batch_size_train, -1), y.view(batch_size_train, -1))
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
            t=grid_x_vali.cuda()
            out = model(x)
            mse=F.mse_loss(out.view(batch_size_vali, -1), y.view(batch_size_vali, -1), reduction='mean')
            vali_l2 += myloss(out.view(batch_size_vali, -1), y.view(batch_size_vali, -1)).item()
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
    t2 = default_timer()
    print("Epoch: %d, time: %.3f, Train Loss: %.3e,Vali Loss: %.3e, Train l2: %.4f, Vali l2: %.4f" % (ep, t2-t1, train_mse, vali_mse,train_l2, vali_l2))
elapsed = time.time() - start_time
print("\n=============================")
print("Training done...")
print('Training time: %.3f'%(elapsed))
print("=============================\n")

# ====================================
# saving settings
# ====================================
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
pred = torch.zeros(y_test.shape)
index = 0
test_l2 = 0.0
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test),
                                        batch_size=1, shuffle=False)

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.cuda(), y.cuda()
        t=grid_x_test.cuda()
        out = model(x).view(1,-1)
        pred[index]= out
        test_l2 += myloss(out, y).item()
        index = index + 1
test_l2/=index

scipy.io.savemat(save_results_to+'wave_states_test.mat', 
                    mdict={'test_err': test_l2,
                            'x_test': grid_x_train.numpy(),
                            'y_test': y_test.numpy(), 
                            'y_pred': pred.cpu().numpy()})  
    
        
    
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





