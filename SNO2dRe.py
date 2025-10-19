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

        # constant grid and factorial (factorial can still be cached safely)
        dtype = torch.float64
        self.register_buffer('x_grid',torch.linspace(0, (s-1)*.02, s, dtype=dtype))
        fact = torch.exp(torch.lgamma(torch.linspace(0, s-1, s, dtype=dtype)+1))
        self.register_buffer('factorial', fact)

    def coefficient_training(self, input, degree):
        """
        Fit polynomial coefficients each call using a fresh Vandermonde and pinv.
        """
        B, N, M, S = input.shape
        x_lin = torch.linspace(0, (M-1)*.01, M, dtype=torch.float64)
        y = input.reshape(-1, S).double()           # [B*N*M, s]
        V = torch.vander(x_lin, N=degree, increasing=False)  # [s, degree]
        pinv = torch.linalg.pinv(V, rcond=1e-4)                      # [degree, s]
        coef = (pinv @ y.T).T                                        # [B*N*M, degree]
        return coef.reshape(B, N, M, degree)

    def transform(self, input):
        return input * self.factorial[:input.shape[3]]

    def inverse_transform(self, input):
        return input / self.factorial[:input.shape[3]]

    def approximate_sum(self, width, input):
        B, N, M, D = input.shape
        V = torch.vander(self.x_grid, N=D, increasing=False)   # [s, degree]
        return torch.einsum('b n m d, s d -> b n m s', input, V)

    def weight_mul(self, input):
        t1 = torch.einsum("b i x s, i o -> b o x s", input, self.weight1)
        t2 = torch.einsum("b i t s, i o -> b o t s", input, self.weight2)
        return t1 + t1 * t2

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
width = 16
degree = 4

reader = MatReader('/workspace/gridSearch/Data/Beam/data.mat')
x_train = reader.read_field('f_train')
y_train = reader.read_field('u_train')
T = reader.read_field('t')
X = reader.read_field('x')

x_vali = reader.read_field('f_vali')
y_vali = reader.read_field('u_vali')

x_test = reader.read_field('f_test')
y_test = reader.read_field('u_test')

x_train = x_train.reshape(ntrain,x_train.shape[1],x_train.shape[2],1)
x_vali = x_vali.reshape(nvali,x_vali.shape[1],x_vali.shape[2],1)
x_test = x_test.reshape(ntest,x_test.shape[1],x_test.shape[2],1)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size_train, shuffle=False)
vali_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_vali, y_vali), batch_size=batch_size_vali, shuffle=False)

# model
model = SNO2d(degree, width,s).to(device)

# ====================================
# Training 
# ====================================
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
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
