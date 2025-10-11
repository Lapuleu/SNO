import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io
import time
import torch.optim
import matplotlib.pyplot as plt
import os
from timeit import default_timer
from utilities3 import MatReader, LpLoss   # keep same as LNO

################################################################
# 1D Fourier layer
################################################################
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat)
        )

    def compl_mul1d(self, input, weights):
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(
            batchsize, self.out_channels, x.size(-1)//2 + 1,
            device=x.device, dtype=torch.cfloat
        )
        out_ft[:, :, :self.modes1] = self.compl_mul1d(
            x_ft[:, :, :self.modes1], self.weights1
        )
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

################################################################
# MLP
################################################################
class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv1d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv1d(mid_channels, out_channels, 1)
    def forward(self, x):
        x = self.mlp1(x)
        x = x #F.gelu(x)
        x = self.mlp2(x)
        return x

################################################################
# FNO1d single-layer
################################################################
class FNO1d(nn.Module):
    def __init__(self, modes, width):
        super(FNO1d, self).__init__()

        """
        Single-layer FNO for 1D problems.
        Input: trajectory only (B, N)
        Output: predicted trajectory (B, N)
        """

        self.modes1 = modes
        self.width = width

        # lift input (trajectory only, 1D) -> width channels
        self.p = nn.Linear(1, self.width)

        # one Fourier layer + skip connection
        self.conv = SpectralConv1d(self.width, self.width, self.modes1)
        self.w = nn.Conv1d(self.width, self.width, 1)

        # projection back to output dim
        #self.q = MLP(self.width, 1, self.width * 2)
        self.fc1 = nn.Linear(self.width, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # make sure input is (B, N, 1)
        if x.ndim == 2:
            x = x.unsqueeze(-1)

        B, N, feats = x.shape
        x = self.p(x.view(B * N, feats))      # (B*N, width)
        x = x.view(B, N, self.width)          # (B, N, width)
        x = x.permute(0, 2, 1)                # (B, width, N)

        x1 = self.conv(x)
        x2 = self.w(x)
        x = x1+x2 #F.gelu(x1 + x2)            # (B, width, N)

        #x = self.q(x)                         # (B, out_dim, N)
        x = x.permute(0, 2, 1)                # (B, N, out_dim)
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        return x.squeeze(-1)                  # (B, N) if out_dim=1

    def get_grid(self, shape, device):
        batchsize, size_x, _ = shape
        gridx = torch.linspace(0, 1, size_x, dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat(batchsize, 1, 1)
        return gridx.to(device)
# ====================================
# saving directories
# ====================================
save_index = 1
current_directory = os.getcwd()
results_dir = os.path.join(current_directory, "Case_Fdf0_" + str(save_index))
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
save_results_to = results_dir + "/"
# ====================================
#  Define parameters and Load data
# ====================================
s = 2048

batch_size_train = 20
batch_size_vali = 20

learning_rate = 0.002
epochs = 1000
step_size = 100
gamma = 0.5

modes = 16
width = 4

reader = MatReader('DATA_PATH')
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
model = FNO1d(width,modes).cuda()


# ====================================
# Training 
# ====================================
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
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
        t=grid_x_train.cuda()
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
