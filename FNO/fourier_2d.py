import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
from timeit import default_timer
from utilities3 import *  # your existing helper utilities
import os
import time
import scipy
import matplotlib.pyplot as plt

torch.manual_seed(0)
np.random.seed(0)

# ====================================
# saving settings
# ====================================
save_index = 1   
current_directory = os.getcwd()
case = "Case_FBeam_"
folder_index = str(save_index)

results_dir = "/" + case + folder_index +"/"
save_results_to = current_directory + results_dir
if not os.path.exists(save_results_to):
    os.makedirs(save_results_to)

################################################################
# complex multiplication
################################################################
def compl_mul2d(a: torch.Tensor, b: torch.Tensor):
    """
    (batch,in_c,x,y) * (in_c,out_c,x,y) -> (batch,out_c,x,y)
    a : complex tensor
    b : complex tensor
    """
    # torch.fft already returns complex tensors -> plain einsum works
    return torch.einsum("bixy,ioxy->boxy", a, b)

################################################################
# Fourier layer using the new torch.fft API
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = 1 / (in_channels * out_channels)

        # complex weights
        self.weights1 = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def forward(self, x):
        """
        x : (batch, in_channels, H, W), real
        returns real inverse FFT
        """
        batchsize, _, H, W = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")          # -> (batch,in_c,H,W//2+1), complex

        out_ft = torch.zeros(batchsize, self.out_channels, H, W//2 + 1,
                             dtype=torch.cfloat, device=x.device)

        out_ft[:, :, :self.modes1, :self.modes2] = \
            compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        x = torch.fft.irfft2(out_ft, s=(H, W), norm="ortho").real
        return x

################################################################
# Model blocks
################################################################
class SimpleBlock2d(nn.Module):
    def __init__(self, modes1, modes2, width):
        super().__init__()
        self.modes1, self.modes2, self.width = modes1, modes2, width
        self.fc0 = nn.Linear(3, width)

        self.conv0 = SpectralConv2d(width, width, modes1, modes2)
        self.w0    = nn.Conv1d(width, width, 1)
        self.bn0   = nn.BatchNorm2d(width)

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        batchsize, size_x, size_y, _ = x.shape
        x = self.fc0(x)                       # (b,s,s,width)
        x = x.permute(0, 3, 1, 2)              # (b,width,s,s)

        x1 = self.conv0(x)
        x2 = self.w0(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x  = self.bn0(x1 + x2)

        x  = x.permute(0, 2, 3, 1)             # (b,s,s,width)
        x  = F.relu(self.fc1(x))
        x  = self.fc2(x)
        return x

class Net2d(nn.Module):
    def __init__(self, modes, width):
        super().__init__()
        self.conv1 = SimpleBlock2d(modes, modes, width)

    def forward(self, x):
        return self.conv1(x).squeeze()

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

################################################################
# data / training config
################################################################
TRAIN_PATH = '/content/drive/MyDrive/LNORep/2D_Beam/Data/data.mat'
TEST_PATH  = '/content/drive/MyDrive/LNORep/2D_Beam/Data/data.mat'

ntrain, ntest = 200, 130
batch_size    = 20
learning_rate = 0.001
epochs        = 1000
step_size     = 100
gamma         = 0.5
modes, width  = 12, 32
s             = 50   # spatial resolution

################################################################
# load data
################################################################
reader = MatReader(TRAIN_PATH)
x_train = reader.read_field('f_train')
y_train = reader.read_field('u_train')
T = reader.read_field('t')
X = reader.read_field('x')

reader.load_file(TEST_PATH)
x_test = reader.read_field('f_test')
y_test = reader.read_field('u_test')

x_normalizer = UnitGaussianNormalizer(x_train)
x_train = x_normalizer.encode(x_train)
x_test  = x_normalizer.encode(x_test)

y_normalizer = UnitGaussianNormalizer(y_train)
y_train = y_normalizer.encode(y_train)

# add coordinates
gridx = np.linspace(0, 1, s)
gridy = np.linspace(0, 1, s)
grid  = np.stack(np.meshgrid(gridx, gridy, indexing="ij"), axis=-1)  # (s,s,2)
grid  = torch.tensor(grid, dtype=torch.float).unsqueeze(0)           # (1,s,s,2)

x_train = torch.cat([x_train.reshape(ntrain, s, s, 1), grid.repeat(ntrain,1,1,1)], dim=3)
x_test  = torch.cat([x_test.reshape(ntest, s, s, 1), grid.repeat(ntest,1,1,1)], dim=3)

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x_train, y_train),
    batch_size=batch_size, shuffle=True)
test_loader  = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x_test, y_test),
    batch_size=batch_size, shuffle=False)

################################################################
# training
################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x_normalizer.mean = x_normalizer.mean.to(device)
x_normalizer.std  = x_normalizer.std.to(device)

y_normalizer.mean = y_normalizer.mean.to(device)
y_normalizer.std  = y_normalizer.std.to(device)

model = Net2d(modes, width).to(device)
print("Parameter count:", model.count_params())

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
loss_fn   = LpLoss(size_average=False)
start_time = time.time()

train_error = np.zeros((epochs, 1))
train_loss = np.zeros((epochs, 1))
vali_error = np.zeros((epochs, 1))
vali_loss = np.zeros((epochs, 1))
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_mse = 0
    train_l2 = 0
    train_ls = 0.0
    n_train = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        out = y_normalizer.decode(out)
        y   = y_normalizer.decode(y)
        mse = F.mse_loss(out.view(batch_size, -1), y.view(batch_size, -1), reduction='mean')
        loss = loss_fn(out.view(batch_size, -1), y.view(batch_size, -1))
        loss.backward()
        optimizer.step()
        train_ls += loss.item()
        train_mse += mse.item()
        train_l2 += loss.item()
        n_train += 1

    scheduler.step()
    train_ls /= ntrain

    # evaluation
    model.eval()
    rel_err = 0.0
    vali_mse = 0.0
    vali_l2 = 0.0
    with torch.no_grad():
        n_vali = 0
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out  = y_normalizer.decode(model(x))
            rel_err += loss_fn(out.view(batch_size,-1), y.view(batch_size,-1)).item()
            mse=F.mse_loss(out.view(batch_size, -1), y.view(batch_size, -1), reduction='mean')
            vali_l2 += loss_fn(out.view(batch_size, -1), y.view(batch_size, -1)).item()
            vali_mse += mse.item()
            n_vali += 1
    rel_err /= ntest

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
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=1, shuffle=False)
pred_u = torch.zeros(ntest,y_test.shape[1],y_test.shape[2])
index = 0
test_l2 = 0.0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.cuda(), y.cuda()
        out = model(x)
        test_l2 += loss_fn(out.view(-1,x_test.shape[1],x_test.shape[2]), y).item()
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