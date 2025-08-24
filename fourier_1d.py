import numpy as np
import torch
import scipy
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time, os
import operator
from functools import reduce
from timeit import default_timer
from utilities3 import MatReader, LpLoss

torch.manual_seed(0)
np.random.seed(0)

# ====================================
# saving settings
# ====================================
save_index = 2
current_directory = os.getcwd()
case = "Case_F1d0_"
folder_index = str(save_index)
results_dir = os.path.join(current_directory, case + folder_index)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
save_results_to = results_dir + "/"

# ====================================
# complex multiplication
# ====================================
def compl_mul1d(a, b):
    # a: (batch, in_channel, x) complex
    # b: (in_channel, out_channel, x) complex
    return torch.einsum("bix,iox->box", a, b)

################################################################
# 1d Fourier layer
################################################################
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1

        scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    def forward(self, x):
        batchsize = x.shape[0]
        # x: (B, C, N)
        x_ft = torch.fft.rfft(x, norm="ortho")
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1] = compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)
        x = torch.fft.irfft(out_ft, n=x.size(-1), norm="ortho")
        return x

################################################################
# Simple 1D block
################################################################
class SimpleBlock1d(nn.Module):
    def __init__(self, modes, width):
        super(SimpleBlock1d, self).__init__()
        self.modes1 = modes
        self.width = width

        self.fc0 = nn.Linear(1, self.width)    # project scalar input channel → width

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.bn0 = nn.BatchNorm1d(self.width)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # x: (B, 1, N)
        x = x.permute(0, 2, 1)        # (B, N, 1)
        x = self.fc0(x)               # (B, N, width)
        x = x.permute(0, 2, 1)        # (B, width, N)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = self.bn0(x1 + x2)
        x = F.relu(x)

        x = x.permute(0, 2, 1)        # (B, N, width)
        x = self.fc1(x)               # (B, N, 128)
        x = F.relu(x)
        x = self.fc2(x)               # (B, N, 1)
        return x.squeeze(-1)          # (B, N)

class Net1d(nn.Module):
    def __init__(self, modes, width):
        super(Net1d, self).__init__()
        self.conv1 = SimpleBlock1d(modes, width)

    def forward(self, x):
        return self.conv1(x)

    def count_params(self):
        return sum(reduce(operator.mul, p.size()) for p in self.parameters())

################################################################
# Configurations
################################################################
ntrain, ntest = 1000, 100
batch_size = 20
learning_rate = 0.001
epochs = 500
step_size, gamma = 100, 0.5
modes, width = 16, 4
s = 2048

################################################################
# Load data
################################################################
reader = MatReader('/content/drive/MyDrive/FNORep/fourier_neural_operator-master/data/data.mat')
x_train = reader.read_field('f_train')
y_train = reader.read_field('u_train')
grid_x_train = reader.read_field('x_train')
x_vali = reader.read_field('f_vali')
y_vali = reader.read_field('u_vali')
grid_x_vali = reader.read_field('x_vali')
x_test = reader.read_field('f_test')
y_test = reader.read_field('u_test')
grid_x_test = reader.read_field('x_test')

# Reshape inputs to (B,1,N), outputs to (B,N)
x_train = x_train.reshape(x_train.shape[0], 1, s)
x_vali  = x_vali.reshape(x_vali.shape[0], 1, s)
x_test  = x_test.reshape(x_test.shape[0], 1, s)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
vali_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_vali, y_vali), batch_size=batch_size, shuffle=True)

################################################################
# Model
################################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net1d(modes, width).to(device)
print("Parameters:", model.count_params())

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
loss_fn = LpLoss(size_average=False)

################################################################
# Training
################################################################
train_error = np.zeros((epochs, 1))
train_loss = np.zeros((epochs, 1))
vali_error = np.zeros((epochs, 1))
vali_loss = np.zeros((epochs, 1))

for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_mse, train_l2 = 0, 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        mse = F.mse_loss(out, y, reduction='mean')
        l2 = loss_fn(out.view(batch_size, -1), y.view(batch_size, -1))
        l2.backward()
        optimizer.step()
        train_mse += mse.item()
        train_l2 += l2.item()

    scheduler.step()
    model.eval()
    test_l2, vali_mse = 0, 0
    n_vali = 0
    with torch.no_grad():
        for x, y in vali_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            test_l2 += loss_fn(out.view(batch_size, -1), y.view(batch_size, -1)).item()
            vali_mse += F.mse_loss(out, y).item()
            n_vali += 1

    train_mse /= len(train_loader)
    train_l2 /= ntrain
    test_l2 /= ntest
    vali_mse /= n_vali
    train_error[ep,0], vali_error[ep,0] = train_l2, test_l2
    train_loss[ep,0], vali_loss[ep,0] = train_mse, vali_mse

    t2 = default_timer()
    print(f"Epoch {ep}, time: {t2-t1:.3f}, Train Loss: {train_mse:.3e}, Vali Loss: {vali_mse:.3e}, Train l2: {train_l2:.4f}, Vali l2: {test_l2:.4f}")

print("\n=============================")
print("Training done.")
print("=============================\n")

# ====================================
# Saving settings (LNO style)
# ====================================
x = np.linspace(0, epochs-1, epochs)
np.savetxt(save_results_to+'/epoch.txt', x)
np.savetxt(save_results_to+'/train_loss.txt', train_loss)
np.savetxt(save_results_to+'/vali_loss.txt', vali_loss)
np.savetxt(save_results_to+'/train_error.txt', train_error)
np.savetxt(save_results_to+'/vali_error.txt', vali_error)

save_models_to = save_results_to + "model/"
if not os.path.exists(save_models_to):
    os.makedirs(save_models_to)
torch.save(model, save_models_to+'Wave_states')

pred = torch.zeros(y_test.shape)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=1, shuffle=False)
test_l2, index = 0, 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        pred[index] = out.cpu()
        test_l2 += loss_fn(out.view(1,-1), y.view(1,-1)).item()
        index += 1
test_l2 /= index

scipy.io.savemat(save_results_to+'wave_states_test.mat',
                 {'test_err': test_l2,
                  'x_test': grid_x_test.numpy(),
                  'y_test': y_test.numpy(),
                  'y_pred': pred.numpy()})

print("\n=============================")
print(f"Testing error: {test_l2:.3e}")
print("=============================\n")

# Loss history plot
epoch = np.linspace(1, epochs, epochs)
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111)
ax.plot(epoch, train_loss[:,0], color='blue', label='Train Loss')
ax.plot(epoch, vali_loss[:,0], color='red', label='Validation Loss')
ax.set_yscale('log')
ax.set_ylabel('Loss')
ax.set_xlabel('Epochs')
ax.legend(loc='upper left')
fig.savefig(save_results_to+'loss_history.png')
