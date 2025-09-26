

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
'''def get_grid(f):
    x_k = torch.linspace(0, degree-2, degree-1, dtype=torch.double, device=f.device)
    x_n = -torch.cos(torch.pi*(x_k+.5)/degree)
    x = (f.shape[-1]-1)*.01+(f.shape[-1]-1)*.01*x_n
    f = torch.index_select(f,-1,x.long())
    return f
def fct(f):
    # Reverse rows like f(end:-1:1,:) in MATLAB
    f = f.flip(dims=(2,))

    # Get dimensions
    A = f.shape
    N = A[2]
    a = torch.zeros_like(f, dtype=float)
    a = math.sqrt(2 / N) * dct(f,norm=None)
    a[0, :, :] /= math.sqrt(2)

    return a
def ifct(a, l=None):
    # Get dimensions
    k = a.shape
    N = k[0]

    # Adjust scaling: first row multiplied by sqrt(2), others unchanged
    a[:,:,0] = math.sqrt(2) * a[:,:,0]

    # Apply inverse DCT (along axis 0)
    f = idct(math.sqrt(N / 2) * a,norm=None)
    # Reverse rows like f(end:-1:1,:) in MATLAB
    f = f.flip(dims=(2,))

    return f'''
# ====================================
# saving settings
# ====================================
save_index = 1   
current_directory = os.getcwd()
case = "Case_"
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
        super(Sumudu_Transform, self).__init__()
        
        self.degree = degree
        self.width = width
        self.s = s
        self.in_channels = in_channels
        self.out_channels = out_channels



        self.flip = (-1)**torch.randint(0,2,(in_channels, out_channels))
        self.scale = (1 / (in_channels*out_channels))
        self.weight1 = nn.Parameter((self.scale*torch.rand((in_channels, out_channels), dtype=torch.float64)))
        self.weight2 = nn.Parameter((self.scale*torch.rand((in_channels, out_channels), dtype=torch.float64)))

    def coefficient_training(self, input, degree):
        output = input.reshape(input.shape[0]*input.shape[1]*input.shape[2], input.shape[3]).permute(1,0)
        output = torch.from_numpy(np.polyfit(np.linspace(0, (s-1)*.02, s, dtype=np.float64), output.detach().cpu().numpy(), degree-1)).to('cuda').float().permute(1,0)
        output = output.reshape(input.shape[0], input.shape[1], input.shape[2], degree)    
        

        return output
    def transform(self, input):
        fact = torch.from_numpy(sp.factorial(np.linspace(0, input.shape[3]-1, input.shape[3], dtype=np.float64))).to('cuda').flip((0,))
        output = torch.mul(input, fact)
        return output
    def inverse_transform(self, input):
        ##input = input.reshape(-1, input.shape[1]*input.shape[2]).detach().cpu().numpy()
        ##output = np.zeros((input.shape[0], s*width), dtype= np.double)
        ##for i in range(input.shape[0]):
        ##    output[i, :] = chebyshev.cheby_idct(input[i,:])
        ##factorial = torch.from_numpy(sp.factorial(np.linspace(0,s*width-1, s*width, dtype= np.double))).to('cuda').double()
        ##output = torch.from_numpy(output).to('cuda').double()
        ##output = torch.mul(output, factorial)
        ##output = ifct(input)
        fact = torch.from_numpy(sp.factorial(np.linspace(0, input.shape[3]-1, input.shape[3], dtype=np.float64))).to('cuda').flip((0,))
        output = torch.div(input, fact)
        return output

    def approximate_sum(self, width, input):
        '''discretization = torch.linspace(0, (s)*.01, steps = s, dtype=torch.float, device=input.device)
        row_powers = torch.linspace(1, input.shape[2], steps = input.shape[2], dtype=torch.float, device=input.device).unsqueeze(1)
        discretization = discretization**row_powers
        output = torch.einsum("aid,dx->aix", input, discretization)
        output = output.reshape(input.shape[0], s, width)
        output = output.permute(0,2,1)
        output = torch.zeros((input.shape[0], width, s), dtype= torch.float, device=input.device)
        for i in range(input.shape[0]):
            for j in range(width):
                output[i,j,:] = torch.from_numpy(chebyshev.cheby_sum(np.linspace(0, (s-1)*.01, s, dtype=float),input[i,j,:].detach().cpu().numpy(), 0,((s-1)*.01))).to('cuda').float()
'''     
        x = np.linspace(0, (s-1)*.02, s, dtype=np.float64)
        output = torch.zeros(input.shape[0], input.shape[1], input.shape[2], s, dtype= torch.float64, device=input.device)
        for i in range(input.shape[0]):
            for k in range(input.shape[1]):
                for n in range(input.shape[2]):
                    output[i,k,n,:] = torch.from_numpy(np.polyval(input[i,k,n,:].detach().cpu().numpy(), x))
        return output


    def weight_mul(self, input):
        transformed = torch.zeros_like(input, device= input.device)
        for i in range(input.shape[3]):
            transformed[:,:,:,i] = torch.einsum("bix,io -> box", input[:,:,:,i], self.weight1)
        for i in range(input.shape[2]):
            transformed[:,:,i,:] += transformed[:,:,i,:]*torch.einsum("bit,io->bot", input[:,:,i,:], self.weight2)        
        
        return transformed
    def forward(self, x):
        x = self.coefficient_training(x, self.degree)
        x = self.transform(x)
        x = self.approximate_sum(self.width, x)
        x = self.weight_mul(x)
        x = self.coefficient_training(x, self.degree)
        x = self.inverse_transform(x)


        x = self.approximate_sum(self.width, x).float()


        return x


class SNO2d(nn.Module):
    def __init__(self, width, s):
        super(SNO2d, self).__init__()

        self.width = width
        self.fc0 = nn.Linear(3, self.width) 

        self.conv0 = Sumudu_Transform(self.width, self.width, degree, self.width, s)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.norm = nn.InstanceNorm2d(self.width)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self,x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        x1 = self.norm(self.conv0(self.norm(x)))
        x2 = self.w0(x)
        x = x1 +x2
        x = F.leaky_relu(x)
        x1 = self.norm(self.conv0(self.norm(x)))
        x2 = self.w1(x)
        x = x1+x2

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = torch.sin(x)
        x = self.fc2(x)
        return x


  
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

# ====================================
#  Define parameters and Load data
# ====================================
degree = 15
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

reader = MatReader(r'C:\Users\benze\Downloads\SNO main\Beam_SNO\Data\data.beam.mat')
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
model = SNO2d(width, s).cuda()

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

# ====================================
# saving settings
# ====================================
current_directory = os.getcwd()
case = "Case_Beam_"
save_index = 1  
folder_index = str(save_index)

results_dir = "/" + case + folder_index +"/"
save_results_to = current_directory + results_dir
if not os.path.exists(save_results_to):
    os.makedirs(save_results_to)

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
