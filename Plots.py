

from D1utilities3 import *
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib.axes as ax
import sys
sample = 10
### Load data
Lduffc0_data = MatReader(r"C:\Users\benze\Downloads\SNO main\SNO-main\LNOReplication\1D_Duffing_c0\Case_Dufc0_1\wave_states_test.mat")
Lduffc5_data = MatReader(r"C:\Users\benze\Downloads\SNO main\SNO-main\LNOReplication\1D_Duffing_c05\Case_Dufc05_1\wave_states_test.mat")
LLor5_data = MatReader(r"C:\Users\benze\Downloads\SNO main\SNO-main\LNOReplication\1D_Lorenz_rho5\Case_Lor5_1\wave_states_test.mat")
LLor10_data = MatReader(r"C:\Users\benze\Downloads\SNO main\SNO-main\LNOReplication\1D_Lorenz_rho10\Case_Lor10_1\wave_states_test.mat")
LPend0_data = MatReader(r"C:\Users\benze\Downloads\SNO main\SNO-main\LNOReplication\1D_Pendulum_c0\Case_Pen0_1\wave_states_test.mat")
LPend5_data = MatReader(r"C:\Users\benze\Downloads\SNO main\SNO-main\LNOReplication\1D_Pendulum_c05\Case_Pen05_1\wave_states_test.mat")
Sduffc0_data = MatReader(r"C:\Users\benze\Downloads\SNO main\SNO-main\Results\Case_Sdf0_1\wave_states_test.mat")
Sduffc5_data = MatReader(r"C:\Users\benze\Downloads\SNO main\SNO-main\Results\Case_Sdf05_1\wave_states_test.mat")
SLor5_data = MatReader(r"C:\Users\benze\Downloads\SNO main\SNO-main\Results\Case_SLor5_1\wave_states_test.mat")
SLor10_data = MatReader(r"C:\Users\benze\Downloads\SNO main\SNO-main\Results\Case_SLor10_1\wave_states_test.mat")
SPend0_data = MatReader(r"C:\Users\benze\Downloads\SNO main\SNO-main\Results\Case_SPen0_1\wave_states_test.mat")
SPend5_data = MatReader(r"C:\Users\benze\Downloads\SNO main\SNO-main\Results\Case_SPen05_1\wave_states_test.mat")
Fduffc0_data = MatReader(r"C:\Users\benze\Downloads\SNO main\SNO-main\FNOReplication\Case_Fdf0_1\wave_states_test.mat")
Fduffc5_data = MatReader(r"C:\Users\benze\Downloads\SNO main\SNO-main\FNOReplication\Case_Fdf05_1\wave_states_test.mat")
FLor5_data = MatReader(r"C:\Users\benze\Downloads\SNO main\SNO-main\FNOReplication\Case_FLor05_1\wave_states_test.mat")
FLor10_data = MatReader(r"C:\Users\benze\Downloads\SNO main\SNO-main\FNOReplication\Case_FLor10_1\wave_states_test.mat")
FPend0_data = MatReader(r"C:\Users\benze\Downloads\SNO main\SNO-main\FNOReplication\Case_FPen0_1\wave_states_test.mat")
FPend5_data = MatReader(r"C:\Users\benze\Downloads\SNO main\SNO-main\FNOReplication\Case_FPen05_1\wave_states_test.mat")

Lduffc0_data_pointwise_prediction = Lduffc0_data.read_field('y_test')-Lduffc0_data.read_field('y_pred')
Lduffc5_data_pointwise_prediction = Lduffc5_data.read_field('y_test')-Lduffc5_data.read_field('y_pred')
LLor5_data_pointwise_prediction = LLor5_data.read_field('y_test')-LLor5_data.read_field('y_pred')
LLor10_data_pointwise_prediction = LLor10_data.read_field('y_test')-LLor10_data.read_field('y_pred')
LPend0_data_pointwise_prediction = LPend0_data.read_field('y_test')-LPend0_data.read_field('y_pred')             
LPend5_data_pointwise_prediction = LPend5_data.read_field('y_test')-LPend5_data.read_field('y_pred')
Sduffc0_data_pointwise_prediction = Sduffc0_data.read_field('y_test')-Sduffc0_data.read_field('y_pred')
Sduffc5_data_pointwise_prediction = Sduffc5_data.read_field('y_test')-Sduffc5_data.read_field('y_pred')
SLor5_data_pointwise_prediction = SLor5_data.read_field('y_test')-SLor5_data.read_field('y_pred')
SLor10_data_pointwise_prediction = SLor10_data.read_field('y_test')-SLor10_data.read_field('y_pred')
SPend0_data_pointwise_prediction = SPend0_data.read_field('y_test')-SPend0_data.read_field('y_pred')
SPend5_data_pointwise_prediction = SPend5_data.read_field('y_test')-SPend5_data.read_field('y_pred')
Fduffc0_data_pointwise_prediction = Fduffc0_data.read_field('y_test')-Fduffc0_data.read_field('y_pred')
Fduffc5_data_pointwise_prediction = Fduffc5_data.read_field('y_test')-Fduffc5_data.read_field('y_pred')
FLor5_data_pointwise_prediction = FLor5_data.read_field('y_test')-FLor5_data.read_field('y_pred')
FLor10_data_pointwise_prediction = FLor10_data.read_field('y_test')-FLor10_data.read_field('y_pred')
FPend0_data_pointwise_prediction = FPend0_data.read_field('y_test')-FPend0_data.read_field('y_pred')
FPend5_data_pointwise_prediction = FPend5_data.read_field('y_test')-FPend5_data.read_field('y_pred')

duffc0_truth = Lduffc0_data.read_field('y_test')
duffc5_truth = Lduffc5_data.read_field('y_test')
Lor5_truth = LLor5_data.read_field('y_test')
Lor10_truth = LLor10_data.read_field('y_test')
Pend0_truth = LPend0_data.read_field('y_test')
Pend5_truth = LPend5_data.read_field('y_test')
grid = Lduffc0_data.read_field('x_test')

### Plotting (Plotly)
fig = make_subplots(rows=6, cols=4)
fig.add_trace(
    go.Scatter(y=duffc0_truth[sample,:], x=grid.squeeze()),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(y=Sduffc0_data_pointwise_prediction[sample,:], x=grid.squeeze()),
    row=1, col=2
)

fig.add_trace(
    go.Scatter(y=Fduffc0_data_pointwise_prediction[sample,:], x=grid.squeeze()),
    row=1, col=3
)

fig.add_trace(
    go.Scatter(y=Lduffc0_data_pointwise_prediction[sample,:], x=grid.squeeze()),
    row=1, col=4
)

fig.add_trace(
    go.Scatter(y=duffc5_truth[sample,:], x=grid.squeeze()),
    row=2, col=1
)

fig.add_trace(
    go.Scatter(y=Sduffc5_data_pointwise_prediction[sample,:], x=grid.squeeze()),
    row=2, col=2
)

fig.add_trace(
    go.Scatter(y=Fduffc5_data_pointwise_prediction[sample,:], x=grid.squeeze()),
    row=2, col=3
)

fig.add_trace(
    go.Scatter(y=Lduffc5_data_pointwise_prediction[sample,:], x=grid.squeeze()),
    row=2, col=4
)
fig.add_trace(
    go.Scatter(y=Lor5_truth[sample,:], x=grid.squeeze()),
    row=3, col=1
)

fig.add_trace(
    go.Scatter(y=SLor5_data_pointwise_prediction[sample,:], x=grid.squeeze()),
    row=3, col=2
)

fig.add_trace(
    go.Scatter(y=FLor5_data_pointwise_prediction[sample,:], x=grid.squeeze()),
    row=3, col=3
)
fig.add_trace(
    go.Scatter(y=LLor5_data_pointwise_prediction[sample,:], x=grid.squeeze()),
    row=3, col=4
)
fig.add_trace(
    go.Scatter(y=Lor10_truth[sample,:], x=grid.squeeze()),
    row=4, col=1
)

fig.add_trace(
    go.Scatter(y=SLor10_data_pointwise_prediction[sample,:], x=grid.squeeze()),
    row=4, col=2
)

fig.add_trace(
    go.Scatter(y=FLor10_data_pointwise_prediction[sample,:], x=grid.squeeze()),
    row=4, col=3
)
fig.add_trace(
    go.Scatter(y=LLor10_data_pointwise_prediction[sample,:], x=grid.squeeze()),
    row=4, col=4
)

fig.add_trace(
    go.Scatter(y=Pend0_truth[sample,:], x=grid.squeeze()),
    row=5, col=1
)

fig.add_trace(
    go.Scatter(y=SPend0_data_pointwise_prediction[sample,:], x=grid.squeeze()),
    row=5, col=2
)

fig.add_trace(
    go.Scatter(y=FPend0_data_pointwise_prediction[sample,:], x=grid.squeeze()),
    row=5, col=3
)

fig.add_trace(
    go.Scatter(y=LPend0_data_pointwise_prediction[sample,:], x=grid.squeeze()),
    row=5, col=4
)
fig.add_trace(
    go.Scatter(y=Pend5_truth[sample,:], x=grid.squeeze()),
    row=6, col=1
)

fig.add_trace(
    go.Scatter(y=SPend5_data_pointwise_prediction[sample,:], x=grid.squeeze()),
    row=6, col=2
)

fig.add_trace(
    go.Scatter(y=FPend5_data_pointwise_prediction[sample,:], x=grid.squeeze()),
    row=6, col=3
)

fig.add_trace(
    go.Scatter(y=LPend5_data_pointwise_prediction[sample,:], x=grid.squeeze()),
    row=6, col=4
)
fig.update_layout(
    height=800, 
    width=800,
    title = "ODE Pointwise Prediction Errors",
    showlegend=False
)

fig.write_image(r"C:\Users\benze\Downloads\line_plot.png")

###Plotting (Matplotlib)
models = ['Ground Truth', 'SNO', 'FNO', 'LNO']
models_data = ['','S', 'F', 'L']
tasks = ['Duffing', 'Lorenz', 'Pendulum']
coefficients_Duffing = ['c=0', 'c=0.5']
coefficients_Lorenz = ['ρ=5', 'ρ=10']
coefficients_Pendulum = ['c=0', 'c=0.5']
tasks_data = ['duffc0', 'duffc5', 'Lor5', 'Lor10', 'Pend0', 'Pend5']
datas = ['_truth', '_data_pointwise_prediction', '_data_pointwise_prediction', '_data_pointwise_prediction']

fig = plt.figure(constrained_layout=False, figsize=(30,40))
fig.suptitle("ODE Pointwise Prediction Errors", fontsize=40)
subfigs = fig.subfigures(nrows=3,ncols=1)
subfigs_list = enumerate(subfigs)
for (row, subfig), task in zip(subfigs_list, tasks):
       subfig.supylabel(f"{task}" , fontsize=30)
       subfigs2 = subfig.subfigures(nrows=2,ncols=1)
       for subfig2, task_data, coeff in zip(subfigs2, tasks_data, globals()[f"coefficients_{task}"]):
                subfig2.supylabel(f"{coeff}", fontsize=30)
                axs = subfig2.subplots(nrows=1,ncols=4, width_ratios=[.5,.5,.5,.5])
                for ax, model_data, data, model in zip(axs, models_data, datas, models):
                    ax.plot(grid.squeeze(), globals()[f"{model_data}{task_data}{data}"][sample,:])
                    if row == 0:
                        ax.set_title(f"{model}")
fig.savefig(r"C:\Users\benze\Downloads\line_plot_matplotlib.png")
### Heatmap plots
from D1utilities3 import *
import plotly.graph_objs as go
import plotly.express as px
import pandas
from plotly.subplots import make_subplots
### Load data
LNOBeam_data = MatReader(r"C:\Users\benze\Downloads\SNO main\SNO-main\LNOReplication\2D_Beam\Case_Beam_1\wave_states_test.mat")
LNOBurger_data = MatReader(r"C:\Users\benze\Downloads\SNO main\SNO-main\LNOReplication\2D_Burger\Case_Burger_2\wave_states_test.mat")
LNORDif_data = MatReader(r"C:\Users\benze\Downloads\SNO main\SNO-main\LNOReplication\2D_Reac_diffusion\Case_RDif_1\wave_states_test.mat")
LNODif_data = MatReader(r"C:\Users\benze\Downloads\SNO main\SNO-main\LNOReplication\2D_Diffusion\Case_Diff_1\wave_states_test.mat")
SNOBeam_data = MatReader(r"C:\Users\benze\Downloads\SNO main\SNO-main\Results\Case_SBeam_1\wave_states_test.mat")
SNOBurger_data = MatReader(r"C:\Users\benze\Downloads\SNO main\SNO-main\Results\Case_SBurger_1\wave_states_test.mat")
SNORDif_data = MatReader(r"C:\Users\benze\Downloads\SNO main\SNO-main\Results\Case_SRdiff_1\wave_states_test.mat")
SNODif_data = MatReader(r"C:\Users\benze\Downloads\SNO main\SNO-main\Results\Case_Sdiff_1\wave_states_test.mat")
FNOBeam_data = MatReader(r"C:\Users\benze\Downloads\SNO main\SNO-main\FNOReplication\Case_FBeam_1\wave_states_test.mat")
FNOBurger_data = MatReader(r"C:\Users\benze\Downloads\SNO main\SNO-main\FNOReplication\Case_FBurger_1\wave_states_test.mat")
FNORDif_data = MatReader(r"C:\Users\benze\Downloads\SNO main\SNO-main\FNOReplication\Case_FRdiff_1\wave_states_test.mat")
FNODif_data = MatReader(r"C:\Users\benze\Downloads\SNO main\SNO-main\FNOReplication\Case_Fdiff_1\wave_states_test.mat")

Beam_truth = LNOBeam_data.read_field('y_test')
Burger_truth = LNOBurger_data.read_field('y_test')
RDif_truth = LNORDif_data.read_field('y_test')
Dif_truth = LNODif_data.read_field('y_test')

LNOBeam_pred = LNOBeam_data.read_field('y_pred')-Beam_truth
LNOBurger_pred = LNOBurger_data.read_field('y_pred')-Burger_truth
LNORDif_pred = LNORDif_data.read_field('y_pred')-RDif_truth
LNODif_pred = LNODif_data.read_field('y_pred')-Dif_truth

SNOBeam_pred = SNOBeam_data.read_field('y_pred')-Beam_truth
SNOBurger_pred = SNOBurger_data.read_field('y_pred')-Burger_truth
SNORDif_pred = SNORDif_data.read_field('y_pred')-RDif_truth
SNODif_pred = SNODif_data.read_field('y_pred')-Dif_truth

FNOBeam_pred = FNOBeam_data.read_field('y_pred')-Beam_truth
FNOBurger_pred = FNOBurger_data.read_field('y_pred')-Burger_truth
FNORDif_pred = FNORDif_data.read_field('y_pred')-RDif_truth
FNODif_pred = FNODif_data.read_field('y_pred')-Dif_truth

###axes
T_Beam = LNOBeam_data.read_field('T')
X_Beam = LNOBeam_data.read_field('X')
T_Burger = LNOBurger_data.read_field('T')
X_Burger = LNOBurger_data.read_field('X')
T_RDif = LNORDif_data.read_field('T')
X_RDif = LNORDif_data.read_field('X')
T_Dif = LNODif_data.read_field('T')
X_Dif = LNODif_data.read_field('X')
print(FNORDif_pred.shape)
### Plotting
sample = 10
colorscale = 'Viridis'
models = ['Ground Truth', 'SNO', 'FNO', 'LNO']
models_data = ['','SNO', 'FNO', 'LNO']
tasks = ['Beam', "Burger's", 'Reaction Diffusion', 'Diffusion']
tasks_data = ['Beam', 'Burger', 'RDif', 'Dif']
datas = ['_truth', '_pred', '_pred', '_pred']

fig = plt.figure(constrained_layout=True, figsize=(8.5, 11))
fig.suptitle("PDE Pointwise Prediction Errors", fontsize=30)
subfigs = fig.subfigures(nrows=4,ncols=1)
subfigs_list = enumerate(subfigs)
for (row, subfig), task_data, task in zip(subfigs_list, tasks_data, tasks):
       subfig.supylabel(f"{task}" , fontsize=20)
       for ax, model_data, data, model in zip(subfig.subplots(nrows=1,ncols=4), models_data, datas, models):
            im = ax.imshow(globals()[f"{model_data}{task_data}{data}"][sample,:,:], cmap='jet', aspect='auto')
            if row == 0:
                ax.set_title(f"{model}")
            cbar = fig.colorbar(im, ax=ax)
            cbar.ax.tick_params(labelsize=10)
            ax.tick_params(axis='both', which='major', labelsize=10)

fig.savefig(r"C:\Users\benze\Downloads\heatmap_plot_matplotlib.png")