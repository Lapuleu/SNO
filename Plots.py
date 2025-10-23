

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
Llor5_data_pointwise_prediction = LLor5_data.read_field('y_test')-LLor5_data.read_field('y_pred')
Llor10_data_pointwise_prediction = LLor10_data.read_field('y_test')-LLor10_data.read_field('y_pred')
Lpend0_data_pointwise_prediction = LPend0_data.read_field('y_test')-LPend0_data.read_field('y_pred')             
Lpend5_data_pointwise_prediction = LPend5_data.read_field('y_test')-LPend5_data.read_field('y_pred')
Sduffc0_data_pointwise_prediction = Sduffc0_data.read_field('y_test')-Sduffc0_data.read_field('y_pred')
Sduffc5_data_pointwise_prediction = Sduffc5_data.read_field('y_test')-Sduffc5_data.read_field('y_pred')
Slor5_data_pointwise_prediction = SLor5_data.read_field('y_test')-SLor5_data.read_field('y_pred')
Slor10_data_pointwise_prediction = SLor10_data.read_field('y_test')-SLor10_data.read_field('y_pred')
Spend0_data_pointwise_prediction = SPend0_data.read_field('y_test')-SPend0_data.read_field('y_pred')
Spend5_data_pointwise_prediction = SPend5_data.read_field('y_test')-SPend5_data.read_field('y_pred')
Fduffc0_data_pointwise_prediction = Fduffc0_data.read_field('y_test')-Fduffc0_data.read_field('y_pred')
Fduffc5_data_pointwise_prediction = Fduffc5_data.read_field('y_test')-Fduffc5_data.read_field('y_pred')
Flor5_data_pointwise_prediction = FLor5_data.read_field('y_test')-FLor5_data.read_field('y_pred')
Flor10_data_pointwise_prediction = FLor10_data.read_field('y_test')-FLor10_data.read_field('y_pred')
Fpend0_data_pointwise_prediction = FPend0_data.read_field('y_test')-FPend0_data.read_field('y_pred')
Fpend5_data_pointwise_prediction = FPend5_data.read_field('y_test')-FPend5_data.read_field('y_pred')

duffc0_truth = Lduffc0_data.read_field('y_test')
duffc5_truth = Lduffc5_data.read_field('y_test')
lor5_truth = LLor5_data.read_field('y_test')
lor10_truth = LLor10_data.read_field('y_test')
pend0_truth = LPend0_data.read_field('y_test')
pend5_truth = LPend5_data.read_field('y_test')
grid = Lduffc0_data.read_field('x_test')


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# -----------------------------------------------
# Configuration
# -----------------------------------------------
models = ['Truth', 'SNO', 'FNO', 'LNO']
models_data = ['', 'S', 'F', 'L']
tasks = ['Duffing', 'Lorenz', 'Pendulum']
coefficients = {
    'Duffing': ['c=0', 'c=0.5'],
    'Lorenz': [r'ρ=5', r'ρ=10'],
    'Pendulum': ['c=0', 'c=0.5']
}
tasks_data = {
    'Duffing': ['duffc0', 'duffc5'],
    'Lorenz': ['lor5', 'lor10'],
    'Pendulum': ['pend0', 'pend5']
}
datas = ['_truth', '_data_pointwise_prediction', '_data_pointwise_prediction', '_data_pointwise_prediction']



# -----------------------------------------------
# Plotting setup
# -----------------------------------------------
fig = plt.figure(figsize=(15, 15), constrained_layout=False)
fig.suptitle("ODE Pointwise Prediction Errors", fontsize=24, y=0.995)

# There are 3 tasks × 2 coefficients each → 6 major rows
# Each coefficient has 1 row of plots (no stacked magnitude/error distinction)
gs = gridspec.GridSpec(
    nrows=6, ncols=4, figure=fig,
    hspace=0.25, wspace=0.25,
    left=0.08, right=0.97, top=0.95, bottom=0.05
)

# -----------------------------------------------
# Plot each panel
# -----------------------------------------------
for task_idx, task in enumerate(tasks):
    coeffs = coefficients[task]
    task_data_list = tasks_data[task]

    for coeff_idx, (coeff_label, task_data) in enumerate(zip(coeffs, task_data_list)):
        row = 2 * task_idx + coeff_idx
        for col, (model, model_data, data_suffix) in enumerate(zip(models, models_data, datas)):
            ax = fig.add_subplot(gs[row, col])

            # Retrieve data using the globals() logic from your original code
            data_name = globals()[f"{model_data}{task_data}{data_suffix}"][sample,:]

            # Color by column group (each task/column could share color)
            color_map = ['tab:red', 'tab:blue', 'tab:green', 'tab:purple']
            ax.plot(grid.squeeze(), data_name, linewidth=2.5, color=color_map[col])

            # Titles on top row only
            if row == 0:
                ax.set_title(model, fontsize=13, pad=4)

            # Format axes
            ax.tick_params(axis='both', which='both', length=0, labelsize=8)
            ax.set_xlim([grid.min(), grid.max()])
            if col == 0:
                ax.set_ylabel("Magnitude (m)", fontsize=10)
            if col == 1:
                ax.set_ylabel("Error (m)", fontsize=10)

            if row != 5:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("Time (s)", fontsize=10)

        # Add coefficient label (left)
        y_center = 1 - (row + 0.5) / 6
        fig.text(0.03, y_center, coeff_label, va='center', ha='center', rotation=90, fontsize=11)

    # Add task label spanning both coefficient rows
    y_center = 1 - (2 * task_idx + 1) / 6
    fig.text(0.015, y_center, task, va='center', ha='center', rotation=90, fontsize=13, fontweight='bold')



fig.savefig(r"C:\Users\benze\Downloads\line_plot_gridspec.png")
