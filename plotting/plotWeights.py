import torch
import numpy as np
import matplotlib.pyplot as plt


training_id = '20240529_1'

model_path = '/data/dust/user/wolfmor/Refinement/TrainingOutput/model_refinement_regression_' + training_id + '.pt'
out_path = '/afs/desy.de/user/w/wolfmor/Plots/Refinement/Training/Regression/weights_' + training_id + '.png'

model = torch.jit.load(model_path)
model.eval()

weights = []

i = 0
while hasattr(model, 'LinearWithSkipConnection_' + str(i)):
    weights.append([])
    skipblock = getattr(model, 'LinearWithSkipConnection_' + str(i))
    for modulelist in skipblock.children():
        for child in modulelist.children():
            if child.original_name == 'Linear':
                weights[i].append(torch.flatten(child.weight).detach().cpu().numpy())  # .bias
    i += 1
weights.append([])
skipblock = getattr(model, 'LinearWithSkipConnection_last')
for modulelist in skipblock.children():
    for child in modulelist.children():
        if child.original_name == 'Linear':
            weights[i].append(torch.flatten(child.weight).detach().cpu().numpy())  # .bias


plotwidth = 10
plotheight = 5
labelsize = 20
ticklabelsize = 15

nrows = len(weights)
ncols = max([len(row) for row in weights])

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * plotwidth, nrows * plotheight))

if nrows == 1 and ncols == 1:
    axes = [[axes]]
elif nrows == 1:
    axes = [axes]
elif ncols == 1:
    axes = [[a] for a in axes]

for i in range(nrows):
    for j in range(len(weights[i])):

        rangeedge = max(abs(weights[i][j].min()), abs(weights[i][j].max()))

        axes[i][j].hist(weights[i][j], bins=51, range=(-rangeedge, rangeedge))

        axes[i][j].set_title('Skip Block ' + str(i+1) + ', Layer ' + str(j+1)
                             + ': Mean = ' + str(np.round(np.mean(weights[i][j]), 5))
                             + ', Std = ' + str(np.round(np.std(weights[i][j]), 5)), size=labelsize)

        axes[i][j].set_xlabel('Value', size=labelsize)
        axes[i][j].set_ylabel('Weight Matrix Entries', size=labelsize)

        axes[i][j].tick_params(axis='x', labelsize=ticklabelsize)
        axes[i][j].tick_params(axis='y', labelsize=ticklabelsize)

        axes[i][j].axvline(0., color='black')

fig.tight_layout()
fig.savefig(out_path)

print('just created ' + out_path)
