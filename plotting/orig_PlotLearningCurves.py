import csv
import numpy as np
import matplotlib.lines as mlines
import matplotlib.pyplot as plt


def read_csv(filepath):

    with open(filepath) as f:
        rows = 0
        reader = csv.reader(f)
        for irow, row in enumerate(reader):
            if irow == 0:
                values = [[key] for key in row]
                benchmarks = [[key] for key in row[2:]]
                starts = [[key] for key in row[2:]]
            elif irow == 1:
                for icol, value in enumerate(row[2:]):
                    benchmarks[icol].append(float(value))
            elif irow == 2:
                for icol, value in enumerate(row[2:]):
                    starts[icol].append(float(value))
            else:
                rows += 1
                for icol, value in enumerate(row):
                    values[icol].append(float(value))

        nepochs_ = int(row[0]) + 1
        nbatches_ = int(rows / nepochs_)

    return nepochs_, nbatches_, \
        {v[0]: np.array(v[1:]) for v in values}, \
        {b[0]: np.array(b[1]) for b in benchmarks}, \
        {s[0]: np.array(s[1]) for s in starts}


# specify the training_id and where the csv output file is (TRAININGID will be replaced by the training_id)
#in_path = '/data/dust/user/beinsam/FastSim/Refinement/Regress/traininglog_refinement_regression_TRAININGID'
in_path = '/data/dust/user/beinsam/FastSim/Refinement/Regress/TrainingOutput/traininglog_refineJet_regression_TRAININGID'
training_id = '20240306'

# specify where the plot should be saved
out_path = 'learning_curves_' + training_id + '.png'

# specify what losses should be plotted (need to be present in the csv output file)
plots = [
    ['mmdfixsigma_output_target'],
    ['mmd_output_target'],
    ['mmd_output_target_hadflav0', 'mmd_output_target_hadflav4', 'mmd_output_target_hadflav5'],
    ['mse_output_target'],
    ['mse_input_output'],
    # ['lmbda_mmdfixsigma_output_target'],
]

labels = {
    'mmdfixsigma_output_target': r'$MMD^{fix}(Ref.,Full)$',
    'mmd_output_target': r'$MMD(Ref.,Full)$',
    'mmd_output_target_hadflavSum': r'$MMD_{hadFlavSum}(Ref.,Full)$',
    'mmd_output_target_hadflav0': r'$MMD_{hadFlav0}(Ref.,Full)$',
    'mmd_output_target_hadflav4': r'$MMD_{hadFlav4}(Ref.,Full)$',
    'mmd_output_target_hadflav5': r'$MMD_{hadFlav5}(Ref.,Full)$',
    'mse_output_target': r'$MSE(Ref.,Full)$',
    'mse_input_output': r'$MSE(Fast,Ref.)$',
    'huber_output_target': r'$Huber(Ref.,Full)$',
    'huber_input_output': r'$Huber(Fast,Ref.)$',
    'deepjetsum_mean': r'$DeepJetSum Mean$',
    'deepjetsum_std': r'$DeepJetSum Std$',
    'lmbda_deepjetsum_mean': r'$\lambda_{MDMM}(DeepJetSum Mean)$',
    'lmbda_deepjetsum_std': r'$\lambda_{MDMM}(DeepJetSum Std)$',
    'lmbda_mmd_output_target_hadflavSum': r'$\lambda_{MDMM}(MMD_{hadFlavSum}(Ref.,Full))$',
    'lmbda_mmdfixsigma_output_target': r'$\lambda_{MDMM}(MMD^{fix}(Ref.,Full))$',
    'lmbda_mse_output_target': r'$\lambda_{MDMM}(MSE(Ref.,Full))$',
}

linewidth = 2.
labelsize = 20
legendsize = 15
ticklabelsize = 15
markersize = 20
plotwidth = 10
plotheight = 5

print('\n### read csv files')
print('hopefully this exists', in_path.replace('TRAININGID', training_id) + '_train.csv')

nepochs_train, nbatches_train, values_train, benchmarks_train, starts_train = read_csv(in_path.replace('TRAININGID', training_id) + '_train.csv')
nepochs_val, nbatches_val, values_val, benchmarks_val, starts_val = read_csv(in_path.replace('TRAININGID', training_id) + '_validation.csv')

assert nepochs_train == nepochs_val
nepochs = nepochs_train
epochs = np.array(range(nepochs + 1))

data_train = {
    key: np.array([np.mean(values_train[key][epoch*nbatches_train:(epoch+1)*nbatches_train]) for epoch in range(nepochs)])
    for key in values_train if key in [p for plot in plots for p in plot]
}

mdmm_constraints = {key.replace('lmbda_', ''): benchmarks_train[key] for key in benchmarks_train if key.startswith('lmbda_')}

for key in data_train:
    if key in ['epoch', 'iteration']: continue
    assert len(data_train[key]) == nepochs
    data_train[key] = np.insert(data_train[key], 0, starts_train[key])

data_val = {
    key: np.array([np.mean(values_val[key][epoch*nbatches_val:(epoch+1)*nbatches_val]) for epoch in range(nepochs)])
    for key in values_val if key in [p for plot in plots for p in plot]
}

for key in data_val:
    assert len(data_val[key]) == nepochs
    data_val[key] = np.insert(data_val[key], 0, starts_val[key])


def styleAx(a):

    handles, _ = a.get_legend_handles_labels()

    handles.insert(int(np.ceil(len(handles) / 2)), mlines.Line2D([], [], color='black', label='Validation', linewidth=linewidth, linestyle='--'))
    handles.insert(0, mlines.Line2D([], [], color='black', label='Train', linewidth=linewidth))

    a.legend(loc='upper right', prop={'size': legendsize}, ncol=2, handles=handles)

    a.tick_params(axis='x', labelsize=ticklabelsize)
    a.tick_params(axis='y', labelsize=ticklabelsize)

    a.set_xlabel('Epoch', size=labelsize)
    a.set_xlim([0, nepochs])
    a.xaxis.get_major_locator().set_params(integer=True)
    a.grid(which='both', linestyle='--', linewidth=0.5, color='black', alpha=0.5)


print('\n### make plots')

fig, axes = plt.subplots(nrows=len(plots), ncols=2, figsize=(2 * plotwidth, len(plots) * plotheight))

if len(plots) == 1:
    axes = [axes]

for iplot, plot in enumerate(plots):

    mdmm_constraint_plotted = False

    for p in plot:

        if p in labels: label = labels[p]
        else: label = p

        c = axes[iplot][0].plot(epochs, data_train[p], label=label if len(plot) > 1 else '', linewidth=linewidth)
        axes[iplot][0].plot(epochs, data_val[p], label='', linewidth=linewidth, color=c[0].get_color(), linestyle='--')
        if len(plot) == 1: axes[iplot][0].set_ylabel(label, size=labelsize)

        axes[iplot][1].plot(epochs, abs(data_train[p]), label='|' + label + '|' if len(plot) > 1 else '', linewidth=linewidth, color=c[0].get_color())
        axes[iplot][1].plot(epochs, abs(data_val[p]), label='', linewidth=linewidth, color=c[0].get_color(), linestyle='--')
        if len(plot) == 1: axes[iplot][1].set_ylabel('|' + label + '|', size=labelsize)

        if p in mdmm_constraints:
            for idx in [0, 1]:
                axes[iplot][idx].axhline(mdmm_constraints[p], label='' if mdmm_constraint_plotted else 'MDMM Constraint', linewidth=linewidth, color='black', linestyle=':')
            mdmm_constraint_plotted = True

    for idx in [0, 1]:
        styleAx(axes[iplot][idx])
    axes[iplot][1].set_yscale('log')

fig.tight_layout()
fig.savefig(out_path)
print('just created ' + out_path)
