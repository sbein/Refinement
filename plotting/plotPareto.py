import csv
import numpy as np
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


# specify the axes
xaxis = 'mmdfixsigma_output_target/BM'
yaxes = ['mse_output_target', 'mse_input_output']

trainorval = 'train'
iszoom = False

# specify the samples with
# (title, training_id, {constraint: epsilon}, linestyle, alpha)
samples = [
    ('No MDMM, only MSE loss', '20231107_1', {}, '-.', 1.),
    ('No MDMM, only MMD loss', '20231107', {}, '-.', 1.),
    (r'MDMM, MSE + $\lambda$(0.0017 - MMD)', '20231107_2', {'mmdfixsigma_output_target': 0.0017}, '-', 0.7),
]

# specify the input and output
in_path = '/data/dust/user/wolfmor/Refinement/TrainingOutput/traininglog_refinement_regression_TRAININGID_' + trainorval + '.csv'

nameout = 'pareto_' + samples[0][1] + '_' + trainorval + '.png'
if iszoom:
    nameout = nameout.replace('.png', '_zoom.png')

axisranges = {
    'mmdfixsigma_output_target/BM': (0.2, 0.5) if iszoom else (0., 1.5),
    'mse_output_target': (0.02, 0.1) if iszoom else (0.02, 0.15),
    'mse_input_output': (0., 0.1) if iszoom else (0., 0.1),
    'huber_output_target': (0.008, 0.02) if iszoom else (0.008, 0.02),
    'huber_input_output': (0., 0.015) if iszoom else (0., 0.015),
}

labels = {
    'mmdfixsigma_output_target/BM': r'$MMD^{fix}(Ref.,Full) \, / \, MMD^{fix}(Fast,Full)$',
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
}

print('### read csv files')
data = {}
benchmarks = {}
starts = {}
for title, trainingID, constraints, _, _ in samples:
    
    print(trainingID)
    
    nepochs, nbatches, values, benchmarks[trainingID], starts[trainingID] = read_csv(in_path.replace('TRAININGID', trainingID))

    data[trainingID] = {
        key: np.array([np.mean(values[key][epoch*nbatches:(epoch+1)*nbatches]) for epoch in range(nepochs)])
        for key in values if key in [axis.replace('/BM', '') for axis in [xaxis] + yaxes]
    }

    for axis in [xaxis] + yaxes:

        if '/BM' in axis:
            # always use the first BM because it might be slightly different
            axis = axis.replace('/BM', '')
            starts[trainingID][axis] /= benchmarks[samples[0][1]][axis]
            data[trainingID][axis] /= benchmarks[samples[0][1]][axis]
            if axis in constraints.keys():
                constraints[axis] /= benchmarks[samples[0][1]][axis]

        data[trainingID][axis] = np.insert(data[trainingID][axis], 0, starts[trainingID][axis])


print('### make plots')

labelsize = 20
ticklabelsize = 15
legendsize = 20
linewidth = 2

fig, ax = plt.subplots(nrows=1, ncols=len(yaxes), figsize=(len(yaxes)*10, 10))

for iy, yaxis in enumerate(yaxes):
    
    for title, trainingID, constraints, linestyle, alpha in samples:
        
        p = ax[iy].plot(data[trainingID][xaxis.replace('/BM', '')], data[trainingID][yaxis.replace('/BM', '')],
                        label=title, linewidth=linewidth, linestyle=linestyle, marker='', alpha=alpha)

        ax[iy].plot(data[trainingID][xaxis.replace('/BM', '')][-1], data[trainingID][yaxis.replace('/BM', '')][-1],
                    color=p[0].get_color(), marker='*', markersize=15)

        if xaxis.replace('/BM', '') in constraints.keys():
            ax[iy].axvline(constraints[xaxis.replace('/BM', '')], color=p[0].get_color(), ls='dashed', lw=1)

        if yaxis.replace('/BM', '') in constraints.keys():
            ax[iy].axhline(constraints[yaxis.replace('/BM', '')], color=p[0].get_color(), ls='dashed', lw=1)

    if '/BM' in yaxis:
        ax[iy].axhline(1., color='black', ls='solid', lw=1)

    ax[iy].set_xlabel(labels[xaxis], size=labelsize)
    ax[iy].set_ylabel(labels[yaxis], size=labelsize)

    ax[iy].set_xlim(axisranges[xaxis])
    ax[iy].set_ylim(axisranges[yaxis])

    ax[iy].legend(loc='upper left', prop={'size': legendsize}, ncol=1)

    ax[iy].tick_params(axis='x', labelsize=ticklabelsize)
    ax[iy].tick_params(axis='y', labelsize=ticklabelsize)

    ax[iy].set_title(r'$\bf{CMS}\ \it{Simulation}$', loc='left', fontsize=30)

fig.tight_layout()

fig.savefig(nameout)
print('just created ' + nameout)
