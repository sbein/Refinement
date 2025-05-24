import csv
import numpy as np
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import os, sys

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

    print('starts', starts)
    return nepochs_, nbatches_, \
        {v[0]: np.array(v[1:]) for v in values}, \
        {b[0]: np.array(b[1]) for b in benchmarks}, \
        {s[0]: np.array(s[1]) for s in starts if len(s)>1}



try: training_id = sys.argv[1]
except: 
    training_id = '7April2025'
    training_id = '20231105_1'
    training_id = '20250407'
    training_id = '7April2025withClassifier'
    training_id = '7April2025'
    
    
in_path = '/data/dust/user/beinsam/FastSim/Refinement/Regress/TrainingOutput/traininglog_refinement_regression_TRAININGID'
if not os.path.exists('figs'+training_id):
    os.system('mkdir figs'+training_id)
    
# with training ID:
#out_path = 'LCs_' + in_path.split('/')[-1].split('traininglog_')[-1].replace('_regression','').replace('TRAININGID',training_id)
# without training ID 
out_path = 'figs'+training_id+'/Rec' + in_path.split('traininglog_refine')[-1].split('_regression')[0]+'0'
print('out_path',out_path)
# specify what losses should be plotted (need to be present in the csv output file)
plots = [
    ['mmdfixsigma_output_target'],
    ['mmd_output_target'],
    #['mmd_output_target_hadflav0', 'mmd_output_target_hadflav4', 'mmd_output_target_hadflav5'],
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
labelsize = 34  # For axis labels
legendsize = 30  # For legend text
ticklabelsize = 25  # For tick labels on both axes
markersize = 20  # Assuming you might use markers
plotwidth = 10  # Adjusted for bigger text
plotheight = 8  # Adjusted for bigger text

#labelsize = 20
#legendsize = 15
#ticklabelsize = 15
#markersize = 20
#plotwidth = 10
#plotheight = 5

print('\n### read csv files')
print('hopefully this exists', in_path.replace('TRAININGID', training_id) + '_train.csv')
nepochs_train, nbatches_train, values_train, benchmarks_train, starts_train = read_csv(in_path.replace('TRAININGID', training_id) + '_train.csv')

print('hopefully this exists too', in_path.replace('TRAININGID', training_id) + '_validation.csv')
nepochs_val, nbatches_val, values_val, benchmarks_val, starts_val = read_csv(in_path.replace('TRAININGID', training_id) + '_validation.csv')
print('done with that')
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
    a.grid(which='both', linestyle='--', linewidth=0.5, color='black')


print('\n### make plots')

fig, axes = plt.subplots(nrows=len(plots), ncols=2, figsize=(2 * plotwidth, len(plots) * plotheight))

if len(plots) == 1:
    axes = [axes]


for iplot, plot in enumerate(plots):
    fig, ax = plt.subplots(figsize=(plotwidth, plotheight))  # Create figure for each plot

    mdmm_constraint_plotted = False
    
    max_final_epoch_train_value = 0
    
    min_y_axis_value = 9999
    max_y_axis_value = 0
    for p in plot:

        if p in labels: 
            label = labels[p]
        else: 
            label = p

        c = ax.plot(epochs, data_train[p], label=label if len(plot) > 1 else '', linewidth=linewidth)
        ax.plot(epochs, data_val[p], label='', linewidth=linewidth, color=c[0].get_color(), linestyle='--')
        if len(plot) == 1: 
            ax.set_ylabel(label, size=labelsize)

        if p in mdmm_constraints:
            ax.axhline(mdmm_constraints[p], label='MDMM Constraint' if not mdmm_constraint_plotted else '', linewidth=linewidth, color='black', linestyle=':')
            mdmm_constraint_plotted = True
        print('we should be here', plot, min_y_axis_value)
        
        min_y_axis_value = min(min_y_axis_value, data_train[p][-3] * 0.95)
        max_y_axis_value = max(min_y_axis_value, data_train[p][-3] * 1.2)        
        print('and after', plot, data_train[p], min_y_axis_value)

    styleAx(ax)
    ax.set_xlim(1,nepochs_train)
    ax.set_xscale('log',base=10)
    
    ax.set_yscale('linear')  # Explicitly set to linear for clarity, though it's the default
    ax.set_ylim(bottom=min_y_axis_value)
    ax.set_ylim(top=max_y_axis_value)
    if 'mmd' in plot[0].lower():#can set to false if you've changed the loss function or inputs
        print('taking action')
        ax.set_ylim(bottom=0.00063)
        ax.set_ylim(top=0.00076)    

    # Adjust layout and save the plot
    plt.tight_layout()

    # Construct a filename for the plot
    plot_name = '_'.join(plot)  # Concatenates all metric names, consider simplifying
    print('plot_name', plot_name)
    individual_out_path = out_path+'_'+plot_name+'.png'
    fig.savefig(individual_out_path)
    print(f'just created {individual_out_path}')

    plt.close(fig)  # Close the figure to free up memory


