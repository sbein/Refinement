"""
to load packages, e.g.:
source /cvmfs/sft.cern.ch/lcg/views/LCG_106_cuda/x86_64-el9-gcc11-opt/setup.sh

to write terminal output to a txt file:
python trainRegression.py 2>&1 | tee traininglog_regression.txt
"""

import os
import csv
from datetime import datetime

import ROOT
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import colors

import torch
torch.set_printoptions(edgeitems=5, linewidth=160)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import my_mmd
import my_mdmm
from my_modules import *


def snapshot(_inp, _target, _out, _epoch, is_transformed=False, plot_kde=True):

    print('snapshot!')

    palette = {'Refined': 'tab:blue', 'Fast': 'tab:red', 'Full': 'tab:green'}

    # iterators will be used for Fast/Full in parallel and then for Refined _or_ only for Refined

    columns = [n[0].replace('_CLASS', '') for n in real_parameters + VARIABLES]
    if _epoch == 0: columns_iter = iter([c for c in columns for _ in range(2)] + columns)
    else: columns_iter = iter(columns)

    off_diag = []
    for iy in range(1, len(columns)):
        for ix in range(0, iy):
            off_diag.append((columns[ix], columns[iy]))
    if _epoch == 0: offdiag_iter = iter([od for od in off_diag for _ in range(2)] + off_diag)
    else: offdiag_iter = iter(off_diag)

    if is_transformed:
        transformed_bins = np.linspace(-15, 15, 7)
        bins = {
            'GenJet_pt': transformed_bins,
            'GenJet_eta': np.linspace(-6, 6, 7),
            'GenJet_hadronFlavour': np.linspace(-5, 11, 17),
            'RecJet_hadronFlavour_FastSim': np.linspace(-5, 11, 17),
            'RecJet_btagDeepFlavB': transformed_bins,
            'RecJet_btagDeepFlavCvB': transformed_bins,
            'RecJet_btagDeepFlavCvL': transformed_bins,
            'RecJet_btagDeepFlavQG': transformed_bins,
        }
    else:
        deepflavbins = np.linspace(-0.5, 1.5, 9)
        bins = {
            'GenJet_pt': np.linspace(-500, 1500, 9),
            'GenJet_eta': np.linspace(-6, 6, 7),
            'GenJet_hadronFlavour': np.linspace(-5, 11, 17),
            'RecJet_hadronFlavour_FastSim': np.linspace(-5, 11, 17),
            'RecJet_btagDeepFlavB': deepflavbins,
            'RecJet_btagDeepFlavCvB': deepflavbins,
            'RecJet_btagDeepFlavCvL': deepflavbins,
            'RecJet_btagDeepFlavQG': deepflavbins,
        }

    def myhist(x, **kwargs):
        b = bins[next(columns_iter)]
        sns.histplot(x, bins=b, **kwargs)

    def mykde(x, y, **kwargs):
        thex, they = next(offdiag_iter)
        c = ((bins[thex][0], bins[thex][-1]), (bins[they][0], bins[they][-1]))
        theax = sns.kdeplot(x=x, y=y, clip=c, **kwargs)
        theax.set_xlim((bins[thex][0], bins[thex][-1]))
        theax.set_ylim((bins[they][0], bins[they][-1]))

    # ----------------------------------------------
    # to plot all three classes every time:

    # # add parameters to Full and Refined lists
    # _target = torch.cat((_inp[:, :realLenParameters], _target), axis=1)
    # _out = torch.cat((_inp[:, :realLenParameters], _out), axis=1)
    #
    # # the order defines what is plotted on top
    # data = torch.cat((_inp, _target, _out)).cpu().numpy()
    # categories = ['Fast'] * _inp.shape[0] + ['Full'] * _target.shape[0] + ['Refined'] * _out.shape[0]
    #
    # df = pd.DataFrame({n[0].replace('_CLASS', ''): data[:, idim] for idim, n in enumerate(realPARAMETERS + VARIABLES)})
    # df['Category'] = categories
    #
    # g = sns.PairGrid(df, vars=columns, hue='Category', palette={'Refined': 'tab:blue', 'Fast': 'tab:red', 'Full': 'tab:green'}, diag_sharey=False)
    #
    # g.map_diag(myhist, fill=False, element='bars')
    # g.map_upper(sns.scatterplot)
    # if plot_kde: g.map_lower(mykde, levels=[1-0.997, 1-0.955, 1-0.683])
    #
    # g.add_legend(title='Epoch ' + str(_epoch))


    # ----------------------------------------------
    # to plot Fast/Full only for first plot:

    if _epoch == 0:

        globals()['snapshot_parameters' + str(is_transformed)] = _inp[:, :real_len_parameters]

        # add parameters to Full list
        _target = torch.cat((globals()['snapshot_parameters' + str(is_transformed)], _target), axis=1)

        dataXy = torch.cat((_inp, _target)).cpu().numpy()
        categoriesXy = ['Fast'] * _inp.shape[0] + ['Full'] * _target.shape[0]
        dfXy = pd.DataFrame({n[0].replace('_CLASS', ''): dataXy[:, idim] for idim, n in enumerate(real_parameters + VARIABLES)})
        dfXy['Category'] = categoriesXy

        globals()['snapshot_g' + str(is_transformed)] = sns.PairGrid(dfXy, vars=columns, hue='Category', palette=palette, diag_sharey=False)

        globals()['snapshot_g' + str(is_transformed)].map_diag(myhist, fill=False, element='bars')
        globals()['snapshot_g' + str(is_transformed)].map_upper(sns.scatterplot)
        if plot_kde: globals()['snapshot_g' + str(is_transformed)].map_lower(mykde, levels=[1-0.997, 1-0.955, 1-0.683])


    # ... and then add Refined:

    # add parameters to Refined list
    _out = torch.cat((globals()['snapshot_parameters' + str(is_transformed)], _out), axis=1)

    dataOutput = _out.cpu().numpy()
    categoriesOutput = ['Refined'] * _out.shape[0]
    dfOutput = pd.DataFrame({n[0].replace('_CLASS', ''): dataOutput[:, idim] for idim, n in enumerate(real_parameters + VARIABLES)})
    dfOutput['Category'] = categoriesOutput


    # update PairGrid object
    g = globals()['snapshot_g' + str(is_transformed)]

    # remove Refined plots
    if _epoch > 0:
        for ax in g.figure.axes:

            # to remove histogram
            for patch in ax.patches:
                if patch.get_edgecolor() == colors.to_rgba(palette['Refined']):
                    patch.set_visible(False)

            # to remove contours and scatter plots
            for collection in ax.collections:
                facecolors = collection.get_facecolor()
                edgecolors = collection.get_edgecolor()
                for color in [facecolors, edgecolors]:
                    if len(color) > 0:
                        if tuple(color[0]) == colors.to_rgba(palette['Refined']):
                            collection.set_visible(False)
                    else:
                        if tuple(color) == colors.to_rgba(palette['Refined']):
                            collection.set_visible(False)

    g.data = dfOutput
    g.hue_names = ['Refined']
    g._hue_order = ['Refined']
    g.hue_vals = dfOutput['Category']
    g.palette = g._get_palette(dfOutput, 'Category', ['Refined'], palette)

    g.map_diag(myhist, fill=False, element='bars')
    g.map_upper(sns.scatterplot)
    if plot_kde: g.map_lower(mykde, levels=[1-0.997, 1-0.955, 1-0.683])


    g.figure.set_size_inches(15, 15)
    if g.legend:
        g.legend.remove()
    g.add_legend(title='Epoch ' + str(_epoch))


    # ----------------------------------------------

    if not os.path.exists(snapshot_out_path):
        os.makedirs(snapshot_out_path)

    outname = training_id
    if is_transformed: outname += '_transformed'
    outname += '_epoch_' + str(_epoch)

    g.savefig(snapshot_out_path + '/' + outname + '.png')


'''
# ##### ##### #####
# general settings
# ##### ##### #####
'''

training_id = datetime.today().strftime('%Y%m%d') + ''  # will be used as an identifier in the output filenames, adapt if needed

is_test = True
is_verbose = False

save_snapshots = False  # save snapshot plots to monitor training progress
snapshot_out_path = 'Snapshots/' + training_id  # where to save the snapshots
snapshot_num_batches = 4  # how many batches to use for each snapshot
snapshot_everyXepoch = 9  # how many snapshots to save
snapshot_plot_kde = True
# snapshots with kde take quite some time due to kde analysis for the correlation plots...
# roughly ~60s for batches with 1024 jets Fast/Full/Refined (~25s for only Refined)
# scales quite linearly with number of jets
# e.g. for 10 batches w/ 1024 jets and 10 epochs/snapshots: 1 * 10*60s + 9 * 10*25 = 2,850s = 47.5min = 0.8h
# e.g. for 10 batches w/ 1024 jets and 100 epochs/snapshots: 1 * 10*60s + 99 * 10*25 = 25,350s = 422.5min = 7h
# e.g. for 100 batches w/ 1024 jets and 10 epochs/snapshots: 1 * 100*60s + 9 * 100*25 = 28,500s = 475min = 8h


'''
# ##### ##### ##### #####
# define input/output
# ##### ##### ##### #####
'''

# the input file is expected to contain a tree filled with jet triplets: RecJet_x_FastSim, RecJet_x_FullSim, GenJet_y,...
in_path = '/nfs/dust/cms/user/wolfmor/Refinement/littletree_CMSSW_14_0_12_T1ttttRun3PU_step2_SIM_RECOBEFMIX_DIGI_L1_DIGI2RAW_L1Reco_RECO_PAT_NANO_PU_coffea_PUPPI.root'
in_tree = 'tJet'
preselection = 'GenJet_nearest_dR>0.5&&RecJet_nearest_dR_FastSim>0.5&&RecJet_nearest_dR_FullSim>0.5' \
               '&&RecJet_btagUParTAK4B_FastSim>0&&RecJet_btagUParTAK4B_FullSim>0'  # make sure the ParT taggers are defined

out_path = '/nfs/dust/cms/user/wolfmor/Refinement/TrainingOutput/output_refinement_regression_' + training_id + '.root'


'''
# ##### ##### #####
# model settings
# ##### ##### #####
'''

onnxcompatible = True

num_skipblocks = 5
num_layers_per_skipblock = 2
nodes_hidden_layer = 1024

dropout = 0.2

castto16bit = True  # cast the output as if it is 16-bit float

adddeepjetconstraintlayer = False

# for min. and max. values in the logit-transformation layers
tiny = 1e-8
epsilon = torch.finfo(torch.float16).eps / 2.  # = 1 / 2048 = 1 - 0.99951172 (1 - largest number < 1 in 16-bit)
logitfactor = 1.


'''
# ##### ##### #####
# training settings
# ##### ##### #####
'''

if is_test: num_epochs = 2
else: num_epochs = 100

learning_rate = 1e-5

lr_scheduler_gamma = False  # 1.
cyclic_lr = False  # (base_lr, max_lr, step_size_up)

if is_test: batch_size = 2048
else: batch_size = 2048

if is_test: num_batches = [2, 2, 2]
else: num_batches = [500, 100, 200]

'''
# ##### ##### ##### ##### ##### ##### #####
# define variables and transformations
# ##### ##### ##### ##### ##### ##### #####
'''

onehotencode = ('RecJet_hadronFlavour_FastSim', [0, 4, 5])  # add one-hot-encoding for given input variable with the given values
# onehotencode = False

PARAMETERS = [
    ('GenJet_pt', ['tanh200', 'logit']),
    ('GenJet_eta', []),
    ('RecJet_hadronFlavour_FastSim', [])
]

# if using DeepJetConstraint the DeepJet transformations have to be explicitly adapted in the DeepJetConstraint module
VARIABLES = [
    ('RecJet_btagDeepFlavB_CLASS', ['logit']),
    ('RecJet_btagDeepFlavCvB_CLASS', ['logit']),
    ('RecJet_btagDeepFlavCvL_CLASS', ['logit']),
    ('RecJet_btagDeepFlavQG_CLASS', ['logit']),

    ('RecJet_btagUParTAK4B_CLASS', ['logit']),
    ('RecJet_btagUParTAK4CvB_CLASS', ['logit']),
    ('RecJet_btagUParTAK4CvL_CLASS', ['logit']),
    ('RecJet_btagUParTAK4QvG_CLASS', ['logit']),
]

spectators_raw = [
    'GenJet_pt',
    'GenJet_eta',
    'GenJet_phi',
    'GenJet_mass',
    'GenJet_hadronFlavour',
    'GenJet_partonFlavour',

    'GenJet_nearest_dR',
    'GenJet_nearest_pt',

    'RecJet_pt_CLASS',
    'RecJet_eta_CLASS',
    'RecJet_phi_CLASS',
    'RecJet_mass_CLASS',
    'RecJet_mass_CLASS_log10:=log10(RecJet_mass_CLASS)',
    'RecJet_hadronFlavour_CLASS',
    'RecJet_partonFlavour_CLASS',
    'RecJet_jetId_CLASS',

    'RecJet_response_CLASS',

    'RecJet_event_nJet_CLASS',
    'RecJet_event_PV_npvsGood_CLASS',
    'RecJet_event_MET_pt_CLASS',

    'RecJet_nearest_dR_CLASS',
    'RecJet_nearest_pt_CLASS',
]
excludespectators = [var[0] for var in PARAMETERS + VARIABLES]
SPECTATORS = [
    s for s in spectators_raw if s not in excludespectators
    and s.replace('CLASS', 'FastSim') not in excludespectators
    and s.replace('CLASS', 'FullSim') not in excludespectators
]
SPECTATORS = [
    n for n in SPECTATORS if 'CLASS' not in n] + [
    n.replace('CLASS', 'FastSim') for n in SPECTATORS if 'CLASS' in n] + [
    n.replace('CLASS', 'FullSim') for n in SPECTATORS if 'CLASS' in n
]

# to use in omniscient loss, have to be part of SPECTATORS
hiddenvariables = [
    'RecJet_eta_CLASS',
    'RecJet_mass_CLASS_log10:=log10(RecJet_mass_CLASS)',
]

hiddenvariables_indices_fast = [i for i, n in enumerate(SPECTATORS) if n in hiddenvariables + [h.replace('CLASS', 'FastSim') for h in hiddenvariables]]
hiddenvariables_indices_full = [i for i, n in enumerate(SPECTATORS) if n in hiddenvariables + [h.replace('CLASS', 'FullSim') for h in hiddenvariables]]

'''
# ##### ##### #####
# get training data
# ##### ##### #####
'''
print('\n### get training data')

ntrain = num_batches[0] * batch_size
nval = num_batches[1] * batch_size
ntest = num_batches[2] * batch_size

if ntest == 0:
    rdf = ROOT.RDataFrame(in_tree, in_path).Filter(preselection)
else:
    rdf = ROOT.RDataFrame(in_tree, in_path).Filter(preselection).Range(ntrain + nval + ntest)

for n, _ in PARAMETERS + VARIABLES + [(s, None) for s in SPECTATORS]:
    if ':=' in n:
        if 'CLASS' in n:
            for cls in ['FastSim', 'FullSim']:
                rdf = rdf.Define(n.split(':=')[0].replace('CLASS', cls), n.split(':=')[1].replace('CLASS', cls))
        else:
            rdf = rdf.Define(n.split(':=')[0], n.split(':=')[1])

dict_input = rdf.AsNumpy([n[0].split(':=')[0].replace('CLASS', 'FastSim') for n in PARAMETERS + VARIABLES if '_ohe_' not in n[0]])
dict_target = rdf.AsNumpy([n[0].split(':=')[0].replace('CLASS', 'FullSim') for n in VARIABLES])
dict_spectators = rdf.AsNumpy([s.split(':=')[0] for s in SPECTATORS])

my_dtype = torch.float32
data_input = torch.tensor(np.stack([dict_input[var] for var in dict_input], axis=1), dtype=my_dtype, device=device)
data_target = torch.tensor(np.stack([dict_target[var] for var in dict_target], axis=1), dtype=my_dtype, device=device)
data_spec = torch.tensor(np.stack([dict_spectators[var] for var in dict_spectators], axis=1), dtype=my_dtype, device=device)

dataset = torch.utils.data.TensorDataset(data_input, data_target, data_spec)

if ntest == 0:
    ntest = len(dataset) - (ntrain + nval)

if rdf.Count().GetValue() < ntrain + nval + ntest:
    raise Exception('input dataset too small, choose smaller/fewer batches')

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [ntrain, nval, ntest],
    generator=torch.Generator().manual_seed(42)
)


def collate_fn(batch_):
    batch_ = list(filter(lambda x: torch.all(torch.isfinite(torch.cat(x))), batch_))
    return torch.utils.data.dataloader.default_collate(batch_)


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=not is_test, collate_fn=collate_fn)
validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

len_train_loader = len(train_loader)
len_validation_loader = len(validation_loader)
len_test_loader = len(test_loader)


'''
# ##### ##### ##### ##### ##### ##### #####
# process variables and transformations
# ##### ##### ##### ##### ##### ##### #####
'''

if 'RecJet_hadronFlavour_FastSim' in [name[0] for name in PARAMETERS]:

    hadronFlavourIndex = [name[0] for name in PARAMETERS].index('RecJet_hadronFlavour_FastSim')

    len_data_input = len(data_input)
    hadflav_fraction_0 = len(data_input[data_input[:, hadronFlavourIndex] == 0]) / len_data_input
    hadflav_fraction_4 = len(data_input[data_input[:, hadronFlavourIndex] == 4]) / len_data_input
    hadflav_fraction_5 = len(data_input[data_input[:, hadronFlavourIndex] == 5]) / len_data_input

else:

    hadronFlavourIndex = None

    hadflav_fraction_0 = None
    hadflav_fraction_4 = None
    hadflav_fraction_5 = None

deepjetindicesWithParameters = [idx for idx, name in enumerate(PARAMETERS + VARIABLES) if 'Jet_btagDeepFlav' in name[0]]
deepjetindicesWithoutParameters = [idx for idx, name in enumerate(VARIABLES) if 'Jet_btagDeepFlav' in name[0]]

robustpartak4indicesWithParameters = [idx for idx, name in enumerate(PARAMETERS + VARIABLES) if 'Jet_btagUParTAK4' in name[0]]
robustpartak4indicesWithoutParameters = [idx for idx, name in enumerate(VARIABLES) if 'Jet_btagUParTAK4' in name[0]]

logitmaskWithParameters = [int('logit' in name[1]) for name in PARAMETERS + VARIABLES]
logitmaskWithoutParameters = [int('logit' in name[1]) for name in VARIABLES]

tanh200maskWithParameters = [int('tanh200' in name[1]) for name in PARAMETERS + VARIABLES]
tanh200maskWithoutParameters = [int('tanh200' in name[1]) for name in VARIABLES]

# without artificial increase due to one-hot-encoding
real_parameters = PARAMETERS.copy()
real_len_parameters = len(PARAMETERS)

if onehotencode:
    if onehotencode[0] in [p[0] for p in PARAMETERS]:
        for ip, p in enumerate(PARAMETERS):
            if p[0] == onehotencode[0]:
                for ival, val in enumerate(onehotencode[1]):
                    if ival == 0:
                        PARAMETERS.append((onehotencode[0], PARAMETERS[ip][1]))
                    else:
                        PARAMETERS.append((onehotencode[0] + '_onehotencode_' + str(ival), PARAMETERS[ip][1]))
                PARAMETERS.pop(ip)
                onehotencode += (ip,)
                break
    elif onehotencode[0] in [v[0] for v in VARIABLES]:
        raise NotImplementedError('don\'t want to interpret one-hot-encoding: ', onehotencode)
    else:
        raise NotImplementedError('can\'t interpret one-hot-encoding: ', onehotencode)

# skipindices = [idx for idx in range(len(PARAMETERS), len(PARAMETERS + VARIABLES))]  # skip only variables
skipindices = [idx for idx in range(len(PARAMETERS + VARIABLES))]  # skip parameters and variables


'''
# ##### #####
# build model
# ##### #####
'''
print('\n### build model')

nodes_hidden_layer_list = [nodes_hidden_layer for _ in range(num_skipblocks)]

model = nn.Sequential()

if any(tanh200maskWithParameters): model.add_module('Tanh200Transform', TanhTransform(mask=tanh200maskWithParameters, norm=200))
if any(logitmaskWithParameters): model.add_module('LogitTransform', LogitTransform(mask=logitmaskWithParameters, factor=logitfactor, onnxcompatible=onnxcompatible, eps=epsilon, tiny=tiny))

if onehotencode: model.add_module('OneHotEncode_' + onehotencode[0], OneHotEncode(source_idx=onehotencode[2], target_vals=onehotencode[1]))

model.add_module('LinearWithSkipConnection_0', LinearWithSkipConnection(in_features=len(VARIABLES) + len(PARAMETERS),
                                                                        out_features=nodes_hidden_layer_list[0] if num_skipblocks > 1 else len(VARIABLES),
                                                                        hidden_features=None if num_skipblocks > 1 else nodes_hidden_layer_list[0],
                                                                        n_params=len(PARAMETERS),
                                                                        n_vars=len(VARIABLES),
                                                                        skipindices=skipindices,
                                                                        nskiplayers=num_layers_per_skipblock,
                                                                        dropout=dropout,
                                                                        isfirst=True,
                                                                        islast=num_skipblocks == 1))

if num_skipblocks == 1 and adddeepjetconstraintlayer:
    model.add_module('DeepJetConstraint', DeepJetConstraint(deepjetindices=deepjetindicesWithoutParameters, logittransform=any(logitmaskWithoutParameters), logittransformfactor=logitfactor, eps=epsilon, tiny=tiny, nanotransform=True, onnxcompatible=onnxcompatible))

for k in range(num_skipblocks - 1):

    if k == num_skipblocks - 2:
        model.add_module('LinearWithSkipConnection_last', LinearWithSkipConnection(in_features=nodes_hidden_layer_list[k],
                                                                                   out_features=len(VARIABLES),
                                                                                   n_params=len(PARAMETERS),
                                                                                   n_vars=len(VARIABLES),
                                                                                   skipindices=skipindices,
                                                                                   nskiplayers=num_layers_per_skipblock,
                                                                                   dropout=dropout,
                                                                                   islast=True))
        if adddeepjetconstraintlayer:
            model.add_module('DeepJetConstraint', DeepJetConstraint(deepjetindices=deepjetindicesWithoutParameters, logittransform=any(logitmaskWithoutParameters), logittransformfactor=logitfactor, eps=epsilon, tiny=tiny, nanotransform=True, onnxcompatible=onnxcompatible))
    else:
        model.add_module('LinearWithSkipConnection_' + str(k+1), LinearWithSkipConnection(in_features=nodes_hidden_layer_list[k],
                                                                                          out_features=nodes_hidden_layer_list[k+1],
                                                                                          n_params=len(PARAMETERS),
                                                                                          n_vars=len(VARIABLES),
                                                                                          skipindices=skipindices,
                                                                                          nskiplayers=num_layers_per_skipblock,
                                                                                          dropout=dropout))

if any(logitmaskWithoutParameters): model.add_module('LogitTransformBack', LogitTransformBack(mask=logitmaskWithoutParameters, factor=logitfactor))
if any(tanh200maskWithoutParameters): model.add_module('Tanh200TransformBack', TanhTransformBack(mask=tanh200maskWithoutParameters, norm=200))

if castto16bit:
    model.add_module('CastTo16Bit', CastTo16Bit())

# alternative model
# model = ResNet(
#     in_features=len(VARIABLES)+len(PARAMETERS),
#     hidden_features=nodes_hidden_layer,
#     out_features=len(VARIABLES),
#     n_resblocks=num_skipblocks-1
# )

model.to(device)

print(model)
print(sum(p.numel() for p in model.parameters() if p.requires_grad), 'trainable parameters')


'''
# ##### ##### #####
# define losses
# ##### ##### #####
'''

calculatelosseswithtransformedvariables = True
includeparametersinmmd = True

# mmdfixsigma is now unbiased MMD
mmdfixsigma_fn = my_mmd.MMD(kernel_mul=1., kernel_num=1, exclude_diagonals=True, calculate_fix_sigma_for_each_dimension_with_target_only='median')
mmd_fn = my_mmd.MMD(kernel_mul=1., kernel_num=1, exclude_diagonals=False, calculate_fix_sigma_for_each_dimension_with_target_only='median')
mmdomniscient_fn = my_mmd.MMD(kernel_mul=1., kernel_num=1, exclude_diagonals=False, calculate_fix_sigma_for_each_dimension_with_target_only='median')
mse_fn = torch.nn.MSELoss()
mae_fn = torch.nn.L1Loss()
huber_fn = torch.nn.HuberLoss(delta=0.1)


# these will be evaluated/monitored during training and can be used to update the model (see below)
loss_fns = {

    'mmdomniscient_output_target':
        lambda inp_, out_, target_: mmdomniscient_fn(out_, target_,
                                                     parameters=(OneHotEncode(source_idx=onehotencode[2], target_vals=onehotencode[1]).forward(inp_)[:, :len(PARAMETERS)]
                                                                 if onehotencode else inp_[:, :len(PARAMETERS)])
                                                     if includeparametersinmmd else None),

    'mmdfixsigma_output_target':
        lambda inp_, out_, target_: mmdfixsigma_fn(out_, target_,
                                                   parameters=(OneHotEncode(source_idx=onehotencode[2], target_vals=onehotencode[1]).forward(inp_)[:, :len(PARAMETERS)]
                                                               if onehotencode else inp_[:, :len(PARAMETERS)])
                                                   if includeparametersinmmd else None),

    # 'mmdfixsigma_output_target_hadflav0':
    #     lambda inp_, out_, target_: mmdfixsigma_fn(out_, target_,
    #                                                 parameters=(OneHotEncode(source_idx=onehotencode[2], target_vals=onehotencode[1]).forward(inp_)[:, :len(PARAMETERS)]
    #                                                             if onehotencode else inp_[:, :len(PARAMETERS)])
    #                                                 if includeparametersinmmd else None,
    #                                                 mask=inp_[:, hadronFlavourIndex] == 0),

    # 'mmdfixsigma_output_target_hadflav4':
    #     lambda inp_, out_, target_: mmdfixsigma_fn(out_, target_,
    #                                                 parameters=(OneHotEncode(source_idx=onehotencode[2], target_vals=onehotencode[1]).forward(inp_)[:, :len(PARAMETERS)]
    #                                                             if onehotencode else inp_[:, :len(PARAMETERS)])
    #                                                 if includeparametersinmmd else None,
    #                                                 mask=inp_[:, hadronFlavourIndex] == 4),

    # 'mmdfixsigma_output_target_hadflav5':
    #     lambda inp_, out_, target_: mmdfixsigma_fn(out_, target_,
    #                                                 parameters=(OneHotEncode(source_idx=onehotencode[2], target_vals=onehotencode[1]).forward(inp_)[:, :len(PARAMETERS)]
    #                                                             if onehotencode else inp_[:, :len(PARAMETERS)])
    #                                                 if includeparametersinmmd else None,
    #                                                 mask=inp_[:, hadronFlavourIndex] == 5),

    'mmdfixsigma_output_target_hadflavSum':
        lambda inp_, out_, target_: hadflav_fraction_0 * mmdfixsigma_fn(out_, target_,
                                                                        parameters=(OneHotEncode(source_idx=onehotencode[2], target_vals=onehotencode[1]).forward(inp_)[:, :len(PARAMETERS)]
                                                                                    if onehotencode else inp_[:, :len(PARAMETERS)])
                                                                        if includeparametersinmmd else None,
                                                                        mask=inp_[:, hadronFlavourIndex] == 0) +
                                    hadflav_fraction_4 * mmdfixsigma_fn(out_, target_,
                                                                        parameters=(OneHotEncode(source_idx=onehotencode[2], target_vals=onehotencode[1]).forward(inp_)[:, :len(PARAMETERS)]
                                                                                    if onehotencode else inp_[:, :len(PARAMETERS)])
                                                                        if includeparametersinmmd else None,
                                                                        mask=inp_[:, hadronFlavourIndex] == 4) +
                                    hadflav_fraction_5 * mmdfixsigma_fn(out_, target_,
                                                                        parameters=(OneHotEncode(source_idx=onehotencode[2], target_vals=onehotencode[1]).forward(inp_)[:, :len(PARAMETERS)]
                                                                                    if onehotencode else inp_[:, :len(PARAMETERS)])
                                                                        if includeparametersinmmd else None,
                                                                        mask=inp_[:, hadronFlavourIndex] == 5),

    'mmd_output_target':
        lambda inp_, out_, target_: mmd_fn(out_, target_,
                                           parameters=(OneHotEncode(source_idx=onehotencode[2], target_vals=onehotencode[1]).forward(inp_)[:, :len(PARAMETERS)]
                                                       if onehotencode else inp_[:, :len(PARAMETERS)])
                                           if includeparametersinmmd else None),

    # 'mmd_output_target_hadflav0':
    #     lambda inp_, out_, target_: mmd_fn(out_, target_,
    #                                        parameters=(OneHotEncode(source_idx=onehotencode[2], target_vals=onehotencode[1]).forward(inp_)[:, :len(PARAMETERS)]
    #                                                    if onehotencode else inp_[:, :len(PARAMETERS)])
    #                                        if includeparametersinmmd else None,
    #                                        mask=inp_[:, hadronFlavourIndex] == 0),

    # 'mmd_output_target_hadflav4':
    #     lambda inp_, out_, target_: mmd_fn(out_, target_,
    #                                        parameters=(OneHotEncode(source_idx=onehotencode[2], target_vals=onehotencode[1]).forward(inp_)[:, :len(PARAMETERS)]
    #                                                    if onehotencode else inp_[:, :len(PARAMETERS)])
    #                                        if includeparametersinmmd else None,
    #                                        mask=inp_[:, hadronFlavourIndex] == 4),

    # 'mmd_output_target_hadflav5':
    #     lambda inp_, out_, target_: mmd_fn(out_, target_,
    #                                        parameters=(OneHotEncode(source_idx=onehotencode[2], target_vals=onehotencode[1]).forward(inp_)[:, :len(PARAMETERS)]
    #                                                    if onehotencode else inp_[:, :len(PARAMETERS)])
    #                                        if includeparametersinmmd else None,
    #                                        mask=inp_[:, hadronFlavourIndex] == 5),

    'mmd_output_target_hadflavSum':
        lambda inp_, out_, target_: hadflav_fraction_0 * mmd_fn(out_, target_,
                                                                parameters=(OneHotEncode(source_idx=onehotencode[2], target_vals=onehotencode[1]).forward(inp_)[:, :len(PARAMETERS)]
                                                                            if onehotencode else inp_[:, :len(PARAMETERS)])
                                                                if includeparametersinmmd else None,
                                                                mask=inp_[:, hadronFlavourIndex] == 0) +
                                    hadflav_fraction_4 * mmd_fn(out_, target_,
                                                                parameters=(OneHotEncode(source_idx=onehotencode[2], target_vals=onehotencode[1]).forward(inp_)[:, :len(PARAMETERS)]
                                                                            if onehotencode else inp_[:, :len(PARAMETERS)])
                                                                if includeparametersinmmd else None,
                                                                mask=inp_[:, hadronFlavourIndex] == 4) +
                                    hadflav_fraction_5 * mmd_fn(out_, target_,
                                                                parameters=(OneHotEncode(source_idx=onehotencode[2], target_vals=onehotencode[1]).forward(inp_)[:, :len(PARAMETERS)]
                                                                            if onehotencode else inp_[:, :len(PARAMETERS)])
                                                                if includeparametersinmmd else None,
                                                                mask=inp_[:, hadronFlavourIndex] == 5),

    'mse_output_target':
        lambda inp_, out_, target_: mse_fn(out_, target_),

    'mse_input_output':
        lambda inp_, out_, target_: mse_fn(inp_[:, real_len_parameters:], out_),

    # 'mae_output_target':
    #     lambda inp_, out_, target_: mae_fn(out_, target_),

    # 'mae_input_output':
    #     lambda inp_, out_, target_: mae_fn(inp_[:, real_len_parameters:], out_),

    'huber_output_target':
        lambda inp_, out_, target_: huber_fn(out_, target_),

    'huber_input_output':
        lambda inp_, out_, target_: huber_fn(inp_[:, real_len_parameters:], out_),

    'deepjetsum_mean':
        lambda inp_, out_, target_: SelectDeepJets(deepjetindices=deepjetindicesWithoutParameters,
                                                   sigmoidtransform=calculatelosseswithtransformedvariables,
                                                   nanotransform=True).forward(out_).sum(dim=1).mean(),

    'deepjetsum_std':
        lambda inp_, out_, target_: SelectDeepJets(deepjetindices=deepjetindicesWithoutParameters,
                                                   sigmoidtransform=calculatelosseswithtransformedvariables,
                                                   nanotransform=True).forward(out_).sum(dim=1).std(),

    'robustpartak4sum_mean':
        lambda inp_, out_, target_: SelectDeepJets(deepjetindices=robustpartak4indicesWithoutParameters,
                                                   sigmoidtransform=calculatelosseswithtransformedvariables,
                                                   nanotransform=True).forward(out_).sum(dim=1).mean(),

    'robustpartak4sum_std':
        lambda inp_, out_, target_: SelectDeepJets(deepjetindices=robustpartak4indicesWithoutParameters,
                                                   sigmoidtransform=calculatelosseswithtransformedvariables,
                                                   nanotransform=True).forward(out_).sum(dim=1).std(),

}

# if constraints are specified MDMM algorithm will be used (see SB below)
mdmm_primary_loss = 'mse_output_target'

mdmm_switch_off_criterion = EarlyStopper(metric='mmdfixsigma_output_target', patience=3,
                                         mode='signdiffto', diff_to=0.,
                                         verbose=1)
mdmm_primary_loss_after_switch_off = 'mmd_output_target_hadflavSum'

lr_lambda_factor = 20.
# (loss, epsilon, initial lambda, scale)

#SB: Set following to empty list to do 1-stage MDMM
mdmm_constraints_config = [ 
    ('mmdfixsigma_output_target', 0., -1., 1.),

    ('deepjetsum_mean', 1.),
    ('deepjetsum_std', 0.00067),#SB: measured value in FullSim

    ('robustpartak4sum_mean', 1.),
    ('robustpartak4sum_std', 0.00069),
]
mdmm_constraints = [my_mdmm.EqConstraint(loss_fns[c[0]], c[1], lmbda_init=c[2] if len(c) > 2 else 0., scale=c[3] if len(c) > 3 else 1.)
                    for c in mdmm_constraints_config]

# if no constraints are specified no MDMM is used and these loss scales are used
nomdmm_loss_scales = {

    'mmdfixsigma_output_target': 0.,
    'mmdfixsigma_output_target_hadflav0': 0.,
    'mmdfixsigma_output_target_hadflav4': 0.,
    'mmdfixsigma_output_target_hadflav5': 0.,
    'mmdfixsigma_output_target_hadflavSum': 0.,

    'mmd_output_target': 0.,
    'mmd_output_target_hadflav0': 0.,
    'mmd_output_target_hadflav4': 0.,
    'mmd_output_target_hadflav5': 0.,
    'mmd_output_target_hadflavSum': 1.,

    'mse_output_target': 0.,
    'mse_input_output': 0.,
    'mae_output_target': 0.,
    'mae_input_output': 0.,
    'huber_output_target': 0.,
    'huber_input_output': 0.,

    'deepjetsum_mean': 0.,
    'deepjetsum_std': 0.,
}

if len(mdmm_constraints) > 0:
    optimizer = my_mdmm.MDMM(mdmm_constraints)
    trainer = optimizer.make_optimizer(model.parameters(), optimizer=torch.optim.Adam, lr=learning_rate, lr_lambda_factor=lr_lambda_factor)
else:
    optimizer = torch.optim.Adam
    trainer = optimizer(model.parameters(), lr=learning_rate)


if lr_scheduler_gamma:
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(trainer, gamma=lr_scheduler_gamma, last_epoch=-1)
elif cyclic_lr:
    lr_scheduler = torch.optim.lr_scheduler.CyclicLR(trainer, base_lr=cyclic_lr[0], max_lr=cyclic_lr[1], step_size_up=cyclic_lr[2], cycle_momentum=False)
else:
    lr_scheduler = None

'''
# ##### ##### #####
# start training
# ##### ##### #####
'''
print('\n### start training')

csvfile_train = open(out_path.replace('output', 'traininglog').replace('.root', '_train.csv'), 'w')
csvwriter_train = csv.writer(csvfile_train)

csvfile_validation = open(out_path.replace('output', 'traininglog').replace('.root', '_validation.csv'), 'w')
csvwriter_validation = csv.writer(csvfile_validation)

loss_vals = {}
benchmarks_train = {loss: 0. for loss in loss_fns}
benchmarks_validation = {loss: 0. for loss in loss_fns}
start_points_train = {loss: 0. for loss in loss_fns}
start_points_validation = {loss: 0. for loss in loss_fns}
end_points_train = {loss: 0. for loss in loss_fns}
end_points_validation = {loss: 0. for loss in loss_fns}
mdmm_switch_off = False
iteration = 0
for epoch in range(num_epochs):

    if is_verbose:
        print('\n# epoch {}'.format(epoch + 1))

    model.train()

    epoch_train_loss = 0.
    epoch_validation_loss = 0.

    epoch_mdmm_switch_off_criterion_value = 0.
    if mdmm_switch_off: mdmm_primary_loss = mdmm_primary_loss_after_switch_off

    if epoch == 0:

        if is_verbose:
            print('\nevaluate benchmarks and starting points')

        for data_loader, benchmarks, start_points in zip([train_loader, validation_loader], [benchmarks_train, benchmarks_validation], [start_points_train, start_points_validation]):
            for batch, (inp, target, spectators) in enumerate(data_loader):

                out = model(inp)

                if calculatelosseswithtransformedvariables:

                    if any(tanh200maskWithParameters):
                        inp = TanhTransform(mask=tanh200maskWithParameters, norm=200).forward(inp)
                    if any(tanh200maskWithoutParameters):
                        target = TanhTransform(mask=tanh200maskWithoutParameters, norm=200).forward(target)
                        out = TanhTransform(mask=tanh200maskWithoutParameters, norm=200).forward(out)

                    if any(logitmaskWithParameters):
                        inp = LogitTransform(mask=logitmaskWithParameters, factor=logitfactor, onnxcompatible=False, eps=epsilon, tiny=tiny).forward(inp)
                    if any(logitmaskWithoutParameters):
                        target = LogitTransform(mask=logitmaskWithoutParameters, factor=logitfactor, onnxcompatible=False, eps=epsilon, tiny=tiny).forward(target)
                        out = LogitTransform(mask=logitmaskWithoutParameters, factor=logitfactor, onnxcompatible=False, eps=epsilon, tiny=tiny).forward(out)

                for loss in loss_fns:
                    if 'omniscient' in loss:
                        benchmarks[loss] += loss_fns[loss](
                            torch.cat((inp, spectators[:, hiddenvariables_indices_fast]), dim=1),
                            torch.cat((inp[:, real_len_parameters:], spectators[:, hiddenvariables_indices_fast]), dim=1),
                            torch.cat((target, spectators[:, hiddenvariables_indices_full]), dim=1)
                        ).item()
                        start_points[loss] += loss_fns[loss](
                            torch.cat((inp, spectators[:, hiddenvariables_indices_fast]), dim=1),
                            torch.cat((out, spectators[:, hiddenvariables_indices_fast]), dim=1),
                            torch.cat((target, spectators[:, hiddenvariables_indices_full]), dim=1)
                        ).item()
                    else:
                        benchmarks[loss] += loss_fns[loss](inp, inp[:, real_len_parameters:], target).item()
                        start_points[loss] += loss_fns[loss](inp, out, target).item()

        for loss in loss_fns:
            benchmarks_train[loss] /= len_train_loader
            benchmarks_validation[loss] /= len_validation_loader
            start_points_train[loss] /= len_train_loader
            start_points_validation[loss] /= len_validation_loader

        csvwriter_train.writerow(['epoch', 'iteration'] + [loss for loss in loss_fns] + ['lmbda_' + c[0] for c in mdmm_constraints_config] + ['epsilon_' + c[0] for c in mdmm_constraints_config])
        csvwriter_train.writerow(['benchmark', 'benchmark'] + [benchmarks_train[loss] for loss in loss_fns] + [c[1] for c in mdmm_constraints_config] + [c.value.item() if hasattr(c, 'value') else -1. for c in mdmm_constraints])
        csvwriter_train.writerow(['start', 'start'] + [start_points_train[loss] for loss in loss_fns] + [c.lmbda.item() for c in mdmm_constraints] + [c.value.item() if hasattr(c, 'value') else -1. for c in mdmm_constraints])
        
        csvwriter_validation.writerow(['epoch', 'iteration'] + [loss for loss in loss_fns] + ['lmbda_' + c[0] for c in mdmm_constraints_config] + ['epsilon_' + c[0] for c in mdmm_constraints_config])
        csvwriter_validation.writerow(['benchmark', 'benchmark'] + [benchmarks_validation[loss] for loss in loss_fns] + [c[1] for c in mdmm_constraints_config] + [c.value.item() if hasattr(c, 'value') else -1. for c in mdmm_constraints])
        csvwriter_validation.writerow(['start', 'start'] + [start_points_validation[loss] for loss in loss_fns] + [c.lmbda.item() for c in mdmm_constraints] + [c.value.item() if hasattr(c, 'value') else -1. for c in mdmm_constraints])


        print('\nbenchmarks')
        print(', '.join([loss for loss in loss_fns]))
        print(', '.join([str(benchmarks_train[loss]) for loss in loss_fns]))
        print('\nstart points')
        print(', '.join([loss for loss in loss_fns]))
        print(', '.join([str(start_points_train[loss]) for loss in loss_fns]))
        print('')

        if save_snapshots:
            
            model.eval()
            with torch.no_grad():
                
                for batch, (inp, target, spectators) in enumerate(validation_loader):

                    if batch >= snapshot_num_batches: break

                    out = model(inp)

                    if batch == 0:
                        inp_list = inp
                        target_list = target
                        out_list = out
                    else:
                        inp_list = torch.cat((inp_list, inp))
                        target_list = torch.cat((target_list, target))
                        out_list = torch.cat((out_list, out))

                    if any(tanh200maskWithParameters):
                        inp = TanhTransform(mask=tanh200maskWithParameters, norm=200).forward(inp)
                    if any(tanh200maskWithoutParameters):
                        target = TanhTransform(mask=tanh200maskWithoutParameters, norm=200).forward(target)
                        out = TanhTransform(mask=tanh200maskWithoutParameters, norm=200).forward(out)

                    if any(logitmaskWithParameters):
                        inp = LogitTransform(mask=logitmaskWithParameters, factor=logitfactor, onnxcompatible=False, eps=epsilon, tiny=tiny).forward(inp)
                    if any(logitmaskWithoutParameters):
                        target = LogitTransform(mask=logitmaskWithoutParameters, factor=logitfactor, onnxcompatible=False, eps=epsilon, tiny=tiny).forward(target)
                        out = LogitTransform(mask=logitmaskWithoutParameters, factor=logitfactor, onnxcompatible=False, eps=epsilon, tiny=tiny).forward(out)

                    if batch == 0:
                        inp_list_transformed = inp
                        target_list_transformed = target
                        out_list_transformed = out
                    else:
                        inp_list_transformed = torch.cat((inp_list_transformed, inp))
                        target_list_transformed = torch.cat((target_list_transformed, target))
                        out_list_transformed = torch.cat((out_list_transformed, out))

                snapshot(inp_list, target_list, out_list, epoch, is_transformed=False, plot_kde=snapshot_plot_kde)
                snapshot(inp_list_transformed, target_list_transformed, out_list_transformed, epoch, is_transformed=True, plot_kde=snapshot_plot_kde)

            model.train()

    if is_verbose:
        print('\ntraining loop')

    trainer.zero_grad()
    for batch, (inp, target, spectators) in enumerate(train_loader):

        if is_verbose:
            print('batch {}'.format(batch))

        out = model(inp)

        if calculatelosseswithtransformedvariables:

            if any(tanh200maskWithParameters):
                inp = TanhTransform(mask=tanh200maskWithParameters, norm=200).forward(inp)
            if any(tanh200maskWithoutParameters):
                target = TanhTransform(mask=tanh200maskWithoutParameters, norm=200).forward(target)
                out = TanhTransform(mask=tanh200maskWithoutParameters, norm=200).forward(out)

            if any(logitmaskWithParameters):
                inp = LogitTransform(mask=logitmaskWithParameters, factor=logitfactor, onnxcompatible=False, eps=epsilon, tiny=tiny).forward(inp)
            if any(logitmaskWithoutParameters):
                target = LogitTransform(mask=logitmaskWithoutParameters, factor=logitfactor, onnxcompatible=False, eps=epsilon, tiny=tiny).forward(target)
                out = LogitTransform(mask=logitmaskWithoutParameters, factor=logitfactor, onnxcompatible=False, eps=epsilon, tiny=tiny).forward(out)

        for loss in loss_fns:
            if 'omniscient' in loss:
                loss_vals[loss] = loss_fns[loss](
                    torch.cat((inp, spectators[:, hiddenvariables_indices_fast]), dim=1),
                    torch.cat((out, spectators[:, hiddenvariables_indices_fast]), dim=1),
                    torch.cat((target, spectators[:, hiddenvariables_indices_full]), dim=1)
                )
            else:
                loss_vals[loss] = loss_fns[loss](inp, out, target)

        train_loss = None
        if len(mdmm_constraints) > 0:
            mdmm_return = optimizer(loss_vals[mdmm_primary_loss], inp, out, target, switch_off_constraints=mdmm_switch_off)
            train_loss = mdmm_return.value
        else:
            for loss in loss_fns:
                if loss not in nomdmm_loss_scales or nomdmm_loss_scales[loss] == 0.: continue
                if train_loss is None:
                    train_loss = nomdmm_loss_scales[loss] * loss_vals[loss]
                else:
                    train_loss += nomdmm_loss_scales[loss] * loss_vals[loss]

        train_loss.backward()

        trainer.step()
        trainer.zero_grad()

        if lr_scheduler is not None and cyclic_lr:
            lr_scheduler.step()

        epoch_train_loss += train_loss.item()
        if epoch + 1 == num_epochs:
            for loss in loss_fns:
                end_points_train[loss] += loss_vals[loss].item()

        csvwriter_train.writerow([epoch, iteration] + [loss_vals[loss].item() for loss in loss_fns] + [c.lmbda.item() for c in mdmm_constraints] + [c.value.item() if hasattr(c, 'value') else -1. for c in mdmm_constraints])

        if is_verbose:

            print('train loss {:.5f}'.format(train_loss.item()))

        iteration += 1

    print('[{} / {}] train loss: {:.10f}'.format(epoch + 1, num_epochs, epoch_train_loss / len_train_loader))

    if is_verbose:
        print('\nvalidation loop')
        
    model.eval()
    with torch.no_grad():

        for batch, (inp, target, spectators) in enumerate(validation_loader):

            if is_verbose:
                print('batch {}'.format(batch))

            out = model(inp)
            
            if save_snapshots and epoch % snapshot_everyXepoch == 0 and batch < snapshot_num_batches:
                if batch == 0:
                    out_list = out
                else:
                    out_list = torch.cat((out_list, out))

            if calculatelosseswithtransformedvariables:

                if any(tanh200maskWithParameters):
                    inp = TanhTransform(mask=tanh200maskWithParameters, norm=200).forward(inp)
                if any(tanh200maskWithoutParameters):
                    target = TanhTransform(mask=tanh200maskWithoutParameters, norm=200).forward(target)
                    out = TanhTransform(mask=tanh200maskWithoutParameters, norm=200).forward(out)

                if any(logitmaskWithParameters):
                    inp = LogitTransform(mask=logitmaskWithParameters, factor=logitfactor, onnxcompatible=False, eps=epsilon, tiny=tiny).forward(inp)
                if any(logitmaskWithoutParameters):
                    target = LogitTransform(mask=logitmaskWithoutParameters, factor=logitfactor, onnxcompatible=False, eps=epsilon, tiny=tiny).forward(target)
                    out = LogitTransform(mask=logitmaskWithoutParameters, factor=logitfactor, onnxcompatible=False, eps=epsilon, tiny=tiny).forward(out)

            if save_snapshots and epoch % snapshot_everyXepoch == 0 and batch < snapshot_num_batches:
                if batch == 0:
                    out_list_transformed = out
                else:
                    out_list_transformed = torch.cat((out_list_transformed, out))

            for loss in loss_fns:
                if 'omniscient' in loss:
                    loss_vals[loss] = loss_fns[loss](
                        torch.cat((inp, spectators[:, hiddenvariables_indices_fast]), dim=1),
                        torch.cat((out, spectators[:, hiddenvariables_indices_fast]), dim=1),
                        torch.cat((target, spectators[:, hiddenvariables_indices_full]), dim=1)
                    )
                else:
                    loss_vals[loss] = loss_fns[loss](inp, out, target)
                
            
            validation_loss = None
            if len(mdmm_constraints) > 0:
                mdmm_return = optimizer(loss_vals[mdmm_primary_loss], inp, out, target, switch_off_constraints=mdmm_switch_off, allow_adaptive_epsilon=not mdmm_switch_off)
                validation_loss = mdmm_return.value
            else:
                for loss in loss_fns:
                    if loss not in nomdmm_loss_scales or nomdmm_loss_scales[loss] == 0.: continue
                    if validation_loss is None:
                        validation_loss = nomdmm_loss_scales[loss] * loss_vals[loss]
                    else:
                        validation_loss += nomdmm_loss_scales[loss] * loss_vals[loss]


            if len(mdmm_constraints) > 0 and mdmm_switch_off_criterion and isinstance(mdmm_switch_off_criterion, EarlyStopper) and not mdmm_switch_off:
                if mdmm_switch_off_criterion.metric.startswith('lmbda_'):
                    epoch_mdmm_switch_off_criterion_value += {cc[0]: c.lmbda.item() for cc, c in zip(mdmm_constraints_config, mdmm_constraints)}[mdmm_switch_off_criterion.metric.replace('lmbda_', '')]
                else:
                    epoch_mdmm_switch_off_criterion_value += loss_vals[mdmm_switch_off_criterion.metric].item()


            epoch_validation_loss += validation_loss.item()
            if epoch + 1 == num_epochs:
                for loss in loss_fns:
                    end_points_validation[loss] += loss_vals[loss].item()

            csvwriter_validation.writerow([epoch, iteration] + [loss_vals[loss].item() for loss in loss_fns] + [c.lmbda.item() for c in mdmm_constraints] + [c.value.item() if hasattr(c, 'value') else -1. for c in mdmm_constraints])

            if is_verbose:
                print('validation loss {:.5f}'.format(validation_loss.item()))

        print('[{} / {}] validation loss: {:.10f}'.format(epoch + 1, num_epochs, epoch_validation_loss / len_validation_loader))

        if save_snapshots and epoch % snapshot_everyXepoch == 0:
            snapshot(inp_list, target_list, out_list, epoch+1, is_transformed=False, plot_kde=snapshot_plot_kde)
            snapshot(inp_list_transformed, target_list_transformed, out_list_transformed, epoch+1, is_transformed=True, plot_kde=snapshot_plot_kde)

    if lr_scheduler is not None and lr_scheduler_gamma:
        lr_scheduler.step()

    if len(mdmm_constraints) > 0 and mdmm_switch_off_criterion and not mdmm_switch_off:
        if isinstance(mdmm_switch_off_criterion, EarlyStopper):
            mdmm_switch_off = mdmm_switch_off_criterion.is_converged(epoch_mdmm_switch_off_criterion_value / len_validation_loader)
        else:
            if epoch >= mdmm_switch_off_criterion:
                mdmm_switch_off = True

    if epoch + 1 == num_epochs:

        for loss in loss_fns:
            end_points_train[loss] /= len_train_loader
            end_points_validation[loss] /= len_validation_loader

        print('\nend points train')
        print(', '.join([loss for loss in loss_fns]))
        print(', '.join([str(end_points_train[loss]) for loss in loss_fns]))
        print('\nend points validation')
        print(', '.join([loss for loss in loss_fns]))
        print(', '.join([str(end_points_validation[loss]) for loss in loss_fns]))


'''
# ##### ##### #####
# save output
# ##### ##### #####
'''

out_dict = {'isTrainValTest': []}
for branch in dict_input:
    out_dict[branch] = []
for branch in dict_target:
    out_dict[branch] = []
for branch in dict_target:
    out_dict[branch.replace('FullSim', 'Refined')] = []
for branch in dict_spectators:
    out_dict[branch] = []

model.eval()

with torch.no_grad():

    for i, (inp, target, spectators) in enumerate(train_loader):

        out = model(inp)

        out_dict['isTrainValTest'].append(torch.ones(inp.size(dim=0), dtype=torch.int) * 0)

        for ib, branch in enumerate(dict_input):
            out_dict[branch].append(inp[:, ib])
        for ib, branch in enumerate(dict_target):
            out_dict[branch].append(target[:, ib])
        for ib, branch in enumerate(dict_target):
            out_dict[branch.replace('FullSim', 'Refined')].append(out[:, ib])
        for ib, branch in enumerate(dict_spectators):
            out_dict[branch].append(spectators[:, ib])

    for i, (inp, target, spectators) in enumerate(validation_loader):

        out = model(inp)

        out_dict['isTrainValTest'].append(torch.ones(inp.size(dim=0), dtype=torch.int) * 1)

        for ib, branch in enumerate(dict_input):
            out_dict[branch].append(inp[:, ib])
        for ib, branch in enumerate(dict_target):
            out_dict[branch].append(target[:, ib])
        for ib, branch in enumerate(dict_target):
            out_dict[branch.replace('FullSim', 'Refined')].append(out[:, ib])
        for ib, branch in enumerate(dict_spectators):
            out_dict[branch].append(spectators[:, ib])

    for i, (inp, target, spectators) in enumerate(test_loader):

        out = model(inp)

        out_dict['isTrainValTest'].append(torch.ones(inp.size(dim=0), dtype=torch.int) * 2)

        for ib, branch in enumerate(dict_input):
            out_dict[branch].append(inp[:, ib])
        for ib, branch in enumerate(dict_target):
            out_dict[branch].append(target[:, ib])
        for ib, branch in enumerate(dict_target):
            out_dict[branch.replace('FullSim', 'Refined')].append(out[:, ib])
        for ib, branch in enumerate(dict_spectators):
            out_dict[branch].append(spectators[:, ib])

for branch in out_dict:
    out_dict[branch] = torch.cat(out_dict[branch]).detach().cpu().numpy()

# out_rdf = ROOT.RDF.MakeNumpyDataFrame(out_dict)
out_rdf = ROOT.RDF.FromNumpy(out_dict)
out_rdf.Snapshot(in_tree, out_path)
print('just created ' + out_path)

csvfile_train.close()
print('just created ' + csvfile_train.name)

csvfile_validation.close()
print('just created ' + csvfile_validation.name)


'''
# ##### ##### #####
# save model
# ##### ##### #####
'''

m = torch.jit.script(model)
torch.jit.save(m, out_path.replace('output', 'model').replace('.root', '.pt'))
print('\njust created ' + out_path.replace('output', 'model').replace('.root', '.pt'))
