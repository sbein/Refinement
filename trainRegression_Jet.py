"""
to load packages, e.g.:
source /cvmfs/sft.cern.ch/lcg/views/LCG_101cuda/x86_64-centos7-gcc8-opt/setup.sh
- or -
source /cvmfs/sft.cern.ch/lcg/views/LCG_101/x86_64-centos7-gcc8-opt/setup.sh

to write terminal output to a txt file:
python trainRegression.py 2>&1 | tee traininglog_regression.txt

#first 
screen
condor_submit -i interactive.submit 
source /afs/desy.de/user/b/beinsam/.bash_profile
torefinement
cmsenv
cd /nfs/dust/cms/user/beinsam/FastSim/Refinement/Regress
source /cvmfs/sft.cern.ch/lcg/views/LCG_101cuda/x86_64-centos7-gcc8-opt/setup.sh
#to write terminal output to a txt file:
python3 trainRegression_Jet.py 2>&1 | tee traininglog_regressionJet.txt

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

is_test = False
is_verbose = True

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
#in_path = '/nfs/dust/cms/user/wolfmor/Refinement/littletree_CMSSW_12_6_0_TTbar_step2_SIM_RECOBEFMIX_DIGI_L1_DIGI2RAW_L1Reco_RECO_PAT_NANO_coffea_new.root'
in_path = '/nfs/dust/cms/user/beinsam/FastSim/Refinement/output/mc_fullfast_T1tttt_JetsMuonsElectronsPhotonsTausEvents.root'
#in_path = '/nfs/dust/cms/user/beinsam/FastSim/Refinement/mc_fullfastnn_1234_step2_SIM_RECOBEFMIX_DIGI_L1_DIGI2RAW_L1Reco_RECO_PAT_NANO_fromNANO_Ootb.root'
in_tree = 'tJet'
preselection = ''#'GenJet_nearest_dR>0.5&&RecJet_nearest_dR_FastSim>0.5&&RecJet_nearest_dR_FullSim>0.5'

#out_path = '/nfs/dust/cms/user/wolfmor/Refinement/TrainingOutput/output_refinement_regression_' + training_id + '.root'
out_path = '/nfs/dust/cms/user/beinsam/FastSim/Refinement/Regress/TrainingOutput/output_refineJet_regression_' + training_id + '.root'


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
fisherfactor = 1.0# this is the 1/2 that is there in the official definition


'''
# ##### ##### #####
# training settings
# ##### ##### #####
'''

if is_test: num_epochs = 2
else: 
    #num_epochs = 1000
    num_epochs = 100    
    #num_epochs = 2

learning_rate = 1e-5
lr_scheduler_gamma = 1.

if is_test: batch_size = 4096
else: batch_size = 4096

batch_size = 2048

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
    ('RecJet_btagDeepFlavQG_CLASS', ['logit'])
]

spectators = [
    'GenJet_pt',
    'GenJet_eta',
    'GenJet_phi',
    'GenJet_mass',
    'GenJet_hadronFlavour',
    'GenJet_partonFlavour',

]
excludespectators = [var[0] for var in PARAMETERS + VARIABLES]
SPECTATORS = [
    s for s in spectators if s not in excludespectators
    and s.replace('CLASS', 'FastSim') not in excludespectators
    and s.replace('CLASS', 'FullSim') not in excludespectators
]


'''
# ##### ##### #####
# get training data
# ##### ##### #####
'''
print('\n### get training data')

ntrain = num_batches[0] * batch_size
nval = num_batches[1] * batch_size
ntest = num_batches[2] * batch_size

print('ntest', ntest)
if ntest == 0:
    rdf = ROOT.RDataFrame(in_tree, in_path).Filter(preselection)
else:
    #rdf = ROOT.RDataFrame(in_tree, in_path).Filter(preselection).Range(ntrain + nval + ntest)
    rdf = ROOT.RDataFrame(in_tree, in_path).Filter('1').Range(ntrain + nval + ntest)

#print('looking for something like print branches', dir(rdf))
#column_names = rdf.GetColumnNames()
#print("Branches/Columns in the DataFrame:")
#for name in column_names:
#    print(name)

dict_input = rdf.AsNumpy([n[0].replace('CLASS', 'FastSim') for n in PARAMETERS + VARIABLES if '_ohe_' not in n[0]])
dict_target = rdf.AsNumpy([n[0].replace('CLASS', 'FullSim') for n in VARIABLES])
dict_spectators = rdf.AsNumpy([n for n in SPECTATORS if 'CLASS' not in n]
                              + [n.replace('CLASS', 'FastSim') for n in SPECTATORS if 'CLASS' in n]
                              + [n.replace('CLASS', 'FullSim') for n in SPECTATORS if 'CLASS' in n])

my_dtype = torch.float32
data_input = torch.tensor(np.stack([dict_input[var] for var in dict_input], axis=1), dtype=my_dtype, device=device)
data_target = torch.tensor(np.stack([dict_target[var] for var in dict_target], axis=1), dtype=my_dtype, device=device)
data_spec = torch.tensor(np.stack([dict_spectators[var] for var in dict_spectators], axis=1), dtype=my_dtype, device=device)

dataset = torch.utils.data.TensorDataset(data_input, data_target, data_spec)

if ntest == 0:
    ntest = len(dataset) - (ntrain + nval)

if rdf.Count().GetValue() < ntrain + nval + ntest:
    raise Exception('input dataset too small, choose smaller/fewer batches')

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [ntrain, nval, ntest])


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

hadronFlavourIndex = [name[0] for name in PARAMETERS].index('RecJet_hadronFlavour_FastSim')

len_data_input = len(data_input)
hadflav_fraction_0 = len(data_input[data_input[:, hadronFlavourIndex] == 0]) / len_data_input
hadflav_fraction_4 = len(data_input[data_input[:, hadronFlavourIndex] == 4]) / len_data_input
hadflav_fraction_5 = len(data_input[data_input[:, hadronFlavourIndex] == 5]) / len_data_input


deepjetindicesWithParameters = [idx for idx, name in enumerate(PARAMETERS + VARIABLES) if 'Jet_btagDeepFlav' in name[0]]
deepjetindicesWithoutParameters = [idx for idx, name in enumerate(VARIABLES) if 'Jet_btagDeepFlav' in name[0]]

logitmaskWithParameters = [int('logit' in name[1]) for name in PARAMETERS + VARIABLES]
logitmaskWithoutParameters = [int('logit' in name[1]) for name in VARIABLES]

log10maskWithParameters = [int('log10' in name[1]) for name in PARAMETERS + VARIABLES]
log10maskWithoutParameters = [int('log10' in name[1]) for name in VARIABLES]

fishermaskWithParameters = [int('fisher' in name[1]) for name in PARAMETERS + VARIABLES]
fishermaskWithoutParameters = [int('fisher' in name[1]) for name in VARIABLES]


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

if any(tanh200maskWithParameters): model.add_module('Tanh200Transform', TanhTransform(mask=tanh200maskWithParameters, norm=200))
if any(logitmaskWithParameters): model.add_module('LogitTransform', LogitTransform(mask=logitmaskWithParameters, factor=logitfactor, onnxcompatible=onnxcompatible, eps=epsilon, tiny=tiny))
if any(log10maskWithParameters): model.add_module('log10Transform', logTransform(mask=log10maskWithParameters, base=10, onnxcompatible=onnxcompatible))
if any(fishermaskWithParameters): model.add_module('FisherTransform', FisherTransform(mask=fishermaskWithParameters, factor=fisherfactor, onnxcompatible=onnxcompatible, eps=epsilon))

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

mmdfixsigma_fn = my_mmd.MMD(kernel_mul=5., kernel_num=5,calculate_fix_sigma_for_each_dimension_with_target_only=True)# fix_sigma=true by default
mmd_fn = my_mmd.MMD(kernel_mul=5., kernel_num=5)
mse_fn = torch.nn.MSELoss()
mae_fn = torch.nn.L1Loss()
huber_fn = torch.nn.HuberLoss(delta=0.1)

# these will be evaluated/monitored during training and can be used to update the model (see below)
loss_fns = {

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

    'mmd_output_target_hadflav0':
        lambda inp_, out_, target_: mmd_fn(out_, target_,
                                           parameters=(OneHotEncode(source_idx=onehotencode[2], target_vals=onehotencode[1]).forward(inp_)[:, :len(PARAMETERS)]
                                                       if onehotencode else inp_[:, :len(PARAMETERS)])
                                           if includeparametersinmmd else None,
                                           mask=inp_[:, hadronFlavourIndex] == 0),

    'mmd_output_target_hadflav4':
        lambda inp_, out_, target_: mmd_fn(out_, target_,
                                           parameters=(OneHotEncode(source_idx=onehotencode[2], target_vals=onehotencode[1]).forward(inp_)[:, :len(PARAMETERS)]
                                                       if onehotencode else inp_[:, :len(PARAMETERS)])
                                           if includeparametersinmmd else None,
                                           mask=inp_[:, hadronFlavourIndex] == 4),

    'mmd_output_target_hadflav5':
        lambda inp_, out_, target_: mmd_fn(out_, target_,
                                           parameters=(OneHotEncode(source_idx=onehotencode[2], target_vals=onehotencode[1]).forward(inp_)[:, :len(PARAMETERS)]
                                                       if onehotencode else inp_[:, :len(PARAMETERS)])
                                           if includeparametersinmmd else None,
                                           mask=inp_[:, hadronFlavourIndex] == 5),

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


}

# if constraints are specified MDMM algorithm will be used
mdmm_primary_loss = 'mmd_output_target_hadflavSum'
mdmm_constraints_config = [
    ('deepjetsum_mean', 1.),
    ('deepjetsum_std', 0.001),
    # ('huber_output_target', 0.00053),
]
mdmm_constraints = [my_mdmm.EqConstraint(loss_fns[c[0]], c[1]) for c in mdmm_constraints_config]

# if no constraints are specified no MDMM is used and these loss scales are used
nomdmm_loss_scales = {

    'mmdfixsigma_output_target': 0.,
    'mmdfixsigma_output_target_hadflav0': 0.,
    'mmdfixsigma_output_target_hadflav4': 0.,
    'mmdfixsigma_output_target_hadflav5': 0.,
    'mmdfixsigma_output_target_hadflavSum': 1.,

    'mmd_output_target': 0.,
    'mmd_output_target_hadflav0': 0.,
    'mmd_output_target_hadflav4': 0.,
    'mmd_output_target_hadflav5': 0.,
    'mmd_output_target_hadflavSum': 0.,

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
    trainer = optimizer.make_optimizer(model.parameters(), optimizer=torch.optim.Adam, lr=learning_rate)
else:
    optimizer = torch.optim.Adam
    trainer = optimizer(model.parameters(), lr=learning_rate)


lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(trainer, gamma=lr_scheduler_gamma, last_epoch=-1)


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
iteration = 0
for epoch in range(num_epochs):

    if is_verbose:
        print('\n# epoch {}'.format(epoch + 1))

    model.train()

    epoch_train_loss = 0.
    epoch_validation_loss = 0.

    if epoch == 0:

        if is_verbose:
            print('\nevaluate benchmarks and starting points')

        for data_loader, benchmarks, start_points in zip([train_loader, validation_loader], [benchmarks_train, benchmarks_validation], [start_points_train, start_points_validation]):
            for batch, (inp, target, _) in enumerate(data_loader):

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
                    benchmarks[loss] += loss_fns[loss](inp, inp[:, real_len_parameters:], target).item()
                    start_points[loss] += loss_fns[loss](inp, out, target).item()

        for loss in loss_fns:
            benchmarks_train[loss] /= len_train_loader
            benchmarks_validation[loss] /= len_validation_loader
            start_points_train[loss] /= len_train_loader
            start_points_validation[loss] /= len_validation_loader

        csvwriter_train.writerow(['epoch', 'iteration'] + [loss for loss in loss_fns] + ['lmbda_' + c[0] for c in mdmm_constraints_config])
        csvwriter_train.writerow(['benchmark', 'benchmark'] + [benchmarks_train[loss] for loss in loss_fns] + [c[1] for c in mdmm_constraints_config])
        csvwriter_train.writerow(['start', 'start'] + [start_points_train[loss] for loss in loss_fns] + [c.lmbda.item() for c in mdmm_constraints])
        
        csvwriter_validation.writerow(['epoch', 'iteration'] + [loss for loss in loss_fns] + ['lmbda_' + c[0] for c in mdmm_constraints_config])
        csvwriter_validation.writerow(['benchmark', 'benchmark'] + [benchmarks_validation[loss] for loss in loss_fns] + [c[1] for c in mdmm_constraints_config])
        csvwriter_validation.writerow(['start', 'start'] + [start_points_validation[loss] for loss in loss_fns] + [c.lmbda.item() for c in mdmm_constraints])


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
                
                for batch, (inp, target, _) in enumerate(validation_loader):

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
    for batch, (inp, target, _) in enumerate(train_loader):

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
            loss_vals[loss] = loss_fns[loss](inp, out, target)

        train_loss = None
        if len(mdmm_constraints) > 0:
            mdmm_return = optimizer(loss_vals[mdmm_primary_loss], inp, out, target)
            train_loss = mdmm_return.value
        else:
            for loss in loss_fns:
                if nomdmm_loss_scales[loss] == 0.: continue
                if train_loss is None:
                    train_loss = nomdmm_loss_scales[loss] * loss_vals[loss]
                else:
                    train_loss += nomdmm_loss_scales[loss] * loss_vals[loss]

        train_loss.backward()

        trainer.step()
        trainer.zero_grad()

        epoch_train_loss += train_loss.item()
        if epoch + 1 == num_epochs:
            for loss in loss_fns:
                end_points_train[loss] += loss_vals[loss].item()

        csvwriter_train.writerow([epoch, iteration] + [loss_vals[loss].item() for loss in loss_fns] + [c.lmbda.item() for c in mdmm_constraints])

        if is_verbose:
            print('train loss {:.5f}'.format(train_loss.item()))

        iteration += 1

    print('[{} / {}] train loss: {:.10f}'.format(epoch + 1, num_epochs, epoch_train_loss / len_train_loader))

    if is_verbose:
        print('\nvalidation loop')
        
    model.eval()
    with torch.no_grad():

        for batch, (inp, target, _) in enumerate(validation_loader):

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
                loss_vals[loss] = loss_fns[loss](inp, out, target)
                
            
            validation_loss = None
            if len(mdmm_constraints) > 0:
                mdmm_return = optimizer(loss_vals[mdmm_primary_loss], inp, out, target)
                validation_loss = mdmm_return.value
            else:
                for loss in loss_fns:
                    if nomdmm_loss_scales[loss] == 0.: continue
                    if validation_loss is None:
                        validation_loss = nomdmm_loss_scales[loss] * loss_vals[loss]
                    else:
                        validation_loss += nomdmm_loss_scales[loss] * loss_vals[loss]


            epoch_validation_loss += validation_loss.item()
            if epoch + 1 == num_epochs:
                for loss in loss_fns:
                    end_points_validation[loss] += loss_vals[loss].item()

            csvwriter_validation.writerow([epoch, iteration] + [loss_vals[loss].item() for loss in loss_fns] + [c.lmbda.item() for c in mdmm_constraints])

            if is_verbose:
                print('validation loss {:.5f}'.format(validation_loss.item()))

        print('[{} / {}] validation loss: {:.10f}'.format(epoch + 1, num_epochs, epoch_train_loss / len_train_loader))

        if save_snapshots and epoch % snapshot_everyXepoch == 0:
            snapshot(inp_list, target_list, out_list, epoch+1, is_transformed=False, plot_kde=snapshot_plot_kde)
            snapshot(inp_list_transformed, target_list_transformed, out_list_transformed, epoch+1, is_transformed=True, plot_kde=snapshot_plot_kde)

    lr_scheduler.step()

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
        print('\nend of cheeky loop')


'''
# ##### ##### #####
# save model
# ##### ##### #####
'''
print('\nsaving model')

m = torch.jit.script(model)
torch.jit.save(m, out_path.replace('output', 'model').replace('.root', '.pt'))
print('\njust created ' + out_path.replace('output', 'model').replace('.root', '.pt'))


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

out_rdf = ROOT.RDF.MakeNumpyDataFrame(out_dict)
out_rdf.Snapshot('tJet', out_path)
print('just created ' + out_path)

csvfile_train.close()
print('just created ' + csvfile_train.name)

csvfile_validation.close()
print('just created ' + csvfile_validation.name)
