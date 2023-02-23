"""
source /cvmfs/sft.cern.ch/lcg/views/LCG_101cuda/x86_64-centos7-gcc8-opt/setup.sh

python trainRegression.py 2>&1 | tee traininglog_regression_
"""

import os
import sys
import csv
import platform
import time
from datetime import datetime

import ROOT
import torch
from torch import nn
import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use('Agg')  # using Agg backend (without X-server running)
import matplotlib.pyplot as plt
from matplotlib import colors

import seaborn as sns

import my_mdmm as mdmm
import my_mmd
from my_modules import *


def snapshot(_X, _y, _output, _epoch, is_transformed=False):

    print('snapshot!')
    start = time.time()

    palette = {'Refined': 'tab:blue', 'Fast': 'tab:red', 'Full': 'tab:green'}

    # iterators will be used for Fast/Full in parallel and then for Refined _or_ only for Refined

    columns = [n[0].replace('_CLASS', '') for n in realPARAMETERS + VARIABLES]
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
    # _y = torch.cat((_X[:, :realLenParameters], _y), axis=1)
    # _output = torch.cat((_X[:, :realLenParameters], _output), axis=1)
    #
    # # the order defines what is plotted on top
    # data = torch.cat((_X, _y, _output)).cpu().numpy()
    # categories = ['Fast'] * _X.shape[0] + ['Full'] * _y.shape[0] + ['Refined'] * _output.shape[0]
    #
    # df = pd.DataFrame({n[0].replace('_CLASS', ''): data[:, idim] for idim, n in enumerate(realPARAMETERS + VARIABLES)})
    # df['Category'] = categories
    #
    # g = sns.PairGrid(df, vars=columns, hue='Category', palette={'Refined': 'tab:blue', 'Fast': 'tab:red', 'Full': 'tab:green'}, diag_sharey=False)
    #
    # g.map_diag(myhist, fill=False, element='bars')
    # g.map_upper(sns.scatterplot)
    # g.map_lower(mykde, levels=[1-0.997, 1-0.955, 1-0.683])
    #
    # g.add_legend(title='Epoch ' + str(_epoch))


    # ----------------------------------------------
    # to plot Fast/Full only for first plot:

    if _epoch == 0:

        globals()['snapshot_parameters' + str(is_transformed)] = _X[:, :realLenParameters]

        # add parameters to Full list
        _y = torch.cat((globals()['snapshot_parameters' + str(is_transformed)], _y), axis=1)

        dataXy = torch.cat((_X, _y)).cpu().numpy()
        categoriesXy = ['Fast'] * _X.shape[0] + ['Full'] * _y.shape[0]
        dfXy = pd.DataFrame({n[0].replace('_CLASS', ''): dataXy[:, idim] for idim, n in enumerate(realPARAMETERS + VARIABLES)})
        dfXy['Category'] = categoriesXy

        globals()['snapshot_g' + str(is_transformed)] = sns.PairGrid(dfXy, vars=columns, hue='Category', palette=palette, diag_sharey=False)

        globals()['snapshot_g' + str(is_transformed)].map_diag(myhist, fill=False, element='bars')
        globals()['snapshot_g' + str(is_transformed)].map_upper(sns.scatterplot)
        globals()['snapshot_g' + str(is_transformed)].map_lower(mykde, levels=[1-0.997, 1-0.955, 1-0.683])


    # ... and then add Refined:

    # add parameters to Refined list
    _output = torch.cat((globals()['snapshot_parameters' + str(is_transformed)], _output), axis=1)

    dataOutput = _output.cpu().numpy()
    categoriesOutput = ['Refined'] * _output.shape[0]
    dfOutput = pd.DataFrame({n[0].replace('_CLASS', ''): dataOutput[:, idim] for idim, n in enumerate(realPARAMETERS + VARIABLES)})
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
    g.map_lower(mykde, levels=[1-0.997, 1-0.955, 1-0.683])


    g.figure.set_size_inches(15, 15)
    if g.legend:
        g.legend.remove()
    g.add_legend(title='Epoch ' + str(_epoch))


    # ----------------------------------------------

    outpath = '/afs/desy.de/user/w/wolfmor/Plots/Refinement/Training/Regression/Snapshots/' + trainingID
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    outname = trainingID
    if is_transformed: outname += '_transformed'
    outname += '_epoch_' + str(_epoch)

    g.savefig(outpath + '/' + outname + '.png')

    end = time.time()
    print(end - start)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')


# TODO: adapt trainingID
trainingID = datetime.today().strftime('%Y%m%d') + ''


# general options
is_fatjet_delphes = False
is_test = False
is_verbose = False
disable_training = False
savesnapshots = False
snaphshotnumbatches = 4  # 10
snapshoteveryXepoch = 9
# snapshots take quite some time due to kde analysis for correlation plots...
# estimates: ~60s for 1024 jets Fast/Full/Refined (~25s for only Refined)
# scales quite linearly with number of jets
# e.g. for 10 batches w/ 1024 jets and 10 epochs: 1 * 10*60s + 9 * 10*25 = 2,850s = 47.5min = 0.8h
# e.g. for 10 batches w/ 1024 jets and 100 epochs: 1 * 10*60s + 99 * 10*25 = 25,350s = 422.5min = 7h
# e.g. for 100 batches w/ 1024 jets and 10 epochs: 1 * 100*60s + 9 * 100*25 = 28,500s = 475min = 8h

# model options
onnxcompatible = True
adddeepjetconstraint = True
transformdeepjet5to4 = False
transformdeepjet4to4 = False  # TODO???: True  # for new samples without C discr.
castto16bit = False  # because Fast/Full are only 16 bit precision  # DON'T DO THIS! it messes with the val loss
useskipconnections = True
usedensesequential = False  # only with skip connections
triangleshape = True  # also adapt number of hidden layers!
print('onnx?', onnxcompatible)

# loss options
includeparamsinmmd = True
individualmmdshadflav = True
normalizeindividualmmdshadflavtostart = False
accumulategradients = 1  # 10
startepochaccgrads = 50

if is_fatjet_delphes:
    preselection = 'GenFatJet_nearest_dR>0.5&&RecFatJet_nearest_dR_FastSim>0.5&&RecFatJet_nearest_dR_FullSim>0.5'  # useless for FatJets
else:
    # preselection = 'GenJet_nearest_dR>0.4&&RecJet_nearest_dR_FastSim>0.4&&RecJet_nearest_dR_FullSim>0.4'  # efficiency 99.8%
    preselection = 'GenJet_nearest_dR>0.5&&RecJet_nearest_dR_FastSim>0.5&&RecJet_nearest_dR_FullSim>0.5'  # efficiency 77.4%
    # preselection = 'GenJet_nearest_dR>0.5&&RecJet_nearest_dR_FastSim>0.5&&RecJet_nearest_dR_FullSim>0.5&&RecJet_jetId_FastSim>5&&RecJet_jetId_FullSim>5'

if is_fatjet_delphes:
    onehotencode = []
else:
    onehotencode = [
        # ('GenJet_hadronFlavour', [0, 4, 5]),
        ('RecJet_hadronFlavour_FastSim', [0, 4, 5]),

        # ('GenFatJet_hadronFlavour', [0, 4, 5]),
    ]

usemdmm = False
# (target value, scale)
constraintsconfig = {
    # 'mmd': (0., 1.),
    # 'mmd_response': (0., 1.),
    # 'mse_fast': (0.1, 1.),  # (0.02, 1.)
    # 'mse_full': (0.5, 1.),  # bm = ~2
    # 'mae_fast': ,
    'mae_full': (0.081, 1.),
}
if not usemdmm: constraintsconfig = {}
print('mmd?', usemdmm)
print(constraintsconfig)

logitfactor = 1.

if is_fatjet_delphes:
    # in_path = '/nfs/dust/cms/user/wolfmor/Refinement/littletree_CMSSW_10_1_0_Delphes_Top1_coffea_new.root'
    in_path = '/nfs/dust/cms/user/wolfmor/Refinement/littletree_CMSSW_10_1_0_Delphes_FastBeta0p9Res2x_Top1_coffea_new.root'
else:
    # in_path = '/nfs/dust/cms/user/wolfmor/Refinement/littletree_step3_inNANOAODSIM_coffea.root'  # 878062 jets
    # in_path = '/nfs/dust/cms/user/wolfmor/Refinement/littletree_CMSSW_12_2_3_T1tttt_step3_inNANOAODSIM_coffea.root'
    in_path = '/nfs/dust/cms/user/wolfmor/Refinement/littletree_CMSSW_12_6_0_TTbar_step2_SIM_RECOBEFMIX_DIGI_L1_DIGI2RAW_L1Reco_RECO_PAT_NANO_coffea_new.root'

out_path = '/nfs/dust/cms/user/wolfmor/Refinement/TrainingOutput/output_regression_' + trainingID + '.root'

if is_fatjet_delphes:
    treename = 'tFatJet'  # 1M jets
else:
    treename = 'tJet'  # 10.5M jets, 3.5M jets for 12_6 file
    # treename = 'tFatJet'  # 2.9M jets


# TODO: how many jets are available?

if is_test:
    my_batch_size = 4096  # 1024  # 512  # 20  # 512  #
    num_batches = [2, 2, 2]
else:
    if is_fatjet_delphes:
        my_batch_size = 1024
        num_batches = [500, 200, 200]
    else:
        my_batch_size = 4096  # 2048  #  1024  # 10000  # 1024  # 512  # 256  # 128  # 32  # 8192  #
        num_batches = [400, 200, 200]  # [800, 400, 400]  # [500, 500, 500]  # 500  # 1000  [200, 100, 100]  #

# need to be multiple of batchsize because otherwise mmd can be biased
# ntest can be 0, meaning that all remaining jets will be used

ntrain = num_batches[0] * my_batch_size  # 300 * my_batch_size  # 1500 * my_batch_size  # 1200 * my_batch_size  # 2000 * my_batch_size
nval = num_batches[1] * my_batch_size  # 300 * my_batch_size  # 1500 * my_batch_size
ntest = num_batches[2] * my_batch_size  # 1000 * my_batch_size
# ntest = 0  # is this a problem to use all remaining jets? because need all for event-level quantities


num_hidden_layers = 16  # 20  # 16  # 14  # 4  # 2  # 3  # 4
num_layers_per_skipblock = 2  # 4  # 2  # 8  # one "skipblock" counts as one "hidden layer"

nodes_hidden_layer = 512  # 1024  # 128  # 64

if triangleshape:
    assert num_hidden_layers % 2 == 0
    # {2: [16], 4: [16, 32], 6: [16, 32, 64], 8: [16, 32, 64, 128], 10: [16, 32, 64, 128, 256], 12: [16, 32, 64, 128, 256, 512], 14: [16, 32, 64, 128, 256, 512, 1024]}
    nodes_hidden_layer_list = [min(16*2**n, nodes_hidden_layer) for n in range(int(num_hidden_layers/2))]
    nodes_hidden_layer_list += nodes_hidden_layer_list[::-1][1:]
else:
    nodes_hidden_layer_list = [nodes_hidden_layer for _ in range(num_hidden_layers)]

dropout = 0.2  # 0.5

if is_test: my_num_epochs = 2
else: my_num_epochs = 100  # 200  # 20

if is_test: lr = 1e-4  # 1e-2
else: lr = 5e-7  # 1e-5  # 1e-4  # 5e-5

cyclic_lr = True
lr_scheduler_gamma = 0.96  # 0  # if == 0 use reduceOnPlateau


if is_fatjet_delphes:
    PARAMETERS = [
        ('GenFatJet_PT', ['tanh200', 'logit']),
        ('GenFatJet_Eta', []),

        ('GenFatJet_Mass', ['tanh200', 'logit']),  # tanh norm should probably be smaller
        ('GenFatJet_Tau1', ['logit']),
        ('GenFatJet_Tau2', ['logit']),
        ('GenFatJet_Tau3', ['logit']),
        ('GenFatJet_Tau4', ['logit']),
        ('GenFatJet_Tau5', ['logit']),
    ]
else:
    # !!!
    # pt has to be first (for definition of response)
    # !!!
    PARAMETERS = [
        ('GenJet_pt', ['tanh200', 'logit']),
        ('GenJet_eta', []),
        # ('GenJet_hadronFlavour', [])
        ('RecJet_hadronFlavour_FastSim', [])
        # RecJet_hadronFlavour_FastSim == GenJet_hadronFlavour for 99.2%
        # RecJet_hadronFlavour_FastSim == GenJet_hadronFlavour && RecJet_hadronFlavour_FastSim == RecJet_hadronFlavour_FullSim for 98.7 %

        # ('GenFatJet_pt', ['tanh200', 'logit']),
        # ('GenFatJet_eta', []),
        # ('GenFatJet_hadronFlavour', [])
    ]
realPARAMETERS = PARAMETERS.copy()

if is_fatjet_delphes:
    variables = [
        ('FatJet_Mass', ['tanh200', 'logit']),
        ('FatJet_Tau1', ['logit']),
        ('FatJet_Tau2', ['logit']),
        ('FatJet_Tau3', ['logit']),
        ('FatJet_Tau4', ['logit']),
        ('FatJet_Tau5', ['logit'])
    ]
else:
    # !!!
    # pt has to be first (for definition of response)
    # btagDeepFlav values have to be last (due to bad implementation of DeepJetConstraint)
    # !!!
    variables = [
        # ('Jet_pt', ['tanh200', 'logit']),
        ('Jet_btagDeepFlavB', ['logit']),  # if using DeepJetConstraint the transformations have to be explicitly adapted in the DeepJetConstraint module
        # ('Jet_btagDeepFlavC', ['logit']),
        ('Jet_btagDeepFlavCvB', ['logit']),
        ('Jet_btagDeepFlavCvL', ['logit']),
        ('Jet_btagDeepFlavQG', ['logit'])

        # ('Jet_btagDeepFlavB', []),  # if using DeepJetConstraint the transformations have to be explicitly adapted in the DeepJetConstraint module
        # ('Jet_btagDeepFlavCvB', []),
        # ('Jet_btagDeepFlavCvL', []),
        # ('Jet_btagDeepFlavQG', [])

        # ('FatJet_mass', ['tanh200', 'logit']),
        # ('FatJet_msoftdrop', ['tanh200', 'logit']),
        # ('FatJet_tau1', ['logit']),
        # ('FatJet_tau2', ['logit']),
        # ('FatJet_tau3', ['logit']),
        # ('FatJet_tau4', ['logit'])
    ]
VARIABLES = [('Rec' + v[0] + '_CLASS', v[1]) for v in variables]

if is_fatjet_delphes:
    spectators = [
        'RecFatJet_PT_CLASS',
        'RecFatJet_Eta_CLASS',
    ]
else:
    spectators = [
        'EventID',

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
        'RecJet_hadronFlavour_CLASS',
        'RecJet_partonFlavour_CLASS',
        'RecJet_jetId_CLASS',

        'RecJet_response_CLASS',

        'RecJet_event_nJet_CLASS',
        'RecJet_event_PV_npvsGood_CLASS',
        'RecJet_event_MET_pt_CLASS',

        'RecJet_nearest_dR_CLASS',
        'RecJet_nearest_pt_CLASS',

        # 'RecFatJet_pt_CLASS',
        # 'RecFatJet_eta_CLASS',
    ]
excludespecs = [var[0] for var in PARAMETERS + VARIABLES]
SPECTATORS = [s for s in spectators if s not in excludespecs
              and s.replace('CLASS', 'FastSim') not in excludespecs
              and s.replace('CLASS', 'FullSim') not in excludespecs]


deepjetindicesWithParameters = [idx for idx, name in enumerate(PARAMETERS + VARIABLES) if 'Jet_btagDeepFlav' in name[0]]
deepjetindicesWithoutParameters = [idx for idx, name in enumerate(VARIABLES) if 'Jet_btagDeepFlav' in name[0]]

logitmaskWithParameters = [int('logit' in name[1]) for name in PARAMETERS + VARIABLES]
logitmaskWithoutParameters = [int('logit' in name[1]) for name in VARIABLES]

tanh200maskWithParameters = [int('tanh200' in name[1]) for name in PARAMETERS + VARIABLES]
tanh200maskWithoutParameters = [int('tanh200' in name[1]) for name in VARIABLES]

realLenParameters = len(PARAMETERS)  # without artificial increase due to one-hot-encoding

onehotencodeWithParamIdx = []
for iohe, ohe in enumerate(onehotencode):

    if iohe > 0:
        raise NotImplementedError('can only interpret one one-hot-encoding')

    if ohe[0] in [p[0] for p in PARAMETERS]:
        for ip, p in enumerate(PARAMETERS):
            if p[0] == ohe[0]:

                for ival, val in enumerate(ohe[1]):
                    if ival == 0:
                        PARAMETERS.append((ohe[0], PARAMETERS[ip][1]))
                    else:
                        PARAMETERS.append((ohe[0] + '_ohe_' + str(ival), PARAMETERS[ip][1]))

                PARAMETERS.pop(ip)

                onehotencodeWithParamIdx.append((ohe[0], ohe[1], ip))

                break
    elif ohe[0] in [v[0] for v in VARIABLES]:
        raise NotImplementedError('don\'t want to interpret one-hot-encoding: ', ohe)
    else:
        raise NotImplementedError('can\'t interpret one-hot-encoding: ', ohe)

# TODO: also skip parameters?
# has to be either all variables or all parameters + all variables
# skipindices = [idx for idx in range(len(PARAMETERS), len(PARAMETERS + VARIABLES))]  # what to skip in skip connections (internal)
skipindices = [idx for idx in range(len(PARAMETERS + VARIABLES))]  # what to skip in skip connections (internal)


if is_verbose:
    print('gpu?', torch.cuda.is_available())
    print('versions')
    print('python', platform.python_version())
    print('torch', torch.__version__)
    print('np', np.__version__)
    print('pd', pd.__version__)
    print('mpl', mpl.__version__)
    print('sns', sns.__version__)


foutcsv = open('traininglog_regression_' + trainingID + '.csv', 'w')
csvwriter = csv.writer(foutcsv)
csvwriter.writerow(['iteration', 'mmd_fixsigma', 'mmd', 'mmd_response', 'mse_fast', 'mse_full', 'mae_fast', 'mae_full']
                   + ['lmbda_' + key for key in constraintsconfig]
                   + ['mmd_hadflav0', 'mmd_hadflav4', 'mmd_hadflav5', 'l2dist_hadflav0', 'l2dist_hadflav4', 'l2dist_hadflav5'])  # TODO: this can be wrong if more than one constraint

foutcsv_val = open('traininglog_regression_val_' + trainingID + '.csv', 'w')
csvwriter_val = csv.writer(foutcsv_val)
csvwriter_val.writerow(['iteration', 'mmd_fixsigma', 'mmd', 'mmd_response', 'mse_fast', 'mse_full', 'mae_fast', 'mae_full']
                       + ['lmbda_' + key for key in constraintsconfig])


'''
###############################################################################################
# build model
###############################################################################################
'''
print('\n###build model')

if transformdeepjet5to4:

    tanh200maskWithParameters = tanh200maskWithParameters[:-1]
    logitmaskWithParameters = logitmaskWithParameters[:-1]
    tanh200maskWithoutParameters = tanh200maskWithoutParameters[:-1]
    logitmaskWithoutParameters = logitmaskWithoutParameters[:-1]
    skipindices = skipindices[:-1]

if usedensesequential:
    if transformdeepjet5to4:
        raise NotImplementedError('DenseSequential not implemented for transformdeepjet5to4')
    model = DenseSequential(n_params=len(PARAMETERS), n_vars=len(VARIABLES), skipindices=skipindices,
                            in_features_list=[len(VARIABLES) + len(PARAMETERS)] + nodes_hidden_layer_list,
                            out_features_list=nodes_hidden_layer_list + [len(VARIABLES)])
else:
    model = nn.Sequential()

if transformdeepjet5to4:
    model.add_module('DeepJetTransform5to4', DeepJetTransform5to4(deepjetindices=deepjetindicesWithParameters))
elif transformdeepjet4to4:
    model.add_module('DeepJetTransform4to4fromNano', DeepJetTransform4to4fromNano(deepjetindices=deepjetindicesWithParameters))

if any(tanh200maskWithParameters): model.add_module('Tanh200Transform', TanhTransform(mask=tanh200maskWithParameters, norm=200))
if any(logitmaskWithParameters): model.add_module('LogitTransform', LogitTransform(mask=logitmaskWithParameters, factor=logitfactor, onnxcompatible=onnxcompatible))

for ohe in onehotencodeWithParamIdx:
    model.add_module('OneHotEncode_' + ohe[0], OneHotEncode(source_idx=ohe[2], target_vals=ohe[1]))

if transformdeepjet5to4:
    if useskipconnections:
        model.add_module('LinearWithSkipConnection_0', LinearWithSkipConnection(in_features=len(VARIABLES) + len(PARAMETERS) - 1,
                                                                                out_features=nodes_hidden_layer_list[0] if num_hidden_layers > 1 else len(VARIABLES)-1,
                                                                                hidden_features=None if num_hidden_layers > 1 else nodes_hidden_layer_list[0],
                                                                                n_params=len(PARAMETERS),
                                                                                n_vars=len(VARIABLES),
                                                                                skipindices=skipindices,
                                                                                nskiplayers=num_layers_per_skipblock,
                                                                                dropout=dropout,
                                                                                noskipping=usedensesequential,
                                                                                isfirst=True,
                                                                                islast=num_hidden_layers == 1))
    else:
        model.add_module('Linear_0', nn.Linear(in_features=len(VARIABLES) + len(PARAMETERS) - 1, out_features=nodes_hidden_layer_list[0] if num_hidden_layers > 1 else len(VARIABLES)-1))
else:
    if useskipconnections:
        model.add_module('LinearWithSkipConnection_0', LinearWithSkipConnection(in_features=len(VARIABLES) + len(PARAMETERS),
                                                                                out_features=nodes_hidden_layer_list[0] if num_hidden_layers > 1 else len(VARIABLES),
                                                                                hidden_features=None if num_hidden_layers > 1 else nodes_hidden_layer_list[0],
                                                                                n_params=len(PARAMETERS),
                                                                                n_vars=len(VARIABLES),
                                                                                skipindices=skipindices,
                                                                                nskiplayers=num_layers_per_skipblock,
                                                                                dropout=dropout,
                                                                                noskipping=usedensesequential,
                                                                                isfirst=True,
                                                                                islast=num_hidden_layers == 1))
    else:
        model.add_module('Linear_0', nn.Linear(in_features=len(VARIABLES) + len(PARAMETERS), out_features=nodes_hidden_layer_list[0] if num_hidden_layers > 1 else len(VARIABLES)))


if num_hidden_layers == 1 and adddeepjetconstraint:
    if transformdeepjet5to4:
        model.add_module('DeepJetConstraint4', DeepJetConstraint4(deepjetindices=deepjetindicesWithoutParameters, applylogittransform=any(logitmaskWithoutParameters), logittransformfactor=logitfactor, skipone=True, onnxcompatible=onnxcompatible))
    elif transformdeepjet4to4:
        model.add_module('DeepJetConstraint4', DeepJetConstraint4(deepjetindices=deepjetindicesWithoutParameters, applylogittransform=any(logitmaskWithoutParameters), logittransformfactor=logitfactor, skipone=False, onnxcompatible=onnxcompatible))
    else:
        model.add_module('DeepJetConstraint4', DeepJetConstraint4(deepjetindices=deepjetindicesWithoutParameters, applylogittransform=any(logitmaskWithoutParameters), logittransformfactor=logitfactor, skipone=False, onnxcompatible=onnxcompatible, nanotransform=not transformdeepjet4to4))
        # model.add_module('DeepJetConstraint', DeepJetConstraint(deepjetindices=deepjetindicesWithoutParameters, applylogittransform=any(logitmaskWithoutParameters), logittransformfactor=logitfactor))


for k in range(num_hidden_layers-1):

    if not useskipconnections:
        model.add_module('LeakyReLU_' + str(k+1), nn.LeakyReLU())
        if dropout: model.add_module('Dropout_' + str(k+1), nn.Dropout(dropout))

    if k == num_hidden_layers-2:
        if adddeepjetconstraint:
            if transformdeepjet5to4:
                if useskipconnections:
                    model.add_module('LinearWithSkipConnection_last', LinearWithSkipConnection(in_features=nodes_hidden_layer_list[k],
                                                                                               out_features=len(VARIABLES)-1,
                                                                                               n_params=len(PARAMETERS),
                                                                                               n_vars=len(VARIABLES),
                                                                                               skipindices=skipindices,
                                                                                               nskiplayers=num_layers_per_skipblock,
                                                                                               dropout=dropout,
                                                                                               noskipping=usedensesequential,
                                                                                               islast=True))
                else:
                    model.add_module('Linear_last', nn.Linear(in_features=nodes_hidden_layer_list[k], out_features=len(VARIABLES)-1))
                model.add_module('DeepJetConstraint4', DeepJetConstraint4(deepjetindices=deepjetindicesWithoutParameters, applylogittransform=any(logitmaskWithoutParameters), logittransformfactor=logitfactor, skipone=True, onnxcompatible=onnxcompatible))
            elif transformdeepjet4to4:
                if useskipconnections:
                    model.add_module('LinearWithSkipConnection_last', LinearWithSkipConnection(in_features=nodes_hidden_layer_list[k],
                                                                                               out_features=len(VARIABLES),
                                                                                               n_params=len(PARAMETERS),
                                                                                               n_vars=len(VARIABLES),
                                                                                               skipindices=skipindices,
                                                                                               nskiplayers=num_layers_per_skipblock,
                                                                                               dropout=dropout,
                                                                                               noskipping=usedensesequential,
                                                                                               islast=True))
                else:
                    model.add_module('Linear_last', nn.Linear(in_features=nodes_hidden_layer_list[k], out_features=len(VARIABLES)))
                model.add_module('DeepJetConstraint4', DeepJetConstraint4(deepjetindices=deepjetindicesWithoutParameters, applylogittransform=any(logitmaskWithoutParameters), logittransformfactor=logitfactor, skipone=False, onnxcompatible=onnxcompatible))
            else:
                if useskipconnections:
                    model.add_module('LinearWithSkipConnection_last', LinearWithSkipConnection(in_features=nodes_hidden_layer_list[k],
                                                                                               out_features=len(VARIABLES),
                                                                                               n_params=len(PARAMETERS),
                                                                                               n_vars=len(VARIABLES),
                                                                                               skipindices=skipindices,
                                                                                               nskiplayers=num_layers_per_skipblock,
                                                                                               dropout=dropout,
                                                                                               noskipping=usedensesequential,
                                                                                               islast=True))  # TODO: out_features=len(VARIABLES)-1 correct?
                else:
                    model.add_module('Linear_last', nn.Linear(in_features=nodes_hidden_layer_list[k], out_features=len(VARIABLES)))  # TODO: out_features=len(VARIABLES)-1 correct?
                model.add_module('DeepJetConstraint4', DeepJetConstraint4(deepjetindices=deepjetindicesWithoutParameters, applylogittransform=any(logitmaskWithoutParameters), logittransformfactor=logitfactor, skipone=False, onnxcompatible=onnxcompatible, nanotransform=not transformdeepjet4to4))
                # model.add_module('DeepJetConstraint', DeepJetConstraint(deepjetindices=deepjetindicesWithoutParameters, applylogittransform=any(logitmaskWithoutParameters), logittransformfactor=logitfactor))
        else:
            if useskipconnections:
                model.add_module('LinearWithSkipConnection_last', LinearWithSkipConnection(in_features=nodes_hidden_layer_list[k],
                                                                                           out_features=len(VARIABLES),
                                                                                           n_params=len(PARAMETERS),
                                                                                           n_vars=len(VARIABLES),
                                                                                           skipindices=skipindices,
                                                                                           nskiplayers=num_layers_per_skipblock,
                                                                                           dropout=dropout,
                                                                                           noskipping=usedensesequential,
                                                                                           islast=True))
            else:
                model.add_module('Linear_last', nn.Linear(in_features=nodes_hidden_layer_list[k], out_features=len(VARIABLES)))

    else:
        if useskipconnections:
            model.add_module('LinearWithSkipConnection_' + str(k+1), LinearWithSkipConnection(in_features=nodes_hidden_layer_list[k],
                                                                                              out_features=nodes_hidden_layer_list[k+1],
                                                                                              n_params=len(PARAMETERS),
                                                                                              n_vars=len(VARIABLES),
                                                                                              skipindices=skipindices,
                                                                                              nskiplayers=num_layers_per_skipblock,
                                                                                              dropout=dropout,
                                                                                              noskipping=usedensesequential))
        else:
            model.add_module('Linear_' + str(k+1), nn.Linear(in_features=nodes_hidden_layer_list[k], out_features=nodes_hidden_layer_list[k+1]))

if any(logitmaskWithoutParameters): model.add_module('LogitTransformBack', LogitTransformBack(mask=logitmaskWithoutParameters, factor=logitfactor))
if any(tanh200maskWithoutParameters): model.add_module('Tanh200TransformBack', TanhTransformBack(mask=tanh200maskWithoutParameters, norm=200))

if transformdeepjet5to4:
    model.add_module('DeepJetTransform4to5', DeepJetTransform4to5(deepjetindices=deepjetindicesWithoutParameters))
elif transformdeepjet4to4:
    model.add_module('DeepJetTransform4to4toNano', DeepJetTransform4to4toNano(deepjetindices=deepjetindicesWithoutParameters))

if castto16bit:
    model.add_module('CastTo16Bit', CastTo16Bit())

model.to(device)

print(model)
print(sum(p.numel() for p in model.parameters() if p.requires_grad), 'trainable parameters')

'''
###############################################################################################
# initialize loss/trainer
###############################################################################################
'''
print('\n###initialize loss/trainer')

mmd_loss_fixsigma = my_mmd.MMD_loss(kernel_mul=10., kernel_num=4, fix_sigma=100., one_sided_bandwidth=True)
# mmd_loss_fixsigma = my_mmd.MMD_loss(kernel_mul=2., kernel_num=5, fix_sigma=2.)
# mmd_loss = my_mmd.MMD_loss(kernel_mul=1., kernel_num=1, one_sided_bandwidth=False, calculate_sigma_without_parameters=True)
mmd_loss = my_mmd.MMD_loss(kernel_mul=1., kernel_num=1, fix_sigma=1., one_sided_bandwidth=False)
# mmd_loss = my_mmd.MMD_loss(kernel_mul=1., kernel_num=1, one_sided_bandwidth=False)
# mmd_loss = lambda a, b: my_mmd.MMD(a, b, kernel='rbf')
mse_loss = nn.MSELoss()
mae_loss = nn.HuberLoss(delta=0.1)  # nn.L1Loss()  #


constraints = []
if usemdmm:

    mmd_scale = 1.
    mmd_response_scale = 0.
    mse_fast_scale = 0.
    mse_full_scale = 0.
    mae_fast_scale = 0.
    mae_full_scale = 0.

    if 'mmd' in constraintsconfig:
        if includeparamsinmmd:
            if len(onehotencodeWithParamIdx) > 0:  # i.e. == 1
                constraints.append(mdmm.EqConstraint(lambda X, y, output: mmd_loss(output, y, parameters=OneHotEncode(
                    source_idx=onehotencodeWithParamIdx[0][2], target_vals=onehotencodeWithParamIdx[0][1]).forward(X)[:, :len(PARAMETERS)]), constraintsconfig['mmd'][0], scale=constraintsconfig['mmd'][1]))
            else:
                constraints.append(mdmm.EqConstraint(lambda X, y, output: mmd_loss(output, y, parameters=X[:, :realLenParameters]), constraintsconfig['mmd'][0], scale=constraintsconfig['mmd'][1]))
        else:
            constraints.append(mdmm.EqConstraint(lambda X, y, output: mmd_loss(output, y), constraintsconfig['mmd'][0], scale=constraintsconfig['mmd'][1]))
    if 'mmd_response' in constraintsconfig:
        constraints.append(mdmm.EqConstraint(lambda X, y, output: mmd_loss_fixsigma(output[:, :1] / X[:, :1], y[:, :1] / X[:, :1]), constraintsconfig['mmd_response'][0], scale=constraintsconfig['mmd_response'][1]))
        mmd_response_scale = constraintsconfig['mmd_response'][1]
    if 'mse_fast' in constraintsconfig:
        constraints.append(mdmm.EqConstraint(lambda X, y, output: mse_loss(output, X[:, realLenParameters:]), constraintsconfig['mse_fast'][0], scale=constraintsconfig['mse_fast'][1]))
        mse_fast_scale = constraintsconfig['mse_fast'][1]
    if 'mse_full' in constraintsconfig:
        constraints.append(mdmm.EqConstraint(lambda X, y, output: mse_loss(output, y), constraintsconfig['mse_full'][0], scale=constraintsconfig['mse_full'][1]))
        mse_full_scale = constraintsconfig['mse_full'][1]
    if 'mae_fast' in constraintsconfig:
        constraints.append(mdmm.EqConstraint(lambda X, y, output: mae_loss(output, X[:, realLenParameters:]), constraintsconfig['mae_fast'][0], scale=constraintsconfig['mae_fast'][1]))
        mae_fast_scale = constraintsconfig['mae_fast'][1]
    if 'mae_full' in constraintsconfig:
        # TODO: what constraint?
        constraints.append(mdmm.EqConstraint(lambda X, y, output: mae_loss(output, y), constraintsconfig['mae_full'][0], scale=constraintsconfig['mae_full'][1]))
        # constraints.append(mdmm.MaxConstraint(lambda X, y, output: mae_loss(output, y), constraintsconfig['mae_full'][0], scale=constraintsconfig['mae_full'][1]))
        mae_full_scale = constraintsconfig['mae_full'][1]

    optimizer = mdmm.MDMM(constraints)  # mdmm_module
    trainer = optimizer.make_optimizer(model.parameters(), optimizer=torch.optim.Adam, lr=lr)

else:

    mmd_scale = 1.  # 5.
    mmd_response_scale = 0.  # 5.
    mse_fast_scale = 0.  # 1.
    mse_full_scale = 0.  # 1.  # 0.1  # 0.05
    mae_fast_scale = 0.
    mae_full_scale = 0.  # 1.  #

    optimizer = torch.optim.Adam
    trainer = optimizer(model.parameters(), lr=lr)

print('loss scales', mmd_scale, mmd_response_scale, mse_fast_scale, mse_full_scale, mae_fast_scale, mae_full_scale)

if cyclic_lr:

    lrscheduler = torch.optim.lr_scheduler.CyclicLR(trainer, base_lr=1e-5, max_lr=1e-4, step_size_up=2000,
                                                    step_size_down=None, mode='triangular2', gamma=1.0,
                                                    scale_fn=None, scale_mode='cycle', cycle_momentum=False,
                                                    base_momentum=0.8, max_momentum=0.9, last_epoch=- 1,
                                                    verbose=False)

else:

    if lr_scheduler_gamma == 0:

        lrscheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(trainer, mode='min', factor=0.5, patience=10, threshold=0.0001,
                                                                 threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08,
                                                                 verbose=True)
    else:

        lrscheduler = torch.optim.lr_scheduler.ExponentialLR(trainer, gamma=lr_scheduler_gamma, last_epoch=-1, verbose=False)

'''
###############################################################################################
# get training data
###############################################################################################
'''
print('\n###get training data')

if ntest == 0:
    rdf = ROOT.RDataFrame(treename, in_path).Filter(preselection)
else:
    rdf = ROOT.RDataFrame(treename, in_path).Filter(preselection).Range(ntrain + nval + ntest)

dict_X = rdf.AsNumpy([n[0].replace('CLASS', 'FastSim') for n in PARAMETERS + VARIABLES if '_ohe_' not in n[0]])
dict_y = rdf.AsNumpy([n[0].replace('CLASS', 'FullSim') for n in VARIABLES])
dict_spec = rdf.AsNumpy([n for n in SPECTATORS if 'CLASS' not in n]
                        + [n.replace('CLASS', 'FastSim') for n in SPECTATORS if 'CLASS' in n]
                        + [n.replace('CLASS', 'FullSim') for n in SPECTATORS if 'CLASS' in n])

my_dtype = torch.float32
data_X = torch.tensor(np.stack([dict_X[var] for var in dict_X], axis=1), dtype=my_dtype, device=device)
data_y = torch.tensor(np.stack([dict_y[var] for var in dict_y], axis=1), dtype=my_dtype, device=device)
data_spec = torch.tensor(np.stack([dict_spec[var] for var in dict_spec], axis=1), dtype=my_dtype, device=device)

dataset = torch.utils.data.TensorDataset(data_X, data_y, data_spec)

if ntest == 0:
    ntest = len(dataset) - (ntrain + nval)

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [ntrain, nval, ntest])


def collate_fn(batch):
    batch = list(filter(lambda x: torch.all(torch.isfinite(torch.cat(x))), batch))
    return torch.utils.data.dataloader.default_collate(batch)


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=my_batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=my_batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=my_batch_size, shuffle=False, collate_fn=collate_fn)

if individualmmdshadflav:
    len_data_X = len(data_X)
    hadflav_fraction_0 = len(data_X[data_X[:, 2] == 0]) / len_data_X
    hadflav_fraction_4 = len(data_X[data_X[:, 2] == 4]) / len_data_X
    hadflav_fraction_5 = len(data_X[data_X[:, 2] == 5]) / len_data_X
    print('had flav fractions', hadflav_fraction_0, hadflav_fraction_4, hadflav_fraction_5)

    # hadflav_fraction_0 = np.sqrt(hadflav_fraction_0)
    # hadflav_fraction_4 = np.sqrt(hadflav_fraction_4)
    # hadflav_fraction_5 = np.sqrt(hadflav_fraction_5)

    if normalizeindividualmmdshadflavtostart:
        hadflav_fraction_0 = 1.
        hadflav_fraction_4 = 1.
        hadflav_fraction_5 = 1.


'''
###############################################################################################
# training loop
###############################################################################################
'''
print('\n###training loop')

best_val = float('inf')
benchmarklosses = []
benchmarklosses_val = []
iteration = 0
for epoch in range(my_num_epochs):

    model.train()

    if epoch == 0:

        starting_point_mmd_fixsigma = 0.0
        starting_point_mmd = 0.0
        starting_point_mmd_response = 0.0
        starting_point_mse_fast = 0.0
        starting_point_mse_full = 0.0
        starting_point_mae_fast = 0.0
        starting_point_mae_full = 0.0

        starting_point_mmd_hadflav0 = 0.0
        starting_point_mmd_hadflav4 = 0.0
        starting_point_mmd_hadflav5 = 0.0
        starting_point_l2dist_hadflav0 = 0.0
        starting_point_l2dist_hadflav4 = 0.0
        starting_point_l2dist_hadflav5 = 0.0

        for i, (X, y, _) in enumerate(train_loader):

            if is_verbose:
                print('start')
                print('X original')
                print(X)
                print('y original')
                print(y)

            output = model(X)

            if transformdeepjet5to4:
                X = DeepJetTransform5to4(deepjetindices=deepjetindicesWithParameters).forward(X)
                y = DeepJetTransform5to4(deepjetindices=deepjetindicesWithoutParameters).forward(y)
            elif transformdeepjet4to4:
                X = DeepJetTransform4to4fromNano(deepjetindices=deepjetindicesWithParameters).forward(X)
                y = DeepJetTransform4to4fromNano(deepjetindices=deepjetindicesWithoutParameters).forward(y)
            if any(tanh200maskWithParameters): X = TanhTransform(mask=tanh200maskWithParameters, norm=200).forward(X)
            if any(tanh200maskWithoutParameters): y = TanhTransform(mask=tanh200maskWithoutParameters, norm=200).forward(y)
            # TODO: uncomment??
            # if any(logitmaskWithParameters): X = LogitTransform(mask=logitmaskWithParameters, factor=logitfactor, onnxcompatible=False).forward(X)
            # if any(logitmaskWithoutParameters): y = LogitTransform(mask=logitmaskWithoutParameters, factor=logitfactor, onnxcompatible=False).forward(y)

            if is_verbose:
                print('X')
                print(X)
                print('y')
                print(y)
                print('output')
                print(output)

            if individualmmdshadflav:
                if includeparamsinmmd:
                    if len(onehotencodeWithParamIdx) > 0:  # i.e. == 1

                        starting_point_mmd_fixsigma += hadflav_fraction_0**2 * mmd_loss_fixsigma(output, y, parameters=OneHotEncode(
                            source_idx=onehotencodeWithParamIdx[0][2], target_vals=onehotencodeWithParamIdx[0][1]).forward(X)[:, :len(PARAMETERS)],
                                                                         mask=X[:, 2] == 0).item()
                        starting_point_mmd_fixsigma += hadflav_fraction_4**2 * mmd_loss_fixsigma(output, y, parameters=OneHotEncode(
                            source_idx=onehotencodeWithParamIdx[0][2], target_vals=onehotencodeWithParamIdx[0][1]).forward(X)[:, :len(PARAMETERS)],
                                                                         mask=X[:, 2] == 4).item()
                        starting_point_mmd_fixsigma += hadflav_fraction_5**2 * mmd_loss_fixsigma(output, y, parameters=OneHotEncode(
                            source_idx=onehotencodeWithParamIdx[0][2], target_vals=onehotencodeWithParamIdx[0][1]).forward(X)[:, :len(PARAMETERS)],
                                                                         mask=X[:, 2] == 5).item()

                        # print(mmd_loss_fixsigma(output, y, parameters=OneHotEncode(
                        #     source_idx=onehotencodeWithParamIdx[0][2], target_vals=onehotencodeWithParamIdx[0][1]).forward(X)[:, :len(PARAMETERS)],
                        #                                                  mask=X[:, 2] == 0).item() * hadflav_fraction_0)
                        # print(mmd_loss_fixsigma(output, y, parameters=OneHotEncode(
                        #     source_idx=onehotencodeWithParamIdx[0][2], target_vals=onehotencodeWithParamIdx[0][1]).forward(X)[:, :len(PARAMETERS)],
                        #                                                  mask=X[:, 2] == 4).item() * hadflav_fraction_4)
                        # print(mmd_loss_fixsigma(output, y, parameters=OneHotEncode(
                        #     source_idx=onehotencodeWithParamIdx[0][2], target_vals=onehotencodeWithParamIdx[0][1]).forward(X)[:, :len(PARAMETERS)],
                        #                                                  mask=X[:, 2] == 5).item() * hadflav_fraction_5)
                        # print(starting_point_mmd_fixsigma)
                        # print(mmd_loss_fixsigma(output, y, parameters=OneHotEncode(
                        #     source_idx=onehotencodeWithParamIdx[0][2], target_vals=onehotencodeWithParamIdx[0][1]).forward(X)[:, :len(PARAMETERS)]).item())
                        # print(mmd_loss_fixsigma(output, y, parameters=OneHotEncode(
                        #     source_idx=onehotencodeWithParamIdx[0][2], target_vals=onehotencodeWithParamIdx[0][1], eps=10000000).forward(X)[:, :len(PARAMETERS)]).item())

                        starting_point_mmd += hadflav_fraction_0**2 * mmd_loss(output, y, parameters=OneHotEncode(
                            source_idx=onehotencodeWithParamIdx[0][2], target_vals=onehotencodeWithParamIdx[0][1]).forward(X)[:, :len(PARAMETERS)],
                                                       mask=X[:, 2] == 0).item()
                        starting_point_mmd += hadflav_fraction_4**2 * mmd_loss(output, y, parameters=OneHotEncode(
                            source_idx=onehotencodeWithParamIdx[0][2], target_vals=onehotencodeWithParamIdx[0][1]).forward(X)[:, :len(PARAMETERS)],
                                                       mask=X[:, 2] == 4).item()
                        starting_point_mmd += hadflav_fraction_5**2 * mmd_loss(output, y, parameters=OneHotEncode(
                            source_idx=onehotencodeWithParamIdx[0][2], target_vals=onehotencodeWithParamIdx[0][1]).forward(X)[:, :len(PARAMETERS)],
                                                       mask=X[:, 2] == 5).item()

                        starting_point_mmd_hadflav0 += hadflav_fraction_0**2 * mmd_loss(output, y, parameters=OneHotEncode(source_idx=onehotencodeWithParamIdx[0][2], target_vals=onehotencodeWithParamIdx[0][1]).forward(X)[:, :len(PARAMETERS)],
                                                               mask=X[:, 2] == 0, l2dist_out='starting_point_l2dist_hadflav0').item()
                        starting_point_mmd_hadflav4 += hadflav_fraction_4**2 * mmd_loss(output, y, parameters=OneHotEncode(source_idx=onehotencodeWithParamIdx[0][2], target_vals=onehotencodeWithParamIdx[0][1]).forward(X)[:, :len(PARAMETERS)],
                                                               mask=X[:, 2] == 4, l2dist_out='starting_point_l2dist_hadflav4').item()
                        starting_point_mmd_hadflav5 += hadflav_fraction_5**2 * mmd_loss(output, y, parameters=OneHotEncode(source_idx=onehotencodeWithParamIdx[0][2], target_vals=onehotencodeWithParamIdx[0][1]).forward(X)[:, :len(PARAMETERS)],
                                                               mask=X[:, 2] == 5, l2dist_out='starting_point_l2dist_hadflav5').item()
                        starting_point_l2dist_hadflav0 += my_mmd.starting_point_l2dist_hadflav0
                        starting_point_l2dist_hadflav4 += my_mmd.starting_point_l2dist_hadflav4
                        starting_point_l2dist_hadflav5 += my_mmd.starting_point_l2dist_hadflav5

                    else:
                        starting_point_mmd_fixsigma += hadflav_fraction_0**2 * mmd_loss_fixsigma(output, y, parameters=X[:, :realLenParameters], mask=X[:, 2] == 0).item()
                        starting_point_mmd_fixsigma += hadflav_fraction_4**2 * mmd_loss_fixsigma(output, y, parameters=X[:, :realLenParameters], mask=X[:, 2] == 4).item()
                        starting_point_mmd_fixsigma += hadflav_fraction_5**2 * mmd_loss_fixsigma(output, y, parameters=X[:, :realLenParameters], mask=X[:, 2] == 5).item()

                        starting_point_mmd += hadflav_fraction_0**2 * mmd_loss(output, y, parameters=X[:, :realLenParameters], mask=X[:, 2] == 0).item()
                        starting_point_mmd += hadflav_fraction_4**2 * mmd_loss(output, y, parameters=X[:, :realLenParameters], mask=X[:, 2] == 4).item()
                        starting_point_mmd += hadflav_fraction_5**2 * mmd_loss(output, y, parameters=X[:, :realLenParameters], mask=X[:, 2] == 5).item()
                else:

                    starting_point_mmd_fixsigma += hadflav_fraction_0**2 * mmd_loss_fixsigma(output, y, mask=X[:, 2] == 0).item()
                    starting_point_mmd_fixsigma += hadflav_fraction_4**2 * mmd_loss_fixsigma(output, y, mask=X[:, 2] == 4).item()
                    starting_point_mmd_fixsigma += hadflav_fraction_5**2 * mmd_loss_fixsigma(output, y, mask=X[:, 2] == 5).item()

                    starting_point_mmd += hadflav_fraction_0**2 * mmd_loss(output, y, mask=X[:, 2] == 0).item()
                    starting_point_mmd += hadflav_fraction_4**2 * mmd_loss(output, y, mask=X[:, 2] == 4).item()
                    starting_point_mmd += hadflav_fraction_5**2 * mmd_loss(output, y, mask=X[:, 2] == 5).item()
            else:
                if includeparamsinmmd:
                    if len(onehotencodeWithParamIdx) > 0:  # i.e. == 1
                        starting_point_mmd_fixsigma += mmd_loss_fixsigma(output, y, parameters=OneHotEncode(
                            source_idx=onehotencodeWithParamIdx[0][2], target_vals=onehotencodeWithParamIdx[0][1]).forward(X)[:, :len(PARAMETERS)]).item()
                        starting_point_mmd += mmd_loss(output, y, parameters=OneHotEncode(
                            source_idx=onehotencodeWithParamIdx[0][2], target_vals=onehotencodeWithParamIdx[0][1]).forward(X)[:, :len(PARAMETERS)]).item()
                    else:
                        starting_point_mmd_fixsigma += mmd_loss_fixsigma(output, y, parameters=X[:, :realLenParameters]).item()
                        starting_point_mmd += mmd_loss(output, y, parameters=X[:, :realLenParameters]).item()
                else:
                    starting_point_mmd_fixsigma += mmd_loss_fixsigma(output, y).item()
                    starting_point_mmd += mmd_loss(output, y).item()


            # starting_point_mmd_response += mmd_loss_fixsigma(output[:, :1] / X[:, :1], y[:, :1] / X[:, :1]).item()
            starting_point_mmd_response += torch.zeros(1).item()
            starting_point_mse_fast += mse_loss(output, X[:, realLenParameters:]).item()
            starting_point_mse_full += mse_loss(output, y).item()
            starting_point_mae_fast += mae_loss(output, X[:, realLenParameters:]).item()
            starting_point_mae_full += mae_loss(output, y).item()

            # initialize with dummy values because don't want to implement for all if/else cases...
            bm_mmd_hadflav0, bm_mmd_hadflav4, bm_mmd_hadflav5, bm_l2dist_hadflav0, bm_l2dist_hadflav4, bm_l2dist_hadflav5 = 0, 0, 0, 0, 0, 0
            if individualmmdshadflav:
                if includeparamsinmmd:
                    if len(onehotencodeWithParamIdx) > 0:  # i.e. == 1
                        bm_mmd_fixsigma = hadflav_fraction_0**2 * mmd_loss_fixsigma(X[:, realLenParameters:], y, parameters=OneHotEncode(
                            source_idx=onehotencodeWithParamIdx[0][2], target_vals=onehotencodeWithParamIdx[0][1]).forward(X)[:, :len(PARAMETERS)],
                                                            mask=X[:, 2] == 0).item()
                        bm_mmd_fixsigma += hadflav_fraction_4**2 * mmd_loss_fixsigma(X[:, realLenParameters:], y, parameters=OneHotEncode(
                            source_idx=onehotencodeWithParamIdx[0][2], target_vals=onehotencodeWithParamIdx[0][1]).forward(X)[:, :len(PARAMETERS)],
                                                            mask=X[:, 2] == 4).item()
                        bm_mmd_fixsigma += hadflav_fraction_5**2 * mmd_loss_fixsigma(X[:, realLenParameters:], y, parameters=OneHotEncode(
                            source_idx=onehotencodeWithParamIdx[0][2], target_vals=onehotencodeWithParamIdx[0][1]).forward(X)[:, :len(PARAMETERS)],
                                                            mask=X[:, 2] == 5).item()

                        bm_mmd = hadflav_fraction_0**2 * mmd_loss(X[:, realLenParameters:], y, parameters=OneHotEncode(
                            source_idx=onehotencodeWithParamIdx[0][2], target_vals=onehotencodeWithParamIdx[0][1]).forward(X)[:, :len(PARAMETERS)],
                                                            mask=X[:, 2] == 0).item()
                        bm_mmd += hadflav_fraction_4**2 * mmd_loss(X[:, realLenParameters:], y, parameters=OneHotEncode(
                            source_idx=onehotencodeWithParamIdx[0][2], target_vals=onehotencodeWithParamIdx[0][1]).forward(X)[:, :len(PARAMETERS)],
                                                            mask=X[:, 2] == 4).item()
                        bm_mmd += hadflav_fraction_5**2 * mmd_loss(X[:, realLenParameters:], y, parameters=OneHotEncode(
                            source_idx=onehotencodeWithParamIdx[0][2], target_vals=onehotencodeWithParamIdx[0][1]).forward(X)[:, :len(PARAMETERS)],
                                                            mask=X[:, 2] == 5).item()

                        bm_mmd_hadflav0 = hadflav_fraction_0**2 * mmd_loss(X[:, realLenParameters:], y, parameters=OneHotEncode(source_idx=onehotencodeWithParamIdx[0][2], target_vals=onehotencodeWithParamIdx[0][1]).forward(X)[:, :len(PARAMETERS)],
                                                            mask=X[:, 2] == 0, l2dist_out='bm_l2dist_hadflav0').item()
                        bm_mmd_hadflav4 = hadflav_fraction_4**2 * mmd_loss(X[:, realLenParameters:], y, parameters=OneHotEncode(source_idx=onehotencodeWithParamIdx[0][2], target_vals=onehotencodeWithParamIdx[0][1]).forward(X)[:, :len(PARAMETERS)],
                                                            mask=X[:, 2] == 4, l2dist_out='bm_l2dist_hadflav4').item()
                        bm_mmd_hadflav5 = hadflav_fraction_5**2 * mmd_loss(X[:, realLenParameters:], y, parameters=OneHotEncode(source_idx=onehotencodeWithParamIdx[0][2], target_vals=onehotencodeWithParamIdx[0][1]).forward(X)[:, :len(PARAMETERS)],
                                                            mask=X[:, 2] == 5, l2dist_out='bm_l2dist_hadflav5').item()
                        bm_l2dist_hadflav0 = my_mmd.bm_l2dist_hadflav0
                        bm_l2dist_hadflav4 = my_mmd.bm_l2dist_hadflav4
                        bm_l2dist_hadflav5 = my_mmd.bm_l2dist_hadflav5

                    else:
                        bm_mmd_fixsigma = hadflav_fraction_0**2 * mmd_loss_fixsigma(X[:, realLenParameters:], y, parameters=X[:, :realLenParameters], mask=X[:, 2] == 0).item()
                        bm_mmd_fixsigma += hadflav_fraction_4**2 * mmd_loss_fixsigma(X[:, realLenParameters:], y, parameters=X[:, :realLenParameters], mask=X[:, 2] == 4).item()
                        bm_mmd_fixsigma += hadflav_fraction_5**2 * mmd_loss_fixsigma(X[:, realLenParameters:], y, parameters=X[:, :realLenParameters], mask=X[:, 2] == 5).item()

                        bm_mmd = hadflav_fraction_0**2 * mmd_loss(X[:, realLenParameters:], y, parameters=X[:, :realLenParameters], mask=X[:, 2] == 0).item()
                        bm_mmd += hadflav_fraction_4**2 * mmd_loss(X[:, realLenParameters:], y, parameters=X[:, :realLenParameters], mask=X[:, 2] == 4).item()
                        bm_mmd += hadflav_fraction_5**2 * mmd_loss(X[:, realLenParameters:], y, parameters=X[:, :realLenParameters], mask=X[:, 2] == 5).item()
                else:
                    bm_mmd_fixsigma = hadflav_fraction_0**2 * mmd_loss_fixsigma(X[:, realLenParameters:], y, mask=X[:, 2] == 0).item()
                    bm_mmd_fixsigma += hadflav_fraction_4**2 * mmd_loss_fixsigma(X[:, realLenParameters:], y, mask=X[:, 2] == 4).item()
                    bm_mmd_fixsigma += hadflav_fraction_5**2 * mmd_loss_fixsigma(X[:, realLenParameters:], y, mask=X[:, 2] == 5).item()

                    bm_mmd = hadflav_fraction_0**2 * mmd_loss(X[:, realLenParameters:], y, mask=X[:, 2] == 0).item()
                    bm_mmd += hadflav_fraction_4**2 * mmd_loss(X[:, realLenParameters:], y, mask=X[:, 2] == 4).item()
                    bm_mmd += hadflav_fraction_5**2 * mmd_loss(X[:, realLenParameters:], y, mask=X[:, 2] == 5).item()
            else:
                if includeparamsinmmd:
                    if len(onehotencodeWithParamIdx) > 0:  # i.e. == 1
                        bm_mmd_fixsigma = mmd_loss_fixsigma(X[:, realLenParameters:], y, parameters=OneHotEncode(
                            source_idx=onehotencodeWithParamIdx[0][2], target_vals=onehotencodeWithParamIdx[0][1]).forward(X)[:, :len(PARAMETERS)]).item()
                        bm_mmd = mmd_loss(X[:, realLenParameters:], y, parameters=OneHotEncode(
                            source_idx=onehotencodeWithParamIdx[0][2], target_vals=onehotencodeWithParamIdx[0][1]).forward(X)[:, :len(PARAMETERS)]).item()
                    else:
                        bm_mmd_fixsigma = mmd_loss_fixsigma(X[:, realLenParameters:], y, parameters=X[:, :realLenParameters]).item()
                        bm_mmd = mmd_loss(X[:, realLenParameters:], y, parameters=X[:, :realLenParameters]).item()
                else:
                    bm_mmd_fixsigma = mmd_loss_fixsigma(X[:, realLenParameters:], y).item()
                    bm_mmd = mmd_loss(X[:, realLenParameters:], y).item()
            # bm_mmd_response = mmd_loss_fixsigma(X[:, realLenParameters:realLenParameters + 1] / X[:, :1], y[:, :1] / X[:, :1]).item()
            bm_mmd_response = torch.zeros(1).item()
            bm_mse_fast = mse_loss(X[:, realLenParameters:], X[:, realLenParameters:]).item()  # should be 0
            bm_mse_full = mse_loss(X[:, realLenParameters:], y).item()
            # bm_mse_full = mse_full_scale * mse_loss(X[:, realLenParameters + 1:], y[:, 1:]).item()
            bm_mae_fast = mae_loss(X[:, realLenParameters:], X[:, realLenParameters:]).item()  # should be 0
            bm_mae_full = mae_loss(X[:, realLenParameters:], y).item()

            if mmd_response_scale > 0 or mse_fast_scale > 0 or mse_full_scale > 0 or mae_fast_scale > 0 or mae_full_scale > 0:  # sometimes NaN error
                benchmarklosses.append((mmd_scale * bm_mmd + mmd_response_scale * bm_mmd_response + mse_fast_scale * bm_mse_fast + mse_full_scale * bm_mse_full + mae_fast_scale * bm_mae_fast + mae_full_scale * bm_mae_full,
                                        bm_mmd_fixsigma, bm_mmd, bm_mmd_response, bm_mse_fast, bm_mse_full, bm_mae_fast, bm_mae_full,
                                        bm_mmd_hadflav0, bm_mmd_hadflav4, bm_mmd_hadflav5, bm_l2dist_hadflav0, bm_l2dist_hadflav4, bm_l2dist_hadflav5))
            else:
                benchmarklosses.append((mmd_scale * bm_mmd,
                                        bm_mmd_fixsigma, bm_mmd, bm_mmd_response, bm_mse_fast, bm_mse_full, bm_mae_fast, bm_mae_full,
                                        bm_mmd_hadflav0, bm_mmd_hadflav4, bm_mmd_hadflav5, bm_l2dist_hadflav0, bm_l2dist_hadflav4, bm_l2dist_hadflav5))

        benchmarklosses = np.array(benchmarklosses)
        print('[{}] benchmark loss: {:.10f}'.format(epoch + 1, np.mean(benchmarklosses[:, 0])))
        print('bm_mmd_fixsigma, bm_mmd, bm_mmd_response, bm_mse_fast, bm_mse_full, bm_mae_fast, bm_mae_full, bm_mmd_hadflav0, bm_mmd_hadflav4, bm_mmd_hadflav5, bm_l2dist_hadflav0, bm_l2dist_hadflav4, bm_l2dist_hadflav5',
              np.mean(benchmarklosses[:, 1]), np.mean(benchmarklosses[:, 2]),
              np.mean(benchmarklosses[:, 3]), np.mean(benchmarklosses[:, 4]),
              np.mean(benchmarklosses[:, 5]), np.mean(benchmarklosses[:, 6]),
              np.mean(benchmarklosses[:, 7]), np.mean(benchmarklosses[:, 8]),
              np.mean(benchmarklosses[:, 9]), np.mean(benchmarklosses[:, 10]),
              np.mean(benchmarklosses[:, 11]), np.mean(benchmarklosses[:, 12]),
              np.mean(benchmarklosses[:, 13]))

        csvwriter.writerow(['benchmark', np.mean(benchmarklosses[:, 1]), np.mean(benchmarklosses[:, 2]), np.mean(benchmarklosses[:, 3]),
                            np.mean(benchmarklosses[:, 4]), np.mean(benchmarklosses[:, 5]), np.mean(benchmarklosses[:, 6]), np.mean(benchmarklosses[:, 7])]
                           + [c.lmbda.item() for c in constraints]
                           + [np.mean(benchmarklosses[:, 8]), np.mean(benchmarklosses[:, 9]), np.mean(benchmarklosses[:, 10]),
                              np.mean(benchmarklosses[:, 11]), np.mean(benchmarklosses[:, 12]), np.mean(benchmarklosses[:, 13])])

        csvwriter_val.writerow(['benchmark', np.mean(benchmarklosses[:, 1]), np.mean(benchmarklosses[:, 2]), np.mean(benchmarklosses[:, 3]),
                                np.mean(benchmarklosses[:, 4]), np.mean(benchmarklosses[:, 5]), np.mean(benchmarklosses[:, 6]), np.mean(benchmarklosses[:, 7])]
                               + [c.lmbda.item() for c in constraints])


        starting_point_mmd_fixsigma = starting_point_mmd_fixsigma / len(train_loader)
        starting_point_mmd = starting_point_mmd / len(train_loader)
        starting_point_mmd_response = starting_point_mmd_response / len(train_loader)
        starting_point_mse_fast = starting_point_mse_fast / len(train_loader)
        starting_point_mse_full = starting_point_mse_full / len(train_loader)
        starting_point_mae_fast = starting_point_mae_fast / len(train_loader)
        starting_point_mae_full = starting_point_mae_full / len(train_loader)
        
        starting_point_mmd_hadflav0 = starting_point_mmd_hadflav0 / len(train_loader)
        starting_point_mmd_hadflav4 = starting_point_mmd_hadflav4 / len(train_loader)
        starting_point_mmd_hadflav5 = starting_point_mmd_hadflav5 / len(train_loader)
        starting_point_l2dist_hadflav0 = starting_point_l2dist_hadflav0 / len(train_loader)
        starting_point_l2dist_hadflav4 = starting_point_l2dist_hadflav4 / len(train_loader)
        starting_point_l2dist_hadflav5 = starting_point_l2dist_hadflav5 / len(train_loader)

        print('starting_point_mmd_fixsigma')
        print(starting_point_mmd_fixsigma)
        print('starting_point_mmd')
        print(starting_point_mmd)
        print('starting_point_mmd_response')
        print(starting_point_mmd_response)
        print('starting_point_mse_fast')
        print(starting_point_mse_fast)
        print('starting_point_mse_full')
        print(starting_point_mse_full)
        print('starting_point_mae_fast')
        print(starting_point_mae_fast)
        print('starting_point_mae_full')
        print(starting_point_mae_full)
        print('starting_point_mmd_hadflav0')
        print(starting_point_mmd_hadflav0)
        print('starting_point_mmd_hadflav4')
        print(starting_point_mmd_hadflav4)
        print('starting_point_mmd_hadflav5')
        print(starting_point_mmd_hadflav5)

        csvwriter.writerow(['start', starting_point_mmd_fixsigma, starting_point_mmd, starting_point_mmd_response,
                            starting_point_mse_fast, starting_point_mse_full, starting_point_mae_fast, starting_point_mae_full]
                           + [c.lmbda.item() for c in constraints]
                           + [starting_point_mmd_hadflav0, starting_point_mmd_hadflav4, starting_point_mmd_hadflav5,
                              starting_point_l2dist_hadflav0, starting_point_l2dist_hadflav4, starting_point_l2dist_hadflav5])

        csvwriter_val.writerow(['start', starting_point_mmd_fixsigma, starting_point_mmd, starting_point_mmd_response,
                                starting_point_mse_fast, starting_point_mse_full, starting_point_mae_fast, starting_point_mae_full]
                               + [c.lmbda.item() for c in constraints])

        if normalizeindividualmmdshadflavtostart:


            hadflav_fraction_0 = np.sqrt(1./starting_point_mmd_hadflav0)
            hadflav_fraction_4 = np.sqrt(1./starting_point_mmd_hadflav4)
            hadflav_fraction_5 = np.sqrt(1./starting_point_mmd_hadflav5)

            print('hadflav_fraction_0')
            print(hadflav_fraction_0)
            print('hadflav_fraction_4')
            print(hadflav_fraction_4)
            print('hadflav_fraction_5')
            print(hadflav_fraction_5)


        if savesnapshots:

            model.eval()
            with torch.no_grad():

                for i, (X, y, _) in enumerate(val_loader):

                    if i >= snaphshotnumbatches: break

                    output = model(X)

                    if i == 0:
                        X_list = X
                        y_list = y
                        output_list = output
                    else:
                        X_list = torch.cat((X_list, X))
                        y_list = torch.cat((y_list, y))
                        output_list = torch.cat((output_list, output))

                    if transformdeepjet5to4:
                        y = DeepJetTransform5to4(deepjetindices=deepjetindicesWithoutParameters).forward(y)
                        X = DeepJetTransform5to4(deepjetindices=deepjetindicesWithParameters).forward(X)
                        output = DeepJetTransform5to4(deepjetindices=deepjetindicesWithoutParameters).forward(output)
                    elif transformdeepjet4to4:
                        y = DeepJetTransform4to4fromNano(deepjetindices=deepjetindicesWithoutParameters).forward(y)
                        X = DeepJetTransform4to4fromNano(deepjetindices=deepjetindicesWithParameters).forward(X)
                        output = DeepJetTransform4to4fromNano(deepjetindices=deepjetindicesWithoutParameters).forward(output)
                    if any(tanh200maskWithParameters): X = TanhTransform(mask=tanh200maskWithParameters, norm=200).forward(X)
                    if any(tanh200maskWithoutParameters):
                        y = TanhTransform(mask=tanh200maskWithoutParameters, norm=200).forward(y)
                        output = TanhTransform(mask=tanh200maskWithoutParameters, norm=200).forward(output)
                    # TODO: uncomment??
                    # if any(logitmaskWithParameters): X = LogitTransform(mask=logitmaskWithParameters, factor=logitfactor, onnxcompatible=False).forward(X)
                    # if any(logitmaskWithoutParameters):
                    #     y = LogitTransform(mask=logitmaskWithoutParameters, factor=logitfactor, onnxcompatible=False).forward(y)
                    #     output = LogitTransform(mask=logitmaskWithoutParameters, factor=logitfactor, onnxcompatible=False).forward(output)

                    if i == 0:
                        X_list_transformed = X
                        y_list_transformed = y
                        output_list_transformed = output
                    else:
                        X_list_transformed = torch.cat((X_list_transformed, X))
                        y_list_transformed = torch.cat((y_list_transformed, y))
                        output_list_transformed = torch.cat((output_list_transformed, output))

                snapshot(X_list, y_list, output_list, epoch, is_transformed=False)
                snapshot(X_list_transformed, y_list_transformed, output_list_transformed, epoch, is_transformed=True)

            model.train()

    epoch_train_loss = 0.0
    running_train_loss = 0.0
    running_val_loss = 0.0

    running_val_mmd_fixsigma = 0.0
    running_val_mmd = 0.0
    running_val_mmd_response = 0.0
    running_val_mse_fast = 0.0
    running_val_mse_full = 0.0
    running_val_mae_fast = 0.0
    running_val_mae_full = 0.0

    epoch_train_loss_mmd_fixsigma = 0.0
    epoch_train_loss_mmd = 0.0
    epoch_train_loss_mmd_response = 0.0
    epoch_train_loss_mse_fast = 0.0
    epoch_train_loss_mse_full = 0.0
    epoch_train_loss_mae_fast = 0.0
    epoch_train_loss_mae_full = 0.0

    nprinted = 0
    trainer.zero_grad()
    for i, (X, y, _) in enumerate(train_loader):

        output = model(X)

        if is_verbose:
            print('training')
            print('X original')
            print(X)
            print('y original')
            print(y)

        if transformdeepjet5to4:
            y = DeepJetTransform5to4(deepjetindices=deepjetindicesWithoutParameters).forward(y)
            X = DeepJetTransform5to4(deepjetindices=deepjetindicesWithParameters).forward(X)
        elif transformdeepjet4to4:
            y = DeepJetTransform4to4fromNano(deepjetindices=deepjetindicesWithoutParameters).forward(y)
            X = DeepJetTransform4to4fromNano(deepjetindices=deepjetindicesWithParameters).forward(X)
        if any(tanh200maskWithParameters): X = TanhTransform(mask=tanh200maskWithParameters, norm=200).forward(X)
        if any(tanh200maskWithoutParameters): y = TanhTransform(mask=tanh200maskWithoutParameters, norm=200).forward(y)
        # TODO: uncomment??
        # if any(logitmaskWithParameters): X = LogitTransform(mask=logitmaskWithParameters, factor=logitfactor, onnxcompatible=False).forward(X)
        # if any(logitmaskWithoutParameters): y = LogitTransform(mask=logitmaskWithoutParameters, factor=logitfactor, onnxcompatible=False).forward(y)

        if is_verbose:
            print('X')
            print(X)
            print('y')
            print(y)
            print('output')
            print(output)


        # initialize with dummy values because don't want to implement for all if/else cases...
        mmd_hadflav0, mmd_hadflav4, mmd_hadflav5, l2dist_hadflav0, l2dist_hadflav4, l2dist_hadflav5 = 0, 0, 0, 0, 0, 0
        if individualmmdshadflav:
            if includeparamsinmmd:
                if len(onehotencodeWithParamIdx) > 0:  # i.e. == 1
                    mmd_fixsigma = hadflav_fraction_0**2 * mmd_loss_fixsigma(output, y, parameters=OneHotEncode(
                        source_idx=onehotencodeWithParamIdx[0][2], target_vals=onehotencodeWithParamIdx[0][1]).forward(X)[:, :len(PARAMETERS)], mask=X[:, 2] == 0) + \
                                   hadflav_fraction_4**2 * mmd_loss_fixsigma(output, y, parameters=OneHotEncode(
                        source_idx=onehotencodeWithParamIdx[0][2], target_vals=onehotencodeWithParamIdx[0][1]).forward(X)[:, :len(PARAMETERS)], mask=X[:, 2] == 4) + \
                                   hadflav_fraction_5**2 * mmd_loss_fixsigma(output, y, parameters=OneHotEncode(
                        source_idx=onehotencodeWithParamIdx[0][2], target_vals=onehotencodeWithParamIdx[0][1]).forward(X)[:, :len(PARAMETERS)], mask=X[:, 2] == 5)

                    mmd = hadflav_fraction_0**2 * mmd_loss(output, y, parameters=OneHotEncode(
                        source_idx=onehotencodeWithParamIdx[0][2], target_vals=onehotencodeWithParamIdx[0][1]).forward(X)[:, :len(PARAMETERS)], mask=X[:, 2] == 0) + \
                          hadflav_fraction_4**2 * mmd_loss(output, y, parameters=OneHotEncode(
                        source_idx=onehotencodeWithParamIdx[0][2], target_vals=onehotencodeWithParamIdx[0][1]).forward(X)[:, :len(PARAMETERS)], mask=X[:, 2] == 4) + \
                          hadflav_fraction_5**2 * mmd_loss(output, y, parameters=OneHotEncode(
                        source_idx=onehotencodeWithParamIdx[0][2], target_vals=onehotencodeWithParamIdx[0][1]).forward(X)[:, :len(PARAMETERS)], mask=X[:, 2] == 5)

                    mmd_hadflav0 = hadflav_fraction_0**2 * mmd_loss(output, y, parameters=OneHotEncode(source_idx=onehotencodeWithParamIdx[0][2], target_vals=onehotencodeWithParamIdx[0][1]).forward(X)[:, :len(PARAMETERS)],
                                            mask=X[:, 2] == 0, l2dist_out='l2dist_hadflav0')
                    mmd_hadflav4 = hadflav_fraction_4**2 * mmd_loss(output, y, parameters=OneHotEncode(source_idx=onehotencodeWithParamIdx[0][2], target_vals=onehotencodeWithParamIdx[0][1]).forward(X)[:, :len(PARAMETERS)],
                                            mask=X[:, 2] == 4, l2dist_out='l2dist_hadflav4')
                    mmd_hadflav5 = hadflav_fraction_5**2 * mmd_loss(output, y, parameters=OneHotEncode(source_idx=onehotencodeWithParamIdx[0][2], target_vals=onehotencodeWithParamIdx[0][1]).forward(X)[:, :len(PARAMETERS)],
                                            mask=X[:, 2] == 5, l2dist_out='l2dist_hadflav5')
                    l2dist_hadflav0 = my_mmd.l2dist_hadflav0
                    l2dist_hadflav4 = my_mmd.l2dist_hadflav4
                    l2dist_hadflav5 = my_mmd.l2dist_hadflav5

                else:
                    mmd_fixsigma = hadflav_fraction_0**2 * mmd_loss_fixsigma(output, y, parameters=X[:, :realLenParameters], mask=X[:, 2] == 0) + \
                                   hadflav_fraction_4**2 * mmd_loss_fixsigma(output, y, parameters=X[:, :realLenParameters], mask=X[:, 2] == 4) + \
                                   hadflav_fraction_5**2 * mmd_loss_fixsigma(output, y, parameters=X[:, :realLenParameters], mask=X[:, 2] == 5)

                    mmd = hadflav_fraction_0**2 * mmd_loss(output, y, parameters=X[:, :realLenParameters], mask=X[:, 2] == 0) + \
                          hadflav_fraction_4**2 * mmd_loss(output, y, parameters=X[:, :realLenParameters], mask=X[:, 2] == 4) + \
                          hadflav_fraction_5**2 * mmd_loss(output, y, parameters=X[:, :realLenParameters], mask=X[:, 2] == 5)
            else:
                mmd_fixsigma = hadflav_fraction_0**2 * mmd_loss_fixsigma(output, y, mask=X[:, 2] == 0) + \
                               hadflav_fraction_4**2 * mmd_loss_fixsigma(output, y, mask=X[:, 2] == 4) + \
                               hadflav_fraction_5**2 * mmd_loss_fixsigma(output, y, mask=X[:, 2] == 5)

                mmd = hadflav_fraction_0**2 * mmd_loss(output, y, mask=X[:, 2] == 0) + \
                      hadflav_fraction_4**2 * mmd_loss(output, y, mask=X[:, 2] == 4) + \
                      hadflav_fraction_5**2 * mmd_loss(output, y, mask=X[:, 2] == 5)
        else:
            if includeparamsinmmd:
                if len(onehotencodeWithParamIdx) > 0:  # i.e. == 1
                    mmd_fixsigma = mmd_loss_fixsigma(output, y, parameters=OneHotEncode(
                        source_idx=onehotencodeWithParamIdx[0][2], target_vals=onehotencodeWithParamIdx[0][1]).forward(X)[:, :len(PARAMETERS)])
                    mmd = mmd_loss(output, y, parameters=OneHotEncode(
                        source_idx=onehotencodeWithParamIdx[0][2], target_vals=onehotencodeWithParamIdx[0][1]).forward(X)[:, :len(PARAMETERS)])
                else:
                    mmd_fixsigma = mmd_loss_fixsigma(output, y, parameters=X[:, :realLenParameters])
                    mmd = mmd_loss(output, y, parameters=X[:, :realLenParameters])
            else:
                mmd_fixsigma = mmd_loss_fixsigma(output, y)
                mmd = mmd_loss(output, y)
        # mmd_response = mmd_loss_fixsigma(output[:, :1] / X[:, :1], y[:, :1] / X[:, :1])
        mmd_response = torch.zeros(1)
        mse_fast = mse_loss(output, X[:, realLenParameters:])
        mse_full = mse_loss(output, y)
        # mse_full = mse_loss(output[:, 1:], y[:, 1:])
        mae_fast = mae_loss(output, X[:, realLenParameters:])
        mae_full = mae_loss(output, y)

        csvwriter.writerow([iteration, mmd_fixsigma.item(), mmd.item(), mmd_response.item(),
                            mse_fast.item(), mse_full.item(), mae_fast.item(), mae_full.item()]
                           + [c.lmbda.item() for c in constraints]
                           + [mmd_hadflav0.item(), mmd_hadflav4.item(), mmd_hadflav5.item(),
                              l2dist_hadflav0, l2dist_hadflav4, l2dist_hadflav5])

        if mmd_response_scale > 0 or mse_fast_scale > 0 or mse_full_scale > 0 or mae_fast_scale > 0 or mae_full_scale > 0:  # sometimes NaN error
            train_loss = mmd_scale * mmd + mmd_response_scale * mmd_response + mse_fast_scale * mse_fast + mse_full_scale * mse_full + mae_fast_scale * mae_fast + mae_full_scale * mae_full
        else:
            train_loss = mmd_scale * mmd

        if torch.isnan(train_loss).any():

            print(torch.isnan(X).any())
            print(X)
            print(torch.isnan(y).any())
            print(y)
            print(torch.isnan(output).any())
            print(output)
            print(torch.min(output))
            print(torch.max(output))
            print(output[torch.isnan(output)])
            print(torch.stack([torch.isnan(p).any() for p in model.parameters()]).any())
            print(train_loss)
            print(mmd_scale)
            print(mmd)
            print(mmd_hadflav0)
            print(mmd_hadflav4)
            print(mmd_hadflav5)
            print(l2dist_hadflav0)
            print(l2dist_hadflav4)
            print(l2dist_hadflav5)

            sys.exit(1)


        if usemdmm:
            mdmm_return = optimizer(mmd, X, y, output)
            # if accumulategradients > 1:
            #     mdmm_return.value = mdmm_return.value / accumulategradients
            mdmm_return.value.backward()
        else:
            # if accumulategradients > 1:
            #     train_loss = train_loss / accumulategradients
            train_loss.backward()

        if not disable_training:
            if (epoch < startepochaccgrads) or ((i + 1) % accumulategradients == 0) or ((i + 1) == len(train_loader)):
                trainer.step()
                trainer.zero_grad()

        if is_verbose:
            print('\ntraining batch', i)
            print('training loss', train_loss.item())

        epoch_train_loss += train_loss.item()
        running_train_loss += train_loss.item()

        epoch_train_loss_mmd_fixsigma += mmd_fixsigma.item()
        epoch_train_loss_mmd += mmd_scale * mmd.item()
        epoch_train_loss_mmd_response += mmd_response_scale * mmd_response.item()
        epoch_train_loss_mse_fast += mse_fast_scale * mse_fast.item()
        epoch_train_loss_mse_full += mse_full_scale * mse_full.item()
        epoch_train_loss_mae_fast += mae_fast_scale * mae_fast.item()
        epoch_train_loss_mae_full += mae_full_scale * mae_full.item()

        if cyclic_lr:
            lrscheduler.step()

        iteration += 1

        if i % 32 == 31:    # print average loss every 32 mini-batches (max. 10 times)
            if nprinted < 10:
                nprinted += 1
                print('[{}, {}] train loss: {:.10f}'.format(epoch+1, i+1, running_train_loss / 32))
            running_train_loss = 0.0

    print('[{}] train loss: {:.10f}'.format(epoch + 1, epoch_train_loss / len(train_loader)))
    print('mmd_fixsigma, mmd, mmd_response, mse_fast, mse_full, mae_fast, mae_full',
          epoch_train_loss_mmd_fixsigma / len(train_loader),
          epoch_train_loss_mmd / len(train_loader),
          epoch_train_loss_mmd_response / len(train_loader),
          epoch_train_loss_mse_fast / len(train_loader),
          epoch_train_loss_mse_full / len(train_loader),
          epoch_train_loss_mae_fast / len(train_loader),
          epoch_train_loss_mae_full / len(train_loader))

    # validation loop
    model.eval()
    with torch.no_grad():

        for i, (X, y, _) in enumerate(val_loader):

            output = model(X)

            if savesnapshots and epoch % snapshoteveryXepoch == 0 and i < snaphshotnumbatches:
                if i == 0:
                    output_list = output
                else:
                    output_list = torch.cat((output_list, output))

            if is_verbose:
                print('validation')
                print('X original')
                print(X)
                print('y original')
                print(y)
                print('output original')
                print(output)

            if transformdeepjet5to4:
                y = DeepJetTransform5to4(deepjetindices=deepjetindicesWithoutParameters).forward(y)
                X = DeepJetTransform5to4(deepjetindices=deepjetindicesWithParameters).forward(X)
                output = DeepJetTransform5to4(deepjetindices=deepjetindicesWithoutParameters).forward(output)
            elif transformdeepjet4to4:
                y = DeepJetTransform4to4fromNano(deepjetindices=deepjetindicesWithoutParameters).forward(y)
                X = DeepJetTransform4to4fromNano(deepjetindices=deepjetindicesWithParameters).forward(X)
                output = DeepJetTransform4to4fromNano(deepjetindices=deepjetindicesWithoutParameters).forward(output)
            if any(tanh200maskWithParameters): X = TanhTransform(mask=tanh200maskWithParameters, norm=200).forward(X)
            if any(tanh200maskWithoutParameters):
                y = TanhTransform(mask=tanh200maskWithoutParameters, norm=200).forward(y)
                output = TanhTransform(mask=tanh200maskWithoutParameters, norm=200).forward(output)
            # TODO: uncomment??
            # if any(logitmaskWithParameters): X = LogitTransform(mask=logitmaskWithParameters, factor=logitfactor, onnxcompatible=False).forward(X)
            # if any(logitmaskWithoutParameters):
            #     y = LogitTransform(mask=logitmaskWithoutParameters, factor=logitfactor, onnxcompatible=False).forward(y)
            #     output = LogitTransform(mask=logitmaskWithoutParameters, factor=logitfactor, onnxcompatible=False).forward(output)

            if savesnapshots and epoch % snapshoteveryXepoch == 0 and i < snaphshotnumbatches:
                if i == 0:
                    output_list_transformed = output
                else:
                    output_list_transformed = torch.cat((output_list_transformed, output))

            if is_verbose:
                print('X')
                print(X)
                print('y')
                print(y)
                print('output')
                print(output)

            if individualmmdshadflav:
                if includeparamsinmmd:
                    if len(onehotencodeWithParamIdx) > 0:  # i.e. == 1
                        mmd_fixsigma_val = hadflav_fraction_0**2 * mmd_loss_fixsigma(output, y, parameters=OneHotEncode(
                            source_idx=onehotencodeWithParamIdx[0][2], target_vals=onehotencodeWithParamIdx[0][1]).forward(X)[:, :len(PARAMETERS)], mask=X[:, 2] == 0) + \
                                           hadflav_fraction_4**2 * mmd_loss_fixsigma(output, y, parameters=OneHotEncode(
                            source_idx=onehotencodeWithParamIdx[0][2], target_vals=onehotencodeWithParamIdx[0][1]).forward(X)[:, :len(PARAMETERS)], mask=X[:, 2] == 4) + \
                                           hadflav_fraction_5**2 * mmd_loss_fixsigma(output, y, parameters=OneHotEncode(
                            source_idx=onehotencodeWithParamIdx[0][2], target_vals=onehotencodeWithParamIdx[0][1]).forward(X)[:, :len(PARAMETERS)], mask=X[:, 2] == 5)

                        mmd_val = hadflav_fraction_0**2 * mmd_loss(output, y, parameters=OneHotEncode(
                            source_idx=onehotencodeWithParamIdx[0][2], target_vals=onehotencodeWithParamIdx[0][1]).forward(X)[:, :len(PARAMETERS)], mask=X[:, 2] == 0) + \
                                  hadflav_fraction_4**2 * mmd_loss(output, y, parameters=OneHotEncode(
                            source_idx=onehotencodeWithParamIdx[0][2], target_vals=onehotencodeWithParamIdx[0][1]).forward(X)[:, :len(PARAMETERS)], mask=X[:, 2] == 4) + \
                                  hadflav_fraction_5**2 * mmd_loss(output, y, parameters=OneHotEncode(
                            source_idx=onehotencodeWithParamIdx[0][2], target_vals=onehotencodeWithParamIdx[0][1]).forward(X)[:, :len(PARAMETERS)], mask=X[:, 2] == 5)
                    else:
                        mmd_fixsigma_val = hadflav_fraction_0**2 * mmd_loss_fixsigma(output, y, parameters=X[:, :realLenParameters], mask=X[:, 2] == 0) + \
                                           hadflav_fraction_4**2 * mmd_loss_fixsigma(output, y, parameters=X[:, :realLenParameters], mask=X[:, 2] == 4) + \
                                           hadflav_fraction_5**2 * mmd_loss_fixsigma(output, y, parameters=X[:, :realLenParameters], mask=X[:, 2] == 5)

                        mmd_val = hadflav_fraction_0**2 * mmd_loss(output, y, parameters=X[:, :realLenParameters], mask=X[:, 2] == 0) + \
                                  hadflav_fraction_4**2 * mmd_loss(output, y, parameters=X[:, :realLenParameters], mask=X[:, 2] == 4) + \
                                  hadflav_fraction_5**2 * mmd_loss(output, y, parameters=X[:, :realLenParameters], mask=X[:, 2] == 5)
                else:
                    mmd_fixsigma_val = hadflav_fraction_0**2 * mmd_loss_fixsigma(output, y, mask=X[:, 2] == 0) + \
                                       hadflav_fraction_4**2 * mmd_loss_fixsigma(output, y, mask=X[:, 2] == 4) + \
                                       hadflav_fraction_5**2 * mmd_loss_fixsigma(output, y, mask=X[:, 2] == 5)

                    mmd_val = hadflav_fraction_0**2 * mmd_loss(output, y, mask=X[:, 2] == 0) + \
                              hadflav_fraction_4**2 * mmd_loss(output, y, mask=X[:, 2] == 4) + \
                              hadflav_fraction_5**2 * mmd_loss(output, y, mask=X[:, 2] == 5)
            else:
                if includeparamsinmmd:
                    if len(onehotencodeWithParamIdx) > 0:  # i.e. == 1
                        mmd_fixsigma_val = mmd_loss_fixsigma(output, y, parameters=OneHotEncode(
                            source_idx=onehotencodeWithParamIdx[0][2], target_vals=onehotencodeWithParamIdx[0][1]).forward(X)[:, :len(PARAMETERS)])
                        mmd_val = mmd_loss(output, y, parameters=OneHotEncode(
                            source_idx=onehotencodeWithParamIdx[0][2], target_vals=onehotencodeWithParamIdx[0][1]).forward(X)[:, :len(PARAMETERS)])
                    else:
                        mmd_fixsigma_val = mmd_loss_fixsigma(output, y, parameters=X[:, :realLenParameters])
                        mmd_val = mmd_loss(output, y, parameters=X[:, :realLenParameters])
                else:
                    mmd_fixsigma_val = mmd_loss_fixsigma(output, y)
                    mmd_val = mmd_loss(output, y)
            # mmd_response_val = mmd_loss_fixsigma(output[:, :1] / X[:, :1], y[:, :1] / X[:, :1])
            mmd_response_val = torch.zeros(1)
            mse_fast_val = mse_loss(output, X[:, realLenParameters:])
            mse_full_val = mse_loss(output, y)
            # mse_full_val = mse_loss(output[:, 1:], y[:, 1:])
            mae_fast_val = mae_loss(output, X[:, realLenParameters:])
            mae_full_val = mae_loss(output, y)

            csvwriter_val.writerow([iteration, mmd_fixsigma_val.item(), mmd_val.item(), mmd_response_val.item(),
                                    mse_fast_val.item(), mse_full_val.item(), mae_fast_val.item(), mae_full_val.item()]
                                   + [c.lmbda.item() for c in constraints])

            if mmd_response_scale > 0 or mse_fast_scale > 0 or mse_full_scale > 0 or mae_fast_scale > 0 or mae_full_scale > 0:  # sometimes NaN error
                val_loss = mmd_scale * mmd_val + mmd_response_scale * mmd_response_val + mse_fast_scale * mse_fast_val + mse_full_scale * mse_full_val + mae_fast_scale * mae_fast_val + mae_full_scale * mae_full_val
            else:
                val_loss = mmd_scale * mmd_val

            running_val_loss += val_loss.item()

            running_val_mmd_fixsigma += mmd_fixsigma_val.item()
            running_val_mmd += mmd_val.item()
            running_val_mmd_response += mmd_response_val.item()
            running_val_mse_fast += mse_fast_val.item()
            running_val_mse_full += mse_full_val.item()
            running_val_mae_fast += mae_fast_val.item()
            running_val_mae_full += mae_full_val.item()

            if is_verbose:
                print('\nvalidation batch', i)
                print('validation loss', val_loss.item())

            if epoch == 0:

                if individualmmdshadflav:
                    if includeparamsinmmd:
                        if len(onehotencodeWithParamIdx) > 0:  # i.e. == 1
                            bm_mmd_val = hadflav_fraction_0**2 * mmd_loss(X[:, realLenParameters:], y, parameters=OneHotEncode(
                                source_idx=onehotencodeWithParamIdx[0][2], target_vals=onehotencodeWithParamIdx[0][1]).forward(X)[:, :len(PARAMETERS)], mask=X[:, 2] == 0).item()
                            bm_mmd_val += hadflav_fraction_4**2 * mmd_loss(X[:, realLenParameters:], y, parameters=OneHotEncode(
                                source_idx=onehotencodeWithParamIdx[0][2], target_vals=onehotencodeWithParamIdx[0][1]).forward(X)[:, :len(PARAMETERS)], mask=X[:, 2] == 4).item()
                            bm_mmd_val += hadflav_fraction_5**2 * mmd_loss(X[:, realLenParameters:], y, parameters=OneHotEncode(
                                source_idx=onehotencodeWithParamIdx[0][2], target_vals=onehotencodeWithParamIdx[0][1]).forward(X)[:, :len(PARAMETERS)], mask=X[:, 2] == 5).item()
                        else:
                            bm_mmd_val = hadflav_fraction_0**2 * mmd_loss(X[:, realLenParameters:], y, parameters=X[:, :realLenParameters], mask=X[:, 2] == 0).item()
                            bm_mmd_val += hadflav_fraction_4**2 * mmd_loss(X[:, realLenParameters:], y, parameters=X[:, :realLenParameters], mask=X[:, 2] == 4).item()
                            bm_mmd_val += hadflav_fraction_5**2 * mmd_loss(X[:, realLenParameters:], y, parameters=X[:, :realLenParameters], mask=X[:, 2] == 5).item()
                    else:
                        bm_mmd_val = hadflav_fraction_0**2 * mmd_loss(X[:, realLenParameters:], y, mask=X[:, 2] == 0).item()
                        bm_mmd_val += hadflav_fraction_4**2 * mmd_loss(X[:, realLenParameters:], y, mask=X[:, 2] == 4).item()
                        bm_mmd_val += hadflav_fraction_5**2 * mmd_loss(X[:, realLenParameters:], y, mask=X[:, 2] == 5).item()
                else:
                    if includeparamsinmmd:
                        if len(onehotencodeWithParamIdx) > 0:  # i.e. == 1
                            bm_mmd_val = mmd_loss(X[:, realLenParameters:], y, parameters=OneHotEncode(
                                source_idx=onehotencodeWithParamIdx[0][2], target_vals=onehotencodeWithParamIdx[0][1]).forward(X)[:, :len(PARAMETERS)]).item()
                        else:
                            bm_mmd_val = mmd_loss(X[:, realLenParameters:], y, parameters=X[:, :realLenParameters]).item()
                    else:
                        bm_mmd_val = mmd_loss(X[:, realLenParameters:], y).item()

                if mmd_response_scale > 0 or mse_fast_scale > 0 or mse_full_scale > 0 or mae_fast_scale > 0 or mae_full_scale > 0:
                    benchmarklosses_val.append(mmd_scale * bm_mmd_val
                                               # + mmd_response_scale * mmd_loss_fixsigma(X[:, realLenParameters:realLenParameters + 1] / X[:, :1], y[:, :1] / X[:, :1]).item()
                                               + mmd_response_scale * torch.zeros(1).item()
                                               + mse_fast_scale * mse_loss(X[:, realLenParameters:], X[:, realLenParameters:]).item()
                                               + mse_full_scale * mse_loss(X[:, realLenParameters + 1:], y[:, 1:]).item()
                                               + mae_fast_scale * mae_loss(X[:, realLenParameters:], X[:, realLenParameters:]).item()
                                               + mae_full_scale * mae_loss(X[:, realLenParameters + 1:], y[:, 1:]).item())
                else:
                    benchmarklosses_val.append(mmd_scale * bm_mmd_val)


        if is_test and False:

            for j in range(2):

                if j == 0:
                    print('\neval mode')
                else:
                    model.train()
                    print('\ntrain mode')

                epoch_train_loss_mmd_fixsigma = 0.0
                epoch_train_loss_mmd = 0.0
                epoch_train_loss_mmd_response = 0.0
                epoch_train_loss_mse_fast = 0.0
                epoch_train_loss_mse_full = 0.0
                epoch_train_loss_mae_fast = 0.0
                epoch_train_loss_mae_full = 0.0

                for i, (X, y, _) in enumerate(train_loader):


                    output = model(X)

                    if is_verbose:
                        print('X original')
                        print(X)
                        print('y original')
                        print(y)
                        print('output original')
                        print(output)

                    if transformdeepjet5to4:
                        y = DeepJetTransform5to4(deepjetindices=deepjetindicesWithoutParameters).forward(y)
                        X = DeepJetTransform5to4(deepjetindices=deepjetindicesWithParameters).forward(X)
                        if j == 0: output = DeepJetTransform5to4(deepjetindices=deepjetindicesWithoutParameters).forward(output)
                    elif transformdeepjet4to4:
                        y = DeepJetTransform4to4fromNano(deepjetindices=deepjetindicesWithoutParameters).forward(y)
                        X = DeepJetTransform4to4fromNano(deepjetindices=deepjetindicesWithParameters).forward(X)
                        if j == 0: output = DeepJetTransform4to4fromNano(deepjetindices=deepjetindicesWithoutParameters).forward(output)
                    if any(tanh200maskWithParameters): X = TanhTransform(mask=tanh200maskWithParameters, norm=200).forward(X)
                    if any(tanh200maskWithoutParameters):
                        y = TanhTransform(mask=tanh200maskWithoutParameters, norm=200).forward(y)
                        if j == 0: output = TanhTransform(mask=tanh200maskWithoutParameters, norm=200).forward(output)
                    # TODO: uncomment??
                    # if any(logitmaskWithParameters): X = LogitTransform(mask=logitmaskWithParameters, factor=logitfactor, onnxcompatible=False).forward(X)
                    # if any(logitmaskWithoutParameters):
                    #     y = LogitTransform(mask=logitmaskWithoutParameters, factor=logitfactor, onnxcompatible=False).forward(y)
                    #     if j == 0: output = LogitTransform(mask=logitmaskWithoutParameters, factor=logitfactor, onnxcompatible=False).forward(output)

                    if is_verbose:
                        print('X')
                        print(X)
                        print('y')
                        print(y)
                        print('output')
                        print(output)

                    if individualmmdshadflav:
                        if includeparamsinmmd:
                            if len(onehotencodeWithParamIdx) > 0:  # i.e. == 1
                                mmd_fixsigma = hadflav_fraction_0**2 * mmd_loss_fixsigma(output, y, parameters=OneHotEncode(
                                    source_idx=onehotencodeWithParamIdx[0][2], target_vals=onehotencodeWithParamIdx[0][1]).forward(X)[:, :len(PARAMETERS)], mask=X[:, 2] == 0) + \
                                               hadflav_fraction_4**2 * mmd_loss_fixsigma(output, y, parameters=OneHotEncode(
                                    source_idx=onehotencodeWithParamIdx[0][2], target_vals=onehotencodeWithParamIdx[0][1]).forward(X)[:, :len(PARAMETERS)], mask=X[:, 2] == 4) + \
                                               hadflav_fraction_5**2 * mmd_loss_fixsigma(output, y, parameters=OneHotEncode(
                                    source_idx=onehotencodeWithParamIdx[0][2], target_vals=onehotencodeWithParamIdx[0][1]).forward(X)[:, :len(PARAMETERS)], mask=X[:, 2] == 5)

                                mmd = hadflav_fraction_0**2 * mmd_loss(output, y, parameters=OneHotEncode(
                                    source_idx=onehotencodeWithParamIdx[0][2], target_vals=onehotencodeWithParamIdx[0][1]).forward(X)[:, :len(PARAMETERS)], mask=X[:, 2] == 0) + \
                                      hadflav_fraction_4**2 * mmd_loss(output, y, parameters=OneHotEncode(
                                    source_idx=onehotencodeWithParamIdx[0][2], target_vals=onehotencodeWithParamIdx[0][1]).forward(X)[:, :len(PARAMETERS)], mask=X[:, 2] == 4) + \
                                      hadflav_fraction_5**2 * mmd_loss(output, y, parameters=OneHotEncode(
                                    source_idx=onehotencodeWithParamIdx[0][2], target_vals=onehotencodeWithParamIdx[0][1]).forward(X)[:, :len(PARAMETERS)], mask=X[:, 2] == 5)
                            else:
                                mmd_fixsigma = hadflav_fraction_0**2 * mmd_loss_fixsigma(output, y, parameters=X[:, :realLenParameters], mask=X[:, 2] == 0) + \
                                               hadflav_fraction_4**2 * mmd_loss_fixsigma(output, y, parameters=X[:, :realLenParameters], mask=X[:, 2] == 4) + \
                                               hadflav_fraction_5**2 * mmd_loss_fixsigma(output, y, parameters=X[:, :realLenParameters], mask=X[:, 2] == 5)

                                mmd = hadflav_fraction_0**2 * mmd_loss(output, y, parameters=X[:, :realLenParameters], mask=X[:, 2] == 0) + \
                                      hadflav_fraction_4**2 * mmd_loss(output, y, parameters=X[:, :realLenParameters], mask=X[:, 2] == 4) + \
                                      hadflav_fraction_5**2 * mmd_loss(output, y, parameters=X[:, :realLenParameters], mask=X[:, 2] == 5)
                        else:
                            mmd_fixsigma = hadflav_fraction_0**2 * mmd_loss_fixsigma(output, y, mask=X[:, 2] == 0) + \
                                           hadflav_fraction_4**2 * mmd_loss_fixsigma(output, y, mask=X[:, 2] == 4) + \
                                           hadflav_fraction_5**2 * mmd_loss_fixsigma(output, y, mask=X[:, 2] == 5)

                            mmd = hadflav_fraction_0**2 * mmd_loss(output, y, mask=X[:, 2] == 0) + \
                                  hadflav_fraction_4**2 * mmd_loss(output, y, mask=X[:, 2] == 4) + \
                                  hadflav_fraction_5**2 * mmd_loss(output, y, mask=X[:, 2] == 5)
                    else:
                        if includeparamsinmmd:
                            if len(onehotencodeWithParamIdx) > 0:  # i.e. == 1
                                mmd_fixsigma = mmd_loss_fixsigma(output, y, parameters=OneHotEncode(
                                    source_idx=onehotencodeWithParamIdx[0][2], target_vals=onehotencodeWithParamIdx[0][1]).forward(X)[:, :len(PARAMETERS)])
                                mmd = mmd_loss(output, y, parameters=OneHotEncode(
                                    source_idx=onehotencodeWithParamIdx[0][2], target_vals=onehotencodeWithParamIdx[0][1]).forward(X)[:, :len(PARAMETERS)])
                            else:
                                mmd_fixsigma = mmd_loss_fixsigma(output, y, parameters=X[:, :realLenParameters])
                                mmd = mmd_loss(output, y, parameters=X[:, :realLenParameters])
                        else:
                            mmd_fixsigma = mmd_loss_fixsigma(output, y)
                            mmd = mmd_loss(output, y)
                    # mmd_response = mmd_loss_fixsigma(output[:, :1] / X[:, :1], y[:, :1] / X[:, :1])
                    mmd_response = torch.zeros(1)
                    mse_fast = mse_loss(output, X[:, realLenParameters:])
                    mse_full = mse_loss(output, y)
                    # mse_full = mse_loss(output[:, 1:], y[:, 1:])
                    mae_fast = mae_loss(output, X[:, realLenParameters:])
                    mae_full = mae_loss(output, y)


                    epoch_train_loss_mmd_fixsigma += mmd_fixsigma.item()
                    epoch_train_loss_mmd += mmd_scale * mmd.item()
                    epoch_train_loss_mmd_response += mmd_response_scale * mmd_response.item()
                    epoch_train_loss_mse_fast += mse_fast_scale * mse_fast.item()
                    epoch_train_loss_mse_full += mse_full_scale * mse_full.item()
                    epoch_train_loss_mae_fast += mae_fast_scale * mae_fast.item()
                    epoch_train_loss_mae_full += mae_full_scale * mae_full.item()

                print('mmd_fixsigma, mmd, mmd_response, mse_fast, mse_full, mae_fast, mae_full',
                      epoch_train_loss_mmd_fixsigma / len(train_loader),
                      epoch_train_loss_mmd / len(train_loader),
                      epoch_train_loss_mmd_response / len(train_loader),
                      epoch_train_loss_mse_fast / len(train_loader),
                      epoch_train_loss_mse_full / len(train_loader),
                      epoch_train_loss_mae_fast / len(train_loader),
                      epoch_train_loss_mae_full / len(train_loader))


        if savesnapshots and epoch % snapshoteveryXepoch == 0:

            snapshot(X_list, y_list, output_list, epoch+1, is_transformed=False)
            snapshot(X_list_transformed, y_list_transformed, output_list_transformed, epoch+1, is_transformed=True)

        curr_val = running_val_loss / len(val_loader)

        # don't save the model with the lowest val loss because it might not be the optimal choice on the pareto front
        # if curr_val < best_val:
        #     m = torch.jit.script(model)
        #     torch.jit.save(m, '/nfs/dust/cms/user/wolfmor/Refinement/TrainingOutput/regression_model_' + trainingID + '.pt')


        curr_val_mmd_fixsigma = running_val_mmd_fixsigma / len(val_loader)
        curr_val_mmd = running_val_mmd / len(val_loader)
        curr_val_mmd_response = running_val_mmd_response / len(val_loader)
        curr_val_mse_fast = running_val_mse_fast / len(val_loader)
        curr_val_mse_full = running_val_mse_full / len(val_loader)
        curr_val_mae_fast = running_val_mae_fast / len(val_loader)
        curr_val_mae_full = running_val_mae_full / len(val_loader)

        print('[{}] val loss: {:.10f}'.format(epoch + 1, curr_val))

        running_val_loss = 0.0

        running_val_mmd_fixsigma = 0.0
        running_val_mmd = 0.0
        running_val_mmd_response = 0.0
        running_val_mse_fast = 0.0
        running_val_mse_full = 0.0
        running_val_mae_fast = 0.0
        running_val_mae_full = 0.0

        if epoch == 0:
            print('[{}] benchmark loss val: {:.10f}'.format(epoch + 1, np.mean(benchmarklosses_val)))

    if not cyclic_lr:
        if lr_scheduler_gamma == 0:

            if usemdmm:
                lr_val = curr_val_mmd
                for c in constraintsconfig:
                    lr_val += abs(constraintsconfig[c][0] - globals()['curr_val_' + c])

                # for ic, c in enumerate(constraints):
                #     lr_val += abs(c.lmbda.item() * mdmm_return.infs[ic].item())

                lrscheduler.step(lr_val)
            else:
                lrscheduler.step(curr_val)

        else:

            lrscheduler.step()

print('finished training on {} epochs!'.format(my_num_epochs))

m = torch.jit.script(model)
torch.jit.save(m, '/nfs/dust/cms/user/wolfmor/Refinement/TrainingOutput/regression_model_' + trainingID + '.pt')

'''
###############################################################################################
# testing loop
###############################################################################################
'''
print('\n###testing loop')

out_dict = {'isTrainValTest': []}
for branch in dict_X:
    out_dict[branch] = []
for branch in dict_y:
    out_dict[branch] = []
for branch in dict_y:
    out_dict[branch.replace('FullSim', 'Refined')] = []
for branch in dict_spec:
    out_dict[branch] = []

model.eval()

with torch.no_grad():
    print('train data')
    for i, (X, y, spec) in enumerate(train_loader):

        output = model(X)

        out_dict['isTrainValTest'].append(torch.ones(X.size(dim=0), dtype=torch.int) * 0)

        for ib, branch in enumerate(dict_X):
            out_dict[branch].append(X[:, ib])
        for ib, branch in enumerate(dict_y):
            out_dict[branch].append(y[:, ib])
        for ib, branch in enumerate(dict_y):
            out_dict[branch.replace('FullSim', 'Refined')].append(output[:, ib])
        for ib, branch in enumerate(dict_spec):
            out_dict[branch].append(spec[:, ib])

    print('val data')
    for i, (X, y, spec) in enumerate(val_loader):

        output = model(X)

        out_dict['isTrainValTest'].append(torch.ones(X.size(dim=0), dtype=torch.int) * 1)

        for ib, branch in enumerate(dict_X):
            out_dict[branch].append(X[:, ib])
        for ib, branch in enumerate(dict_y):
            out_dict[branch].append(y[:, ib])
        for ib, branch in enumerate(dict_y):
            out_dict[branch.replace('FullSim', 'Refined')].append(output[:, ib])
        for ib, branch in enumerate(dict_spec):
            out_dict[branch].append(spec[:, ib])

    print('test data')
    for i, (X, y, spec) in enumerate(test_loader):

        output = model(X)

        out_dict['isTrainValTest'].append(torch.ones(X.size(dim=0), dtype=torch.int) * 2)

        for ib, branch in enumerate(dict_X):
            out_dict[branch].append(X[:, ib])
        for ib, branch in enumerate(dict_y):
            out_dict[branch].append(y[:, ib])
        for ib, branch in enumerate(dict_y):
            out_dict[branch.replace('FullSim', 'Refined')].append(output[:, ib])
        for ib, branch in enumerate(dict_spec):
            out_dict[branch].append(spec[:, ib])

for branch in out_dict:
    out_dict[branch] = torch.cat(out_dict[branch]).detach().cpu().numpy()


'''
###############################################################################################
# save output
###############################################################################################
'''
print('\n###save output')

foutcsv.close()
print('just created ' + foutcsv.name)

foutcsv_val.close()
print('just created ' + foutcsv_val.name)

out_rdf = ROOT.RDF.MakeNumpyDataFrame(out_dict)
out_rdf.Snapshot('tJet', out_path)
print('just created ' + out_path)
