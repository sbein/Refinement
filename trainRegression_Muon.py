"""
to load packages, e.g.:
source /cvmfs/sft.cern.ch/lcg/views/LCG_101cuda/x86_64-centos7-gcc8-opt/setup.sh
- or -
source /cvmfs/sft.cern.ch/lcg/views/LCG_101/x86_64-centos7-gcc8-opt/setup.sh

#first 
screen
#(making sure that the interactive job is created in the screen session and not killed when I log off), 
#then 
condor_submit -i interactive.submit 
#and then when the job is spawned, I source my 
source /afs/desy.de/user/b/beinsam/.bash_profile
torefinement
cmsenv
cd /nfs/dust/cms/user/beinsam/FastSim/Refinement/Regress
source /cvmfs/sft.cern.ch/lcg/views/LCG_101cuda/x86_64-centos7-gcc8-opt/setup.sh
#to write terminal output to a txt file:
python3 trainRegression_Muon.py 2>&1 | tee traininglog_regressionMuon.txt
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


'''
# ##### ##### #####
# general settings
# ##### ##### #####
'''

training_id = datetime.today().strftime('%Y%m%d%H%M') # will be used as an identifier in the output filenames, adapt if needed

is_test = False
#verbose = 10
verbosity = 10

'''
# ##### ##### ##### #####
# define input/output
# ##### ##### ##### #####
'''

in_path = '/nfs/dust/cms/user/beinsam/FastSim/Refinement/output/mc_fullfast_T1tttt_JetsMuonsElectronsPhotonsTausEvents.root'
in_tree = 'tMuon'
preselection = ''#'GenMuon_nearest_dR>0.5&&RecMuon_nearest_dR_FastSim>0.5&&RecMuon_nearest_dR_FullSim>0.5'
preselection = "RecMuon_mvaMuID_FastSim > -10 && RecMuon_mvaMuID_FullSim > -10 && RecMuon_softMva_FastSim > -10 && RecMuon_softMva_FullSim > -10"
out_path = '/nfs/dust/cms/user/beinsam/FastSim/Refinement/Regress/TrainingOutput/output_refineMuon_regression_' + training_id + '.root'


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
    num_epochs = 1000
    num_epochs = 100
    #num_epochs = 5
    #num_epochs = 0

learning_rate = 1e-5
lr_scheduler_gamma = 1.

if is_test: batch_size = 4096
else: batch_size = 4096

batch_size = 1024

if is_test: num_batches = [2, 2, 2]
else: num_batches = [500, 100, 200]


'''
# ##### ##### ##### ##### ##### ##### #####
# define variables and transformations
# ##### ##### ##### ##### ##### ##### #####
'''

onehotencode = ('RecMuon_hadronFlavour_FastSim', [0, 4, 5])  # add one-hot-encoding for given input variable with the given values
onehotencode = False

PARAMETERS = [
    ('GenMuon_pt', []),
    ('GenMuon_eta', []),
]

# if using DeepJetConstraint the DeepJet transformations have to be explicitly adapted in the DeepJetConstraint module
VARIABLES = [
    ('RecMuon_mvaMuID_CLASS', ['logit']),#'logit'
    ('RecMuon_softMva_CLASS', ['logit']),#'logit'
    ('RecMuon_mvaLowPt_CLASS',['fisher']),#'fisher'
    ('RecMuon_mvaTTH_CLASS',  ['fisher']) #'fisher'
]

spectators = [
    'GenMuon_pt',
    'GenMuon_eta',
    'GenMuon_phi'#charge here?
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
column_names = rdf.GetColumnNames()
print("Branches/Columns in the DataFrame:")
for name in column_names:
    print(name)

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

model = nn.Sequential()

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
if any(fishermaskWithoutParameters): model.add_module('FisherTransformBack', FisherTransformBack(mask=fishermaskWithoutParameters, factor=fisherfactor))
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

mmdfixsigma_fn = my_mmd.MMD(kernel_mul=5., kernel_num=5,one_sided_bandwidth=True,calculate_fix_sigma_for_each_dimension_with_target_only=True)# fix_sigma=true by default
mmd_fn = my_mmd.MMD(kernel_mul=2., kernel_num=5)
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
    'mmd_output_target':
        lambda inp_, out_, target_: mmd_fn(out_, target_,
                                           parameters=(OneHotEncode(source_idx=onehotencode[2], target_vals=onehotencode[1]).forward(inp_)[:, :len(PARAMETERS)]
                                                       if onehotencode else inp_[:, :len(PARAMETERS)])
                                           if includeparametersinmmd else None),
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
}

# if constraints are specified MDMM algorithm will be used
mdmm_primary_loss = 'mmd_output_target_hadflavSum'
mdmm_constraints_config = [##this can go
    ('deepjetsum_mean', 1.),
    ('deepjetsum_std', 0.001),
    # ('huber_output_target', 0.00053),
]
mdmm_constraints_config = []##better for the output CSV files
mdmm_constraints = []#As Moritz if this should be made an empty list
#[my_mdmm.EqConstraint(loss_fns[c[0]], c[1]) for c in mdmm_constraints_config]

# if no constraints are specified no MDMM is used and these loss scales are used
nomdmm_loss_scales = {

    'mmdfixsigma_output_target': 1.,
    'mmd_output_target': 0,##Ask Moritz if this would be right to have changed
    'mse_output_target': 0.,
    'mse_input_output': 0.,
    'mae_output_target': 0.,
    'mae_input_output': 0.,
    'huber_output_target': 0.,
    'huber_input_output': 0.,
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

    if epoch%verbosity==0: verbose = True
    else: verbose = False
    
    if verbose:
        print('\n# epoch {}'.format(epoch + 1))

    model.train()

    epoch_train_loss = 0.
    epoch_validation_loss = 0.

    if epoch == 0:

        if verbose:
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
                    if any(log10maskWithParameters):
                        inp = log10Transform(mask=log10maskWithParameters, factor=log10factor, onnxcompatible=False, eps=epsilon, tiny=tiny).forward(inp)
                    if any(log10maskWithoutParameters):
                        target = log10Transform(mask=log10maskWithoutParameters, factor=log10factor, onnxcompatible=False, eps=epsilon, tiny=tiny).forward(target)
                        out = log10Transform(mask=log10maskWithoutParameters, factor=log10factor, onnxcompatible=False, eps=epsilon, tiny=tiny).forward(out)                        
                    if any(fishermaskWithParameters):
                        inp = FisherTransform(mask=fishermaskWithParameters, factor=fisherfactor, onnxcompatible=False, eps=epsilon).forward(inp)
                    if any(fishermaskWithoutParameters):
                        target = FisherTransform(mask=fishermaskWithoutParameters, factor=fisherfactor, onnxcompatible=False, eps=epsilon).forward(target)
                        out = FisherTransform(mask=fishermaskWithoutParameters, factor=fisherfactor, onnxcompatible=False, eps=epsilon).forward(out)                        

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

        

    if verbose:
        print('\ntraining loop')

    trainer.zero_grad()
    for batch, (inp, target, _) in enumerate(train_loader):

        if verbose:
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
            if any(log10maskWithParameters):
                inp = log10Transform(mask=log10maskWithParameters, factor=log10factor, onnxcompatible=False, eps=epsilon, tiny=tiny).forward(inp)
            if any(log10maskWithoutParameters):
                target = log10Transform(mask=log10maskWithoutParameters, factor=log10factor, onnxcompatible=False, eps=epsilon, tiny=tiny).forward(target)
                out = log10Transform(mask=log10maskWithoutParameters, factor=log10factor, onnxcompatible=False, eps=epsilon, tiny=tiny).forward(out)                
            if any(fishermaskWithParameters):
                inp = FisherTransform(mask=fishermaskWithParameters, factor=fisherfactor, onnxcompatible=False, eps=epsilon).forward(inp)
            if any(fishermaskWithoutParameters):
                target = FisherTransform(mask=fishermaskWithoutParameters, factor=fisherfactor, onnxcompatible=False, eps=epsilon).forward(target)
                out = FisherTransform(mask=fishermaskWithoutParameters, factor=fisherfactor, onnxcompatible=False, eps=epsilon).forward(out)                

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

        if verbose:
            print('train loss {:.5f}'.format(train_loss.item()))

        iteration += 1

    print('[{} / {}] train loss: {:.10f}'.format(epoch + 1, num_epochs, epoch_train_loss / len_train_loader))

    if verbose:
        print('\nvalidation loop')
        
    model.eval()
    with torch.no_grad():

        for batch, (inp, target, _) in enumerate(validation_loader):

            if verbose:
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
                if any(log10maskWithParameters):
                    inp = log10Transform(mask=log10maskWithParameters, factor=log10factor, onnxcompatible=False, eps=epsilon, tiny=tiny).forward(inp)
                if any(log10maskWithoutParameters):
                    target = log10Transform(mask=log10maskWithoutParameters, factor=log10factor, onnxcompatible=False, eps=epsilon, tiny=tiny).forward(target)
                    out = log10Transform(mask=log10maskWithoutParameters, factor=log10factor, onnxcompatible=False, eps=epsilon, tiny=tiny).forward(out)                    
                if any(fishermaskWithParameters):
                    inp = FisherTransform(mask=fishermaskWithParameters, factor=fisherfactor, onnxcompatible=False, eps=epsilon).forward(inp)
                if any(fishermaskWithoutParameters):
                    target = FisherTransform(mask=fishermaskWithoutParameters, factor=fisherfactor, onnxcompatible=False, eps=epsilon).forward(target)
                    out = FisherTransform(mask=fishermaskWithoutParameters, factor=fisherfactor, onnxcompatible=False, eps=epsilon).forward(out)                    

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

            if verbose:
                print('validation loss {:.5f}'.format(validation_loss.item()))

        print('[{} / {}] validation loss: {:.10f}'.format(epoch + 1, num_epochs, epoch_train_loss / len_train_loader))


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
print('saving model')

m = torch.jit.script(model)
outfilename = out_path.replace('output', 'model').replace('.root', '.pt')
print('in', outfilename)
torch.jit.save(m, outfilename)
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
out_rdf.Snapshot('tMuon', out_path)
print('just created ' + out_path)

csvfile_train.close()
print('just created ' + csvfile_train.name)

csvfile_validation.close()
print('just created ' + csvfile_validation.name)
