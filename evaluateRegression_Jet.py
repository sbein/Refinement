"""
to load packages, e.g.:
source /cvmfs/sft.cern.ch/lcg/views/LCG_104cuda/x86_64-el9-gcc11-opt/setup.sh
"""

import ROOT
import numpy as np

import torch
torch.set_printoptions(edgeitems=5, linewidth=160)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


'''
# ##### ##### #####
# general settings
# ##### ##### #####
'''

ntest = 0
batch_size = 4096

'''
# ##### ##### ##### #####
# define input/output
# ##### ##### ##### #####
'''

# the input file is expected to contain a tree filled with jet triplets: RecJet_x_FastSim, RecJet_x_FullSim, GenJet_y,...
in_path = '/nfs/dust/cms/user/wolfmor/Refinement/littletree_CMSSW_14_0_12_T1ttttRun3PU_step2_SIM_RECOBEFMIX_DIGI_L1_DIGI2RAW_L1Reco_RECO_PAT_NANO_PU_coffea_new_test.root'
in_tree = 'tJet'
preselection = '1'  # 'GenJet_nearest_dR>0.5&&RecJet_nearest_dR_FastSim>0.5&&RecJet_nearest_dR_FullSim>0.5'

model_path = '/nfs/dust/cms/user/wolfmor/Refinement/TrainingOutput/tanhNorm1200_SB3_LP2_NHL1024_LR1eM04_PTtanhlogit_retrain2_with_dataset4.pt'

out_path = model_path.replace('.pt', '_eval.root')

'''
# ##### ##### ##### ##### ##### ##### #####
# define variables and transformations
# ##### ##### ##### ##### ##### ##### #####
'''

onehotencode = ('RecJet_hadronFlavour_FastSim', [0, 4, 5])  # add one-hot-encoding for given input variable with the given values
# onehotencode = False

PARAMETERS = [
    ('GenJet_pt', ['log']),
    ('GenJet_eta', []),
    ('RecJet_hadronFlavour_FastSim', [])
]

# if using DeepJetConstraint the DeepJet transformations have to be explicitly adapted in the DeepJetConstraint module
VARIABLES = [
    ('RecJet_pt_CLASS', ['log']),
    ('RecJet_btagDeepFlavB_CLASS', ['logit']),
    ('RecJet_btagDeepFlavCvB_CLASS', ['logit']),
    ('RecJet_btagDeepFlavCvL_CLASS', ['logit']),
    ('RecJet_btagDeepFlavQG_CLASS', ['logit'])
]

spectators = [
    'EventID',
    'RecJet_event_MET_pt_FastSim',
    'RecJet_event_MET_pt_FullSim',
    'RecJet_event_MET_phi_FastSim',
    'RecJet_event_MET_phi_FullSim',
    'RecJet_eta_FullSim',
    'RecJet_eta_FastSim',
    'RecJet_phi_FullSim',
    'RecJet_phi_FastSim',
    'GenJet_pt',
    'GenJet_eta',
    'GenJet_phi',
    'GenJet_mass',
    'GenJet_hadronFlavour',
    'GenJet_partonFlavour'
]
excludespectators = [var[0] for var in PARAMETERS + VARIABLES]
SPECTATORS = [
    s for s in spectators if s not in excludespectators
    and s.replace('CLASS', 'FastSim') not in excludespectators
    and s.replace('CLASS', 'FullSim') not in excludespectators
]


'''
# ##### ##### #####
# get data
# ##### ##### #####
'''
print('\n### get data')

print('ntest', ntest)
if ntest == 0:
    rdf = ROOT.RDataFrame(in_tree, in_path).Filter(preselection)
else:
    rdf = ROOT.RDataFrame(in_tree, in_path).Filter(preselection).Range(ntest)

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
test_dataset = dataset


def collate_fn(batch_):
    batch_ = list(filter(lambda x: torch.all(torch.isfinite(torch.cat(x))), batch_))
    return torch.utils.data.dataloader.default_collate(batch_)


test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

len_test_loader = len(test_loader)

model = torch.jit.load(model_path)

model.to(device)

print(model)
print(sum(p.numel() for p in model.parameters() if p.requires_grad), 'trainable parameters')


'''
# ##### ##### #####
# save output
# ##### ##### #####
'''
print('\n### save output')

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

out_rdf.Snapshot('tJet', out_path)
print('just created ' + out_path)
