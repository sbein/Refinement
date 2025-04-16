import uproot
import awkward as ak
import numpy as np
import torch
from torch import nn
import os

class SimpleClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),            
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

MODEL_PATH = "fastsim_vs_fullsim_model.pt"
INPUT_FILE = "/data/dust/user/wolfmor/Refinement/littletree_CMSSW_14_0_12_T1ttttRun3PU_step2_SIM_RECOBEFMIX_DIGI_L1_DIGI2RAW_L1Reco_RECO_PAT_NANO_PU_coffea_PUPPI.root"
TREE_NAME = "tJet"
FEATURES = [
    "RecJet_pt",
    "RecJet_btagDeepFlavB",
    "RecJet_btagDeepFlavCvB",
    "RecJet_btagDeepFlavCvL",
    "RecJet_btagDeepFlavQG",
    "RecJet_btagUParTAK4B",
    "RecJet_btagUParTAK4CvB",
    "RecJet_btagUParTAK4CvL",
    "RecJet_btagUParTAK4QvG",
]

fast_cols = [f + "_FastSim" for f in FEATURES]
full_cols = [f + "_FullSim" for f in FEATURES]
all_input_cols = fast_cols + full_cols

model = SimpleClassifier(len(FEATURES))
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

with uproot.open(INPUT_FILE) as f:
    tree = f[TREE_NAME]
    full_array = tree.arrays(library="ak")
    input_array = tree.arrays(all_input_cols, library="np")

def evaluate_model(input_dict):
    x = np.stack([input_dict[col] for col in input_dict], axis=1)
    with torch.no_grad():
        tensor = torch.tensor(x, dtype=torch.float32)
        preds = model(tensor).squeeze().numpy()
    return preds

fast_inputs = {f: input_array[f + "_FastSim"] for f in FEATURES}
full_inputs = {f: input_array[f + "_FullSim"] for f in FEATURES}

print("Evaluating classifier on FastSim and FullSim jets...")
scores_fast = evaluate_model(fast_inputs)
scores_full = evaluate_model(full_inputs)

full_array["RecJet_ffClassifier_FastSim"] = ak.Array(scores_fast)
full_array["RecJet_ffClassifier_FullSim"] = ak.Array(scores_full)

output_file = os.path.splitext(os.path.basename(INPUT_FILE))[0] + "_withClassifier.root"
with uproot.recreate(output_file) as fout:
    fout[TREE_NAME] = full_array

print("Just wrote classifier scores to", output_file)
