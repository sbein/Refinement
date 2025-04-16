import uproot
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

'''
source /cvmfs/sft.cern.ch/lcg/views/LCG_106_cuda/x86_64-el9-gcc11-opt/setup.sh
to write terminal output to a txt file:
nohup python -u trainClassifier.py 2>&1 | tee classify_log.txt &
'''

ndataset = 250000
nepochs = 250
FILE = "/data/dust/user/wolfmor/Refinement/littletree_CMSSW_14_0_12_T1ttttRun3PU_step2_SIM_RECOBEFMIX_DIGI_L1_DIGI2RAW_L1Reco_RECO_PAT_NANO_PU_coffea_PUPPI.root"
TREE = "tJet"

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

def load_and_prepare(n_max=10000):
    tree = uproot.open(FILE)[TREE]
    fast_cols = [f + "_FastSim" for f in FEATURES]
    full_cols = [f + "_FullSim" for f in FEATURES]
    df = tree.arrays(fast_cols + full_cols, library="pd", entry_stop=n_max).dropna()
    df_fast = df[fast_cols].copy()
    df_fast.columns = FEATURES
    df_fast["label"] = 1
    df_full = df[full_cols].copy()
    df_full.columns = FEATURES
    df_full["label"] = 0
    df_all = pd.concat([df_fast, df_full], ignore_index=True)
    return df_all

print('About to load and prepare')
df_all = load_and_prepare(ndataset)

class JetDataset(Dataset):
    def __init__(self, dataframe):
        self.X = dataframe[FEATURES].values.astype(np.float32)
        self.y = dataframe["label"].values.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

dataset = JetDataset(df_all)
print('len(dataset)', len(dataset))
train_size = int(0.5 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=512)

#simple 2-layer model
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

model = SimpleClassifier(len(FEATURES))
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

#training loop
train_losses, val_losses, val_rocs = [], [], []

print('made it this far')

for epoch in range(nepochs):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        pred = model(xb).squeeze()
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    avg_train_loss = total_loss / len(train_loader)
    
    #validation
    model.eval()
    with torch.no_grad():
        preds, targets = [], []
        for xb, yb in val_loader:
            out = model(xb).squeeze()
            preds.append(out.numpy())
            targets.append(yb.numpy())
        preds = np.concatenate(preds)
        targets = np.concatenate(targets)
        val_loss = criterion(torch.tensor(preds), torch.tensor(targets)).item()
        auc = roc_auc_score(targets, preds)

    train_losses.append(avg_train_loss)
    val_losses.append(val_loss)
    val_rocs.append(auc)
    print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f} | Val Loss = {val_loss:.4f} | AUC = {auc:.4f}")

#Plots - loss
plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Validation Loss")
plt.savefig("loss_evolution_torch.png")
plt.close()

# ROC 
fpr, tpr, _ = roc_curve(targets, preds)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {val_rocs[-1]:.3f}")
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Curve")
plt.legend()
plt.savefig("roc_curve_torch.png")
plt.close()

# histogram overlay
train_preds, train_targets = [], []
model.eval()
with torch.no_grad():
    for xb, yb in train_loader:
        out = model(xb).squeeze()
        train_preds.append(out.numpy())
        train_targets.append(yb.numpy())
train_preds = np.concatenate(train_preds)
train_targets = np.concatenate(train_targets)

bins = np.linspace(0.0, 1.0, 101)
bin_centers = 0.5 * (bins[:-1] + bins[1:])

def hist_data_with_errors(values, n_total):
    counts, _ = np.histogram(values, bins=bins)
    density = counts / n_total
    errors = np.sqrt(counts) / n_total
    return density, errors

train_preds, train_targets = [], []
model.eval()
with torch.no_grad():
    for xb, yb in train_loader:
        out = model(xb).squeeze()
        train_preds.append(out.numpy())
        train_targets.append(yb.numpy())
train_preds = np.concatenate(train_preds)
train_targets = np.concatenate(train_targets)

n_fast_train = np.sum(train_targets == 1)
n_full_train = np.sum(train_targets == 0)
n_fast_test = np.sum(targets == 1)
n_full_test = np.sum(targets == 0)

fastsim_test_density, fastsim_test_err = hist_data_with_errors(preds[targets == 1], n_fast_test)
fullsim_test_density, fullsim_test_err = hist_data_with_errors(preds[targets == 0], n_full_test)

plt.figure(figsize=(8, 6))

plt.hist(train_preds[train_targets == 1], bins=bins, weights=np.ones(n_fast_train)/n_fast_train,
         histtype='step', label="FastSim (train)", color="blue", linewidth=1.5)
plt.hist(train_preds[train_targets == 0], bins=bins, weights=np.ones(n_full_train)/n_full_train,
         histtype='step', label="FullSim (train)", color="red", linewidth=1.5)

plt.errorbar(bin_centers, fastsim_test_density, yerr=fastsim_test_err, fmt='o', color="blue", label="FastSim (test)", markersize=4)
plt.errorbar(bin_centers, fullsim_test_density, yerr=fullsim_test_err, fmt='o', color="red", label="FullSim (test)", markersize=4)

plt.xlabel("Classifier Output")
plt.ylabel("Fraction of Jets")
plt.yscale("log")
plt.title("Classifier Output: FastSim vs FullSim (Train & Test)")
plt.legend()
plt.tight_layout()
plt.savefig("classifier_output_hist_train_test_log.png")
plt.close()


torch.save(model.state_dict(), "fastsim_vs_fullsim_model.pt")
print("Model saved as fastsim_vs_fullsim_model.pt")

# import torch.onnx
# dummy_input = torch.randn(1, len(FEATURES))
# torch.onnx.export(model, dummy_input, "fastsim_vs_fullsim.onnx", input_names=["input"], output_names=["score"], dynamic_axes={"input": {0: "batch_size"}})
