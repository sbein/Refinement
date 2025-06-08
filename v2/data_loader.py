import ROOT
import numpy as np
import pandas as pd
import torch

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config import Config

class DataLoader():
    def __init__(self, config:"Config"):
        
        self.read_config(config = config)
        self.load_data()
        self.extract_dicts()
        self.convert_tensor()
        self.create_dataloader()  

    def read_config(self, config:"Config"):
        self.treeName = config.inputSettings.treeName
        self.filePath = config.inputSettings.filePath
        self.preselection = config.inputSettings.preselection

        self.batchSize = config.trainingSettings.batchSize
        self.numBatches = config.trainingSettings.numBatches
        self.randomSeed = config.trainingSettings.randomSeed

        self.numTrain = self.numBatches[0] * self.batchSize
        self.numVal = self.numBatches[1] * self.batchSize
        self.numTest = self.numBatches[2] * self.batchSize

        self.variables = config.features.variables
        self.parameters = config.features.parameters
        self.spectators = config.features.spectators

        self.input_features_dict = config.features.input_features
        self.target_features_dict = config.features.target_features
        self.spectator_features = config.features.spectator_features

        self.input_features = list(self.input_features_dict.keys())
        self.target_features = list(self.target_features_dict.keys())

    def load_data(self):

        rdf = ROOT.RDataFrame(
            self.treeName, 
            self.filePath
            ).Filter(self.preselection)
        
        if self.numTest == 0:
            rdf = rdf.Range(self.numTrain + self.numVal)
        else:
            rdf = rdf.Range(self.numTrain + self.numVal + self.numTest)

        self.rdf_numpy = rdf.AsNumpy(self.input_features + self.target_features + self.spectator_features)
            
    def extract_dicts(self):
        self.dict_input = {var: self.rdf_numpy[var] for var in self.input_features}
        self.dict_target = {var: self.rdf_numpy[var] for var in self.target_features}
        self.dict_spectators = {var: self.rdf_numpy[var] for var in self.spectator_features}

    def convert_tensor(self):
        self.data_input = torch.tensor(np.stack(list(self.dict_input.values()), axis=1), dtype=torch.float32)
        self.data_target = torch.tensor(np.stack(list(self.dict_target.values()), axis=1), dtype=torch.float32)
        self.data_spectators = torch.tensor(np.stack(list(self.dict_spectators.values()), axis=1), dtype=torch.float32)

    def create_dataloader(self):

        dataset = torch.utils.data.TensorDataset(self.data_input, self.data_target, self.data_spectators)

        if self.numTest == 0:
            self.numTest = len(dataset) - (self.numTrain + self.numVal)

        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [self.numTrain, self.numVal, self.numTest],
            generator=torch.Generator().manual_seed(self.randomSeed)
        )

        def collate_fn(batch_):
            batch_ = list(filter(lambda x: torch.all(torch.isfinite(torch.cat(x))), batch_))
            return torch.utils.data.dataloader.default_collate(batch_)


        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batchSize, shuffle=True, collate_fn=collate_fn)
        validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batchSize, shuffle=False, collate_fn=collate_fn)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batchSize, shuffle=False, collate_fn=collate_fn)

        return train_loader, validation_loader, test_loader
    

class Dataset:

    def __init__(self, config:"Config"):
        
        self.config = config
        dataloader = DataLoader(config)
        train_loader, validation_loader, test_loader = dataloader.create_dataloader()
        
        self.dict_input = dataloader.dict_input        
        self.dict_target = dataloader.dict_target
        self.dict_spectators = dataloader.dict_spectators

        self.train = train_loader
        self.validation = validation_loader
        self.test = test_loader

        config.datasetInfo = {}
        config.datasetInfo.train_numbatch = len(train_loader)
        config.datasetInfo.validation_numbatch = len(validation_loader)
        config.datasetInfo.test_numbatch = len(test_loader)

        (inp_sample_tensor, target_sample_tensor, spectators_sample_tensor) = train_loader.dataset[0]

        config.datasetInfo.inputDim = inp_sample_tensor.shape[0]
        config.datasetInfo.targetDim = target_sample_tensor.shape[0]
        config.datasetInfo.spectatorsDim = spectators_sample_tensor.shape[0]

        config.datasetInfo.trainTotalSamples = len(train_loader.dataset)
        config.datasetInfo.validationTotalSamples = len(validation_loader.dataset)
        config.datasetInfo.testTotalSamples = len(test_loader.dataset)

    def print_summary(self):
        print("#"*50)
        print("Dataset summary:")
        print(f"Train: {self.config.datasetInfo.train_numbatch} batches, {self.config.datasetInfo.trainTotalSamples} samples")
        print(f"Validation: {self.config.datasetInfo.validation_numbatch} batches, {self.config.datasetInfo.validationTotalSamples} samples")
        print(f"Test: {self.config.datasetInfo.test_numbatch} batches, {self.config.datasetInfo.testTotalSamples} samples")
        print("-"*50)
        print(f"Input: {self.config.datasetInfo.inputDim} features")
        print(f"Target: {self.config.datasetInfo.targetDim} features")
        print(f"Spectators: {self.config.datasetInfo.spectatorsDim} features")
        print("#"*50)
    
    def get_summary(self):
        
        return {
            "train": {
                "numbatch": self.config.datasetInfo.train_numbatch,
                "total_samples": self.config.datasetInfo.trainTotalSamples,
            },
            "validation": {
                "numbatch": self.config.datasetInfo.validation_numbatch,
                "total_samples": self.config.datasetInfo.validationTotalSamples,
            },
            "test": {
                "numbatch": self.config.datasetInfo.test_numbatch,
                "total_samples": self.config.datasetInfo.testTotalSamples,
            },
            "input": {
                "dim": self.config.datasetInfo.inputDim,
            },
            "target": {
                "dim": self.config.datasetInfo.targetDim,
            },
            "spectators": {
                "dim": self.config.datasetInfo.spectatorsDim,
            }
        }




    def process_dataset_split(self, model, split_type):

        if split_type == 0:
            dataloader = self.train
        elif split_type == 1:
            dataloader = self.validation
        elif split_type == 2:
            dataloader = self.test
        else:
            raise ValueError("Invalid split type")

        for i, (inp, target, spectators) in enumerate(dataloader):
            out = model(inp)
            
            self.out_dict['isTrainValTest'].append(torch.ones(inp.size(dim=0), dtype=torch.int) * split_type)
            branch_list = []

            for ib, branch in enumerate(self.dict_input):
                if branch in branch_list:
                    continue
                self.out_dict[branch].append(inp[:, ib])
                branch_list.append(branch)
            for ib, branch in enumerate(self.dict_target):
                if branch in branch_list:
                    continue
                self.out_dict[branch].append(target[:, ib])
                branch_list.append(branch)
            for ib, branch in enumerate(self.dict_target):
                if branch.replace('FullSim', 'Refined') in branch_list:
                    continue
                self.out_dict[branch.replace('FullSim', 'Refined')].append(out[:, ib])
                branch_list.append(branch.replace('FullSim', 'Refined'))
            for ib, branch in enumerate(self.dict_spectators):
                if branch in branch_list:
                    continue
                self.out_dict[branch].append(spectators[:, ib])
                branch_list.append(branch)


    def save_root(self, model, treename, path):

        self.out_dict = {key: [] for key in ['isTrainValTest'] + list(self.dict_input.keys()) +  list(self.dict_target.keys()) + [branch.replace('FullSim', 'Refined') for branch in self.dict_target.keys()] +  list(self.dict_spectators.keys())}

        model.eval()
        
        with torch.no_grad():
            self.process_dataset_split(model, 0)
            
            self.process_dataset_split(model, 1)
            
            self.process_dataset_split(model, 2)
        
        for branch in self.out_dict:
            self.out_dict[branch] = torch.cat(self.out_dict[branch]).detach().cpu().numpy()
    
        out_rdf = ROOT.RDF.FromNumpy(self.out_dict)
        out_rdf.Snapshot(treename, path)