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

        train_loader, validation_loader, test_loader = DataLoader(config).create_dataloader()
        
        self.train = train_loader
        self.validation = validation_loader
        self.test = test_loader
    
        self.len_train = len(train_loader)
        self.len_validation = len(validation_loader)
        self.len_test = len(test_loader)
    

    def to(self, device):
        self.train = self.train.to(device)
        self.validation = self.validation.to(device)
        self.test = self.test.to(device)
