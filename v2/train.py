import torch
import torch.nn as nn
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config import Config
    from data_loader import Dataset
    from loss import LossManager
    from model import RefinementModelBuilder
    from scalers import Scalers

class Trainer:
    def __init__(self, 
    config: "Config", 
    dataset: "Dataset", 
    losses: "LossManager", 
    refinement_model_builder: "RefinementModelBuilder",
    scalers: "Scalers",
    ):
        self.config = config
        self.dataset = dataset
        self.losses = losses

        self.epochs = config.trainingSettings.epochs
        self.batch_size = config.trainingSettings.batchSize
        self.learning_rate = config.trainingSettings.learningRate
        
        config_device = config.trainingSettings.device
        if config_device == 'auto' or config_device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(config_device)
        
        optimizer_name = config.trainingSettings.optimizerName
        
        self.model = None
        self.optimizer = None

        self.preprocessor = scalers.get_input_scalers()
        self.refinement_model_builder = refinement_model_builder
        self.postprocessor = scalers.get_target_inverse_scalers()
        self.postprocessor_inverse = scalers.get_target_scalers()
        
        model = refinement_model_builder.build()
        self.model = model.to(self.device)
        self.prepare_optimizer(optimizer_name)

    def prepare_optimizer(self, optimizer_name: str):
        if optimizer_name.lower() == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif optimizer_name.lower() == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Unknown optimizer name: {optimizer_name}")


    def train(self):
                
        print(f"Starting training for {self.epochs} epochs on {self.device}")
        print("-" * 60)
        
        for epoch in range(self.epochs):

            self.model.train()
            train_loss = 0.0
            train_samples = 0
            
            for batch_idx, (inp, target, spectators) in enumerate(self.dataset.train):
                
                inp = inp.to(self.device)
                target = target.to(self.device)

                inputs = self.preprocessor(inp)
                targets = self.postprocessor_inverse(target)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                
                self.losses.calculate(inputs, outputs, targets, epoch, batch_idx, mode='train')
                primary_loss = self.losses.get_primary_loss_fn()(inputs, outputs, targets)
                
                primary_loss.backward()
                self.optimizer.step()
                
                train_loss += primary_loss.item() * inputs.size(0)
                train_samples += inputs.size(0)
            
            avg_train_loss = train_loss / train_samples
            
            val_loss = self.evaluate()
            
            print(f"Epoch {epoch+1:4d}/{self.epochs} | "
                  f"Train Loss: {avg_train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f}")
        
        print("-" * 60)
        print("Training completed!")
        
        return self.model
    
    def evaluate(self):
        
        self.model.eval()
        val_loss = 0.0
        val_samples = 0
        
        with torch.no_grad():
            for batch_idx, (inp, target, spectators) in enumerate(self.dataset.validation):
                
                inp = inp.to(self.device)
                target = target.to(self.device)

                inputs = self.preprocessor(inp)
                targets = self.postprocessor_inverse(target)
                
                outputs = self.model(inputs)
                
                self.losses.calculate(inputs, outputs, targets, 0, batch_idx, mode='validation')
                primary_loss = self.losses.get_primary_loss_fn()(inputs, outputs, targets)
                
                val_loss += primary_loss.item() * inputs.size(0)
                val_samples += inputs.size(0)
        
        return val_loss / val_samples
    
    def test(self):
        
        self.model.eval()
        test_loss = 0.0
        test_samples = 0
        
        print("Running test evaluation...")
        
        with torch.no_grad():
            for batch_idx, (inp, target, spectators) in enumerate(self.dataset.test):
                
                inp = inp.to(self.device)
                target = target.to(self.device)

                inputs = self.preprocessor(inp)
                targets = self.postprocessor_inverse(target)
                
                outputs = self.model(inputs)
                
                self.losses.calculate(inputs, outputs, targets, 0, batch_idx, mode='test')
                primary_loss = self.losses.get_primary_loss_fn()(inputs, outputs, targets)
                
                test_loss += primary_loss.item() * inputs.size(0)
                test_samples += inputs.size(0)
        
        avg_test_loss = test_loss / test_samples
        print(f"Test Loss: {avg_test_loss:.6f}")
        
        return avg_test_loss


    def save_results(self, output_path: str):
        m = torch.jit.script(self.model)
        torch.jit.save(m, output_path + "model.pt")
        print(f"Model saved to {output_path + 'model.pt'}")
        self.dataset.save_root(self.model, "tJet", output_path + "data.root")
        print(f"Data saved to {output_path +'data.root'}")