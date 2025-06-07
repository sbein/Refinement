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

        self.epochs = config.epochs
        self.batch_size = config.trainingSettings.batchSize
        self.learning_rate = config.trainingSettings.learningRate
        
        optimizer_name = config.trainingSettings.optimizerName
        self.prepare_optimizer(optimizer_name)

        self.preprocessor = scalers.get_input_scalers()
        self.model = refinement_model_builder.build()
        self.postprocessor = scalers.get_target_inverse_scalers()
        self.postprocessor_inverse = scalers.get_target_scalers()

    def prepare_optimizer(self, optimizer_name: str):
        if optimizer_name == "Adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif optimizer_name == "SGD":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        else:
            raise ValueError("Unknown optimizer name")
        self.optimizer = optimizer


    def train(self):
        for epoch in range(self.epochs):
            for batch, (inp, target, spectators) in enumerate(self.dataset.train):
                inputs = self.preprocessor(inputs)
                targets = self.postprocessor_inverse(targets)
                outputs = self.model(inputs)
                loss = self.losses.calculate_loss(outputs, targets)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()