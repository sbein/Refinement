#source /cvmfs/sft.cern.ch/lcg/views/LCG_106_cuda/x86_64-el9-gcc11-opt/setup.sh

from config import Config
from data_loader import Dataset
from scalers import Scalers
from model import RefinementModelBuilder
from loss import LossManager
from train import Trainer

import os

def main():

    config = Config(config_path='config.json')

    training_id = config.generalSettings.trainingId

    training_id = training_id if training_id else config.generalSettings.trainingName

    if training_id is None:
        raise ValueError("Training ID or Training Name must be provided.")

    grid_id = config.generalSettings.gridId

    grid_id = grid_id if grid_id else ''
    
    storeFolder = config.outputSettings.storeFolder

    if storeFolder is None:
        raise ValueError("Store Folder path must be provided.")
    
    if storeFolder[-1] != '/':
        storeFolder += '/'

    output_path = f"{storeFolder}{grid_id}/{training_id}/"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    dataset = Dataset(config)
    scalers = Scalers(config)
    refinement_model_builder = RefinementModelBuilder(config=config)
    loss_manager = LossManager(config=config)
    
    dataset.print_summary()
    
    trainer = Trainer(
        config=config,
        dataset=dataset,
        losses=loss_manager,
        refinement_model_builder=refinement_model_builder,
        scalers=scalers
    )

    refinement_model_builder.show_architecture(
        model=trainer.model, 
        depth=3, 
        output_path=output_path
    )
    
    print("\nStarting training...")
    trained_model = trainer.train()
    
    print("\nEvaluating on test set...")
    test_loss = trainer.test()
    
    print("\nTraining completed successfully!")
    print(f"Final test loss: {test_loss:.6f}")

    loss_manager.save_log(output_path = output_path)
    trainer.save_results(output_path = output_path)
    
    return trained_model, trainer

if __name__ == "__main__":
    model, trainer = main()

