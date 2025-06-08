#source /cvmfs/sft.cern.ch/lcg/views/LCG_106_cuda/x86_64-el9-gcc11-opt/setup.sh

from config import Config
from data_loader import Dataset
from scalers import Scalers
from model import RefinementModelBuilder
from loss import LossManager
from train import Trainer

def main():

    config = Config(config_path='config.json')
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
        output_path="model.pdf"
    )
    
    print("\nStarting training...")
    trained_model = trainer.train()
    
    print("\nEvaluating on test set...")
    test_loss = trainer.test()
    
    loss_manager.save_log("training_logs.csv")
    
    print("\nTraining completed successfully!")
    print(f"Final test loss: {test_loss:.6f}")
    
    return trained_model, trainer

if __name__ == "__main__":
    model, trainer = main()

