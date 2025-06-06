#source /cvmfs/sft.cern.ch/lcg/views/LCG_106_cuda/x86_64-el9-gcc11-opt/setup.sh

from config import Config
from data_loader import Dataset
from scalers import Scalers
from model import RefinementModelBuilder

config = Config(config_path='config.json')
dataset = Dataset(config)

scalers = Scalers(config)
refinement_model_builder = RefinementModelBuilder(config = config)
dataset.print_summary()

model_dict = {
    "input_scaler" : scalers.get_input_scalers(),
    "refinement_model" : refinement_model_builder.build(),
    "target_inverse_scaler" : scalers.get_target_inverse_scalers(),
}

model = refinement_model_builder.bind_models(model_dict = model_dict)

refinement_model_builder.show_architecture(model = model, depth = 3, output_path = "/afs/cern.ch/user/d/dboncukc/Refinement_monitor/model.pdf")