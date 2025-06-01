import collections
from collections import OrderedDict
import inspect

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config import Config

import scaler_modules

from torch import nn


SCALER_CONFIG = {
    "tanh" : {
        "scaler" : scaler_modules.TanhScaler,
        "inverse_scaler" : scaler_modules.TanhInverseScaler,
    },
    "log" : {
        "scaler" : scaler_modules.LogScaler,
        "inverse_scaler" : scaler_modules.LogInverseScaler,
    },
    "logit" : {
        "scaler" : scaler_modules.LogitScaler,
        "inverse_scaler" : scaler_modules.LogitInverseScaler,
    }
}

class Scalers:

    def __init__(self, config: "Config"):
        self.input_features_dict = config.features.input_features
        self.target_features_dict = config.features.target_features
        self.config = config.scalers

        raw_input_sequence = Scalers._get_raw_optimized_scaler_sequence(self.input_features_dict)
        raw_target_sequence = Scalers._get_raw_optimized_scaler_sequence(self.target_features_dict)

        input_scalers_dict = self._create_scaler_instances(raw_input_sequence, 'scaler')
        target_scalers_dict = self._create_scaler_instances(raw_target_sequence, 'scaler')

        input_inverse_scalers_dict = self._create_scaler_instances(raw_input_sequence[::-1], 'inverse_scaler')
        target_inverse_scalers_dict = self._create_scaler_instances(raw_target_sequence[::-1], 'inverse_scaler')


        self.input_scaler_model = nn.Sequential(input_scalers_dict)
        self.target_scaler_model = nn.Sequential(target_scalers_dict)
        self.input_inverse_scaler_model = nn.Sequential(input_inverse_scalers_dict)
        self.target_inverse_scaler_model = nn.Sequential(target_inverse_scalers_dict)

    @staticmethod
    def _get_raw_optimized_scaler_sequence(feature_dict):

        feature_names = list(feature_dict.keys())
        num_total_features = len(feature_names)

        current_transform_indices = {fname: 0 for fname in feature_names}
        
        num_total_transforms_for_feature = {
            fname: len(transforms) for fname, transforms in feature_dict.items()
        }

        active_features = set()
        for fname in feature_names:
            if num_total_transforms_for_feature[fname] > 0:
                active_features.add(fname)

        result_sequence = []

        while active_features:
            pending_ops_this_step = collections.defaultdict(list)
            
            for i, fname in enumerate(feature_names):
                if fname in active_features: 
                    if current_transform_indices[fname] < num_total_transforms_for_feature[fname]:
                        transform_name = feature_dict[fname][current_transform_indices[fname]]
                        pending_ops_this_step[transform_name].append(i)
            
            if not pending_ops_this_step:
                break 

            best_transform_type = ""
            max_covered_features = -1

            sorted_transform_types = sorted(pending_ops_this_step.keys())

            for transform_type in sorted_transform_types:
                if len(pending_ops_this_step[transform_type]) > max_covered_features:
                    max_covered_features = len(pending_ops_this_step[transform_type])
                    best_transform_type = transform_type
            
            mask = [0] * num_total_features
            features_processed_in_this_op = []

            for feature_idx in pending_ops_this_step[best_transform_type]:
                mask[feature_idx] = 1
                fname_processed = feature_names[feature_idx]
                features_processed_in_this_op.append(fname_processed)
                current_transform_indices[fname_processed] += 1

            result_sequence.append({best_transform_type: mask})

            features_to_remove_from_active = set()
            for fname_processed in features_processed_in_this_op:
                if current_transform_indices[fname_processed] >= num_total_transforms_for_feature[fname_processed]:
                    features_to_remove_from_active.add(fname_processed)
            
            active_features.difference_update(features_to_remove_from_active)
            
        return result_sequence

    def _create_scaler_instances(self, raw_sequence, process_type: str):
        instances = OrderedDict()
        for i, op_dict in enumerate(raw_sequence):
            scaler_name, mask = list(op_dict.items())[0]
            
            ScalerClass = self.get_scaler_module(scaler_name, process_type)
            
            scaler_json_params = self.config.get(scaler_name, {})
            
            sig = inspect.signature(ScalerClass.__init__)
            valid_params_for_class = {p for p in sig.parameters if p != 'self'}

            constructor_args = {k: v for k, v in scaler_json_params.items() 
                                if k in valid_params_for_class}
            
            if 'mask' in valid_params_for_class:
                constructor_args['mask'] = mask
            elif process_type == 'scaler' or process_type == 'inverse_scaler':
                constructor_args['mask'] = mask

            instance_key = f"{scaler_name}_{i+1}"
            try:
                instances[instance_key] = ScalerClass(**constructor_args)
            except TypeError as e:
                raise TypeError(f"Error instantiating {ScalerClass.__name__} with key {instance_key} and args {constructor_args}. Original error: {e}")
        return instances

    def get_scaler_module(self, scaler_name_from_sequence: str, process_type: str):
        specific_scaler_config = self.config.get(scaler_name_from_sequence)
        if not specific_scaler_config:
            raise ValueError(f"Configuration for scaler '{scaler_name_from_sequence}' not found in config.json scalers section.")

        generic_type = specific_scaler_config.get("type")
        if not generic_type:
            raise ValueError(f"'type' not specified for scaler '{scaler_name_from_sequence}' in config.json.")

        module_details = SCALER_CONFIG.get(generic_type)
        if not module_details:
            raise ValueError(f"Scaler type '{generic_type}' (derived from '{scaler_name_from_sequence}') not found in SCALER_CONFIG.")

        ModuleClass = module_details.get(process_type)
        if not ModuleClass:
            raise ValueError(f"'{process_type}' class not defined for type '{generic_type}' in SCALER_CONFIG.")

        return ModuleClass

    
    def get_input_scalers(self):
        return self.input_scalers
    def get_inv_input_scalers(self):
        return self.inverse_input_scalers
    def get_target_scalers(self):
        return self.target_scalers
    def get_inv_target_scalers(self):
        return self.inverse_target_scalers