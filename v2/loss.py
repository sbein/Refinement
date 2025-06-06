import torch
from torch import nn

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config import Config

from modules import loss_modules

@dataclass
class LogEntry:
    value: float
    epoch: int

@dataclass
class LossLog:
    train: dict[str, list[LogEntry]]
    validation : dict[str, list[LogEntry]]

class LossManager:

    def __init__(self, config: "Config"):
        self.config = config.losses
        self.numParameters = len(config.features.parameters if config.features.parameters else {})
        self.loss_log = LossLog({}, {})
        self._loss_classes = {}
        self.loss_funcs = {}
        self.epoch = None

        for loss_name, loss_config in self.config.items():
            self.init_loss_fn(loss_name, loss_config['type'], loss_config.get('initParams', {}), loss_config.get('forwardParams', {}))

    def set_epoch(self, epoch):
        self.epoch = epoch
  
    def init_loss_fn(self, loss_name:str, loss_type:str, init_params: dict, forward_params: dict):
        fn = loss_modules.get(loss_type)

        if fn is None:
            raise ValueError(f"Loss function {loss_type} not found")
        
        self._loss_classes[loss_name] = fn(**init_params)

        def loss_fn_params(_input, _output, _target, forward_params: dict):

            loss_fn_input = None
            loss_fn_target = None
            
            forward_param_input = forward_params.get('input', '$OUTPUT$')
            forward_param_target = forward_params.get('target', '$TARGET$')

            if forward_param_input == '$OUTPUT$':
                loss_fn_input = _output
            elif forward_param_input == '$PARAMETERS$':
                loss_fn_input = _input[:, : self.numParameters]
            elif forward_param_input == '$VARIABLES$':
                loss_fn_input = _input[:, self.numParameters :]
            elif forward_param_input == '$TARGET$':
                loss_fn_input = _target
            else:
                raise ValueError(f"Invalid forward_param_input: {forward_param_input}")

            if forward_param_target == '$OUTPUT$':
                loss_fn_target = _output
            elif forward_param_target == '$PARAMETERS$':
                loss_fn_target = _input[:, : self.numParameters]
            elif forward_param_target == '$VARIABLES$':
                loss_fn_target = _input[:, self.numParameters :]
            elif forward_param_target == '$TARGET$':
                loss_fn_target = _target
            else:
                raise ValueError(f"Invalid forward_param_target: {forward_param_target}")

            return loss_fn_input, loss_fn_target
        
        self.loss_funcs[loss_name] = lambda input, output, target : self._loss_classes[loss_name](*loss_fn_params(input, output, target, forward_params))

        self.loss_log.train[loss_name] = []
        self.loss_log.validation[loss_name] = []

    def calculate(self, input, output, target, epoch:int, is_val:bool):
        self.set_epoch(epoch)
        for name in self.loss_funcs.keys():
            self._calculate_one(name, is_val, input, output, target)        

    def _calculate_one(self, name:str, is_val:bool, input, output, target):
        value = self.loss_funcs[name](input, output, target)
        
        if is_val:
            if self.epoch is None:
                epoch = len(self.loss_log.validation[name])
            else:
                epoch = self.epoch
            self.loss_log.validation[name].append(LogEntry(value, epoch))
        else:
            if self.epoch is None:
                epoch = len(self.loss_log.train[name])
            else:
                epoch = self.epoch
            self.loss_log.train[name].append(LogEntry(value, epoch))
        return value
    
    def get_loss_fn(self, name: str):
        return self.loss_funcs[name]

    def get_primary_loss(self):
        return self.get_loss_fn('primary')