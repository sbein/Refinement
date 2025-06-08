import torch
from torch import nn
import pandas as pd
import os

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config import Config

from modules import loss_modules

@dataclass
class LogEntry:
    value: float
    epoch: int
    batch: int

@dataclass
class LossLog:
    train: dict[str, list[LogEntry]]
    validation : dict[str, list[LogEntry]]
    test: dict[str, list[LogEntry]]

class LossManager:

    def __init__(self, config: "Config"):
        self.config = config.losses
        self.numParameters = len(config.features.parameters if config.features.parameters else {})
        self.loss_log = LossLog({}, {}, {})
        self._loss_classes = {}
        self.loss_funcs = {}
        self.epoch = None
        self.primary_loss = None
        for loss_name, loss_config in self.config.items():

            if loss_config.get("isPrimary",False):
                if self.primary_loss is not None:
                    raise ValueError("You can only select one primary loss function.")
                self.primary_loss = loss_name

            self.init_loss_fn(loss_name, loss_config['type'], loss_config.get('initParams', {}), loss_config.get('forwardParams', {}))

        if self.primary_loss is None:
            raise ValueError("No primary loss function defined. You have to define one for training.")

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
        self.loss_log.test[loss_name] = []

    def calculate(self, input, output, target, epoch:int, batch:int, mode:str = 'train'):
        self.set_epoch(epoch)
        for name in self.loss_funcs.keys():
            self._calculate_one(name, mode, input, output, target, batch)        

    def _calculate_one(self, name:str, mode:str, input, output, target, batch:int):
        value = self.loss_funcs[name](input, output, target)
        value = value.item()
        
        if self.epoch is None:
            if mode == 'test':
                epoch = len(self.loss_log.test[name])
            elif mode == 'validation':
                epoch = len(self.loss_log.validation[name])
            else:
                epoch = len(self.loss_log.train[name])
        else:
            epoch = self.epoch
        
        if mode == 'test':
            self.loss_log.test[name].append(LogEntry(value, epoch, batch))
        elif mode == 'validation':
            self.loss_log.validation[name].append(LogEntry(value, epoch, batch))
        else:
            self.loss_log.train[name].append(LogEntry(value, epoch, batch))
        
        return value
    
    def get_loss_fn(self, name: str):
        return self.loss_funcs[name]

    def get_primary_loss_fn(self):
        return self.get_loss_fn(self.primary_loss)
    
    def get_primary_loss(self, mode: int, epoch:int, batch:int):

        if mode == "test":
            log_entries = self.loss_log.test[self.primary_loss]
        elif mode == "validation":
            log_entries = self.loss_log.validation[self.primary_loss]
        else:
            log_entries = self.loss_log.train[self.primary_loss]

        for entry in log_entries:
            if entry.epoch == epoch and entry.batch == batch:
                return entry.value

        return None

    def get_epoch_average(self, loss_name: str, epoch: int, is_val: bool = False) -> float:

        log_entries = self.loss_log.validation[loss_name] if is_val else self.loss_log.train[loss_name]
        epoch_values = [entry.value for entry in log_entries if entry.epoch == epoch]
        
        if not epoch_values:
            raise ValueError(f"No data found for loss '{loss_name}' at epoch {epoch}")
        
        return sum(epoch_values) / len(epoch_values)
    
    def save_log(self, output_path: str):

        output_path  = output_path + "loss_logs.csv"
        
        data = []
        
        for loss_name, log_entries in self.loss_log.train.items():
            for entry in log_entries:
                data.append({
                    'loss_name': loss_name,
                    'type': 'train',
                    'epoch': entry.epoch,
                    'batch': entry.batch,
                    'value': entry.value
                })
        
        for loss_name, log_entries in self.loss_log.validation.items():
            for entry in log_entries:
                data.append({
                    'loss_name': loss_name,
                    'type': 'validation',
                    'epoch': entry.epoch,
                    'batch': entry.batch,
                    'value': entry.value
                })
        
        for loss_name, log_entries in self.loss_log.test.items():
            for entry in log_entries:
                data.append({
                    'loss_name': loss_name,
                    'type': 'test',
                    'epoch': entry.epoch,
                    'batch': entry.batch,
                    'value': entry.value
                })
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        print(f"Loss logs saved to {output_path}")