import torch
from torch import nn

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config import Config

from modules import model_modules
from collections import OrderedDict

class RefinementModelBuilder():

    def __init__(self, config: 'Config'):
        self.config = config.modelSettings
        # dataset info

        self.inputDim = config.datasetInfo.inputDim
        self.targetDim = config.datasetInfo.targetDim
        self.train_numbatch = config.datasetInfo.train_numbatch

        # model config
        self.onnxCompatible = config.modelSettings.onnxCompatible
        self.numSkipBlocks = config.modelSettings.numSkipBlocks
        self.numLayersPerSkipBlock = config.modelSettings.numLayersPerSkipBlock
        self.nodesHiddenLayer = config.modelSettings.nodesHiddenLayer
        self.dropout = config.modelSettings.dropout
        self.castTo16bit = config.modelSettings.castTo16bit
        self.addDeepJetConstraintLayer = config.modelSettings.addDeepJetConstraintLayer # TODO: implement

    def build(self):

        nodes_hidden_layer_list = [self.nodesHiddenLayer for _ in range(self.numSkipBlocks)]
        skipindices = [idx for idx in range(self.inputDim)] 

        self.model_dict = OrderedDict()
        
        self.model_dict['LinearWithSkipConnection_0'] = model_modules.LinearWithSkipConnection(in_features=self.inputDim,
                                                                        out_features=nodes_hidden_layer_list[0] if self.numSkipBlocks > 1 else self.targetDim,
                                                                        hidden_features=None if self.numSkipBlocks > 1 else nodes_hidden_layer_list[0],
                                                                        n_params=int(self.inputDim-self.targetDim),
                                                                        n_vars=self.targetDim,
                                                                        skipindices=skipindices,
                                                                        nskiplayers=self.numLayersPerSkipBlock,
                                                                        dropout=self.dropout,
                                                                        isfirst=True,
                                                                        islast=self.numSkipBlocks == 1)

        for k in range(self.numSkipBlocks - 1):

            if k == self.numSkipBlocks - 2:
                self.model_dict['LinearWithSkipConnection_last'] = model_modules.LinearWithSkipConnection(in_features=nodes_hidden_layer_list[k],
                                                                                        out_features=self.targetDim,
                                                                                        n_params=int(self.inputDim-self.targetDim),
                                                                                        n_vars=self.targetDim,
                                                                                        skipindices=skipindices,
                                                                                        nskiplayers=self.numLayersPerSkipBlock,
                                                                                        dropout=self.dropout,
                                                                                        islast=True)
            else:
                self.model_dict['LinearWithSkipConnection_' + str(k+1)] =  model_modules.LinearWithSkipConnection(in_features=nodes_hidden_layer_list[k],
                                                                                                out_features=nodes_hidden_layer_list[k+1],
                                                                                                n_params=int(self.inputDim-self.targetDim),
                                                                                                n_vars=self.targetDim,
                                                                                                skipindices=skipindices,
                                                                                                nskiplayers=self.numLayersPerSkipBlock,
                                                                                                dropout=self.dropout)

        self.model = nn.Sequential(self.model_dict)
        return self.model


    def show_architecture(self, output_path = None, model = None, input_size = None, depth = 3):
        if (model is None) and (not hasattr(self, 'model')):
            print("Before creating model architecture, you should build a model.")
        
        if model is None:
            model = self.model

        if input_size is None:
            input_size = (self.train_numbatch,self.inputDim)
        try:
            from torchview import draw_graph
            
            model_graph = draw_graph(model, input_size=input_size, depth = depth, expand_nested=True).visual_graph

            if output_path is None:
                return model_graph
            else:
                output_path = output_path + 'model_architecture'
                model_graph.render(output_path,format='pdf',cleanup=True)
                return None
        except ImportError as e:
            print('torchview not installed, skipping architecture saving')
        except Exception as e:
            print('Error creating architecture graph')
            print(e)

    def bind_models(self, model_dict:dict):
        return nn.Sequential(OrderedDict(model_dict))