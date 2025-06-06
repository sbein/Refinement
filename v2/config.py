import json

class ConfigDict(dict):
    def __init__(self, data:dict=None):
        if data is not None:
            self.config_dict = data
        super().__init__(self.config_dict)
    
    def __getitem__(self, key):
        if key not in self.config_dict:
            return None
        value = self.config_dict[key]
        if isinstance(value, dict):
            return ConfigDict(data = value)
        return value

    def __getattr__(self, key):
        return self.__getitem__(key)

    def __setattr__(self, key, value):
        if key == 'config_dict':
            super().__setattr__(key, value)
        else:
            if hasattr(self, 'config_dict') and isinstance(self.config_dict, dict):
                self.config_dict[key] = value
                super().__setitem__(key, value)
            else:
                super().__setattr__(key, value)

class Config(ConfigDict):
    def __init__(self, config_path:str=None, data:dict=None):
        if config_path is not None:
            config_dict = self.read_config_file(config_path)
        else:
            config_dict = data if data is not None else {}
        super().__init__(config_dict)
        self.process_config()

    def read_config_file(self, config_path):
        with open(config_path, 'r') as f:
            return json.load(f)

    def process_config(self):
        self.process_features()

    def process_features(self):

        input_features = {}
        target_features = {}
        spectator_features = []

        parameters = self.config_dict['features'].get('parameters', {})
        variables = self.config_dict['features'].get('variables', {})
        spectators = self.config_dict['features'].get('spectators', [])

        for name in set(parameters.keys()):
            feature_name = name.replace('CLASS', 'FastSim')
            input_features[feature_name] = parameters[name] if isinstance(parameters[name], list) else []
        
        for name in set(variables.keys()):
            target_name = name.replace('CLASS', 'FullSim')
            input_name = name.replace('CLASS', 'FastSim')
            target_features[target_name] = variables[name] if isinstance(variables[name], list) else []
            if input_name not in input_features:
                input_features[input_name] = variables[name] if isinstance(variables[name], list) else []

        for name in spectators:
            if "CLASS" in name:
                spectator_features.append(name.replace('CLASS', 'FastSim'))
                spectator_features.append(name.replace('CLASS', 'FullSim'))
            else:
                spectator_features.append(name)

        self.config_dict['features']['input_features'] = input_features
        self.config_dict['features']['target_features'] = target_features
        self.config_dict['features']['spectator_features'] = list(set(spectator_features))