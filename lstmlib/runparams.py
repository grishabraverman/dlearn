import logging

log = logging.getLogger(__name__)

class RunParams:
    def __init__(self, resolution_min: int, epochs: int, epochs_between_validations: int, batch_size: int, optimizer: str,
                 learning_rate: float, min_learn_rate: float, loss_min_change_perc: float,
                 factor_on_improvements: float, factor_on_divergence: float, momentum: float,
                 recursion_len: int, predict_len: int, num_lstm_layers: int, hidden_state_size: int,
                 fc_layers_arc: list, dropout: float, validation_len: int, validation_gap: int, shuffle: bool,
                 num_best_models: int, max_stuck_events: int, loss_out_file_name: str, flatten_features: bool):
        self.resolution_min = resolution_min
        self.epochs = epochs
        self.epochs_between_validations = epochs_between_validations
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.min_learn_rate = min_learn_rate
        self.loss_min_change_perc = loss_min_change_perc
        self.factor_on_improvements = factor_on_improvements
        self.factor_on_divergence = factor_on_divergence
        self.momentum = momentum
        self.optimizer = optimizer
        self.recursion_len = recursion_len
        self.predict_len = predict_len
        self.num_lstm_layers = num_lstm_layers
        self.hidden_state_size = hidden_state_size
        self.fc_layers_arc = fc_layers_arc
        self.validation_len = validation_len
        self.validation_gap = validation_gap
        self.shuffle = shuffle
        self.dropout = dropout
        self.num_best_models = num_best_models
        self.max_stuck_events = max_stuck_events
        self.loss_out_file_name = loss_out_file_name
        self.flatten_features = flatten_features
        return

    @classmethod
    def from_file(cls, file_name: str):
        with open(file_name) as f:
            lines = f.readlines()
            params = {}
        for line in lines:
            tokens = line.split('\n')
            tokens = tokens[0].split(' ')
            if tokens[0] == 'epochs':
                params['epochs'] = int(tokens[1])
            if tokens[0] == 'resolutionMin':
                    params['resolutionMin'] = int(tokens[1])
            elif tokens[0] == 'epochsBetweenValidations':
                params['epochsBetweenValidations'] = int(tokens[1])
            elif tokens[0] == 'batchSize':
                params['batchSize'] = int(tokens[1])
            elif tokens[0] == 'optimizer':
                params['optimizer'] = tokens[1]
            elif tokens[0] == 'learningRate':
                params['learningRate'] = float(tokens[1])
            elif tokens[0] == 'minLearningRate':
                params['minLearningRate'] = float(tokens[1])
            elif tokens[0] == 'lossMinChangePerc':
                params['lossMinChangePerc'] = float(tokens[1])
            elif tokens[0] == 'factorOnImprovements':
                params['factorOnImprovements'] = float(tokens[1])
            elif tokens[0] == 'factorOnDivergence':
                params['factorOnDivergence'] = float(tokens[1])
            elif tokens[0] == 'momentum':
                params['momentum'] = float(tokens[1])
            elif tokens[0] == 'recursionLen':
                params['recursionLen'] = int(tokens[1])
            elif tokens[0] == 'predictLen':
                params['predictLen'] = int(tokens[1])
            elif tokens[0] == 'validationLen':
                params['validationLen'] = int(tokens[1])
            elif tokens[0] == 'validationGap':
                params['validationGap'] = int(tokens[1])
            elif tokens[0] == 'numLSTMLayers':
                params['numLSTMLayers'] = int(tokens[1])
            elif tokens[0] == 'hiddenMemorySize':
                params['hiddenMemorySize'] = int(tokens[1])
            elif tokens[0] == 'numBestModels':
                params['numBestModels'] = int(tokens[1])
            elif tokens[0] == 'maxStuckEvents':
                params['maxStuckEvents'] = int(tokens[1])
            elif tokens[0] == 'fullConnLayersArch':
                if len(tokens) == 1:
                    params['fullConnLayersArch'] = []
                else:
                    fc_layers_arc = []
                    for i in range(1, len(tokens)):
                        fc_layers_arc.append(int(tokens[i]))
                    params['fullConnLayersArch'] = fc_layers_arc
            elif tokens[0] == 'fullConnectedLayerMemSize':
                params['fullConnectedLayerMemSize'] = int(tokens[1])
            elif tokens[0] == 'shuffle':
                params['shuffle'] = int(tokens[1]) > 0
            elif tokens[0] == 'flattenFeatures':
                params['flattenFeatures'] = int(tokens[1]) > 0
            elif tokens[0] == 'dropout':
                params['dropout'] = float(tokens[1])
            elif tokens[0] == 'lossOutFileName':
                params['lossOutFileName'] = tokens[1]
        return cls(resolution_min=params['resolutionMin'],
                   epochs=params['epochs'],
                   epochs_between_validations=params['epochsBetweenValidations'],
                   batch_size=params['batchSize'],
                   optimizer=params['optimizer'],
                   learning_rate=params['learningRate'],
                   min_learn_rate=params['minLearningRate'],
                   loss_min_change_perc=params['lossMinChangePerc'],
                   factor_on_improvements=params['factorOnImprovements'],
                   factor_on_divergence=params['factorOnDivergence'],
                   momentum=params['momentum'],
                   recursion_len=params['recursionLen'],
                   predict_len=params['predictLen'],
                   num_lstm_layers=params['numLSTMLayers'],
                   hidden_state_size=params['hiddenMemorySize'],
                   fc_layers_arc=params['fullConnLayersArch'],
                   dropout=params['dropout'],
                   validation_len=params['validationLen'],
                   validation_gap=params['validationGap'],
                   shuffle=params['shuffle'],
                   num_best_models=params['numBestModels'],
                   max_stuck_events=params['maxStuckEvents'],
                   loss_out_file_name=params['lossOutFileName'],
                   flatten_features=params['flattenFeatures'])

    def __eq__(self, other) -> bool:
        if not isinstance(other, RunParams):
            return False
        if self.recursion_len != other.recursion_len:
            return False
        if self.predict_len != other.predict_len:
            return False
        if self.num_lstm_layers != other.num_lstm_layers:
            return False
        if self.hidden_state_size != other.hidden_state_size:
            return False
        if self.validation_len != other.validation_len:
            return False
        if self.validation_gap != other.validation_gap:
            return False
        if self.flatten_features != other.flatten_features:
            return False
        return True

    def log(self):
        log.info('\tRunParams:')
        for key, value in self.__dict__.items():
            log.info('\t{}: {}'.format(key, value))
