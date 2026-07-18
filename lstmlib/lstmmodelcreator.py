import copy
import logging
import math

import torch
from sympy.strategies.core import switch
from torch.utils.data import DataLoader

from lstmlib.dataUtils import convert_values_2d, is_valid
from lstmlib.lossscheduleradam import LossSchedulerAdam
from lstmlib.lossschedulersgd import LossSchedulerSGD
from lstmlib.predictiongapfillerbase import PredictionGapFillerBase
from lstmlib.predictiongapfilleruniform import PredictionGapFillerUniform
from lstmlib.runparams import RunParams
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch.nn as nn

from lstmlib.sequencedataset import SequenceDataset

log = logging.getLogger(__name__)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LstmModelCreator(nn.Module):
    def __init__(self, parameters: RunParams):
        super().__init__()
        self.parameters = parameters
        self.x_tensor_valid = None
        self.y_tensor_valid = None
        self.t_array_valid = []
        self.t_array_train = []
        self.train_loader = []
        self.train_scaler = StandardScaler()
        self.features_scaler = StandardScaler()
        self.n_features = 0
        self.lstm = None
        self.fc_stack = None
        self.last_epoch = None
        self.last_loss = None
        self.optimizer = None
        self.best_models = dict()
        return

    def prepare_data(self, time_stamps: list, train_values: list, features: list) -> int:
        log.info("Starting Data preparation")
        flatten_features = self.parameters.flatten_features
        transformed_features, transformed_train_values = self.scale_data(features, train_values)
        a_array = []
        b_array = []
        t_array = []
        n_features = len(transformed_features[0])
        invalid_chunks =0
        filler = create_prediction_gap_filler(parameters=self.parameters)
        len_no_filler = self.parameters.recursion_len - self.parameters.prediction_gap
        for i in range(len(time_stamps) - self.parameters.predict_len - self.parameters.recursion_len):
            a_mat = []
            if flatten_features:
                a_mat.append([])
            b_line = []
            last_train_value = None
            times_of_chunk = []
            for j in range(i, i + len_no_filler):
                times_of_chunk.append(time_stamps[j])
                a_line = [transformed_train_values[j]]
                for k in range(n_features):
                    a_line.append(transformed_features[j][k])
                if len(transformed_features[j]) != n_features:
                    log.error("Wrong number of features")
                    return -1
                if flatten_features:
                    a_mat[0].extend(a_line)
                else:
                    a_mat.append(a_line)
                last_train_value = transformed_train_values[j]
            for j in range(i + len_no_filler, i + self.parameters.recursion_len + self.parameters.predict_len):
                times_of_chunk.append(time_stamps[j])
                if last_train_value is None:
                    log.error("Train value is None")
                    return -1
                a_line = [filler.get(time_stamps[j], [last_train_value])]
                for k in range(n_features):
                    a_line.append(transformed_features[j][k])
                if flatten_features:
                    a_mat[0].extend(a_line)
                else:
                    a_mat.append(a_line)
                if j >= i + self.parameters.recursion_len:
                    b_line.append(transformed_train_values[j])
            if not is_valid(times_of_chunk, self.parameters.resolution_min):
                invalid_chunks += 1
                continue
            a_array.append(a_mat)
            b_array.append(b_line)
            t_array.append(time_stamps[i])
        if len(t_array) <= self.parameters.validation_len:
            log.error("Wrong number of validation samples")
            return -1
        log.info("Training data size: " + str(len(t_array)))
        log.info("invalid chunks: " + str(invalid_chunks))
        a_array_valid = a_array[-self.parameters.validation_len:]
        b_array_valid = b_array[-self.parameters.validation_len:]
        self.t_array_valid = t_array[-self.parameters.validation_len:]
        a_array_train = a_array[:len(t_array) - self.parameters.validation_len]
        b_array_train = b_array[:len(t_array) - self.parameters.validation_len]
        self.t_array_train = t_array[:len(t_array) - self.parameters.validation_len]
        a_np_valid = np.array(a_array_valid)
        b_np_valid = np.array(b_array_valid)
        a_np_train = np.array(a_array_train)
        b_np_train = np.array(b_array_train)
        self.x_tensor_valid = torch.tensor(a_np_valid, dtype=torch.float)
        self.y_tensor_valid = torch.tensor(b_np_valid, dtype=torch.float)
        x_tensor_train = torch.tensor(a_np_train, dtype=torch.float)
        y_tensor_train = torch.tensor(b_np_train, dtype=torch.float)
        self.n_features = n_features + 1
        train_dataset = SequenceDataset(x_tensor_train, y_tensor_train)
        self.train_loader = DataLoader(train_dataset, batch_size=self.parameters.batch_size, shuffle=self.parameters.shuffle)
        return 0

    def init_lstm(self) -> int:
        if self.n_features == 0 :
            log.error("Wrong number of features: " + str(self.n_features))
            return -1
        flatted_features = self.parameters.flatten_features
        drop_out = 0
        if self.parameters.num_lstm_layers > 1:
            drop_out = self.parameters.dropout
        if flatted_features:
            input_size = self.n_features * (self.parameters.recursion_len + self.parameters.predict_len)
        else:
            input_size = self.n_features
        self.lstm = nn.LSTM(input_size, self.parameters.hidden_state_size, self.parameters.num_lstm_layers,
                            batch_first=True, dropout=drop_out)
        fc_layers = []
        in_dim = self.parameters.hidden_state_size  # first dense in‑features

        for i, out_dim in enumerate(self.parameters.fc_layers_arc, 1):
            fc_layers.append(nn.Linear(in_dim, out_dim))

            # add non‑linearity + dropout after every layer except the last head layer
            fc_layers.append(nn.ReLU())
            if self.parameters.dropout > 0:
                fc_layers.append(nn.Dropout(self.parameters.dropout))

            in_dim = out_dim  # next layer’s input size

        # final projection to prediction length (no activation after this)
        fc_layers.append(nn.Linear(in_dim, self.parameters.predict_len))

        # register the whole stack as one module
        self.fc_stack = nn.Sequential(*fc_layers)
        return 0

    def train_model(self) -> float:
        self.parameters.log()
        log.info("Device: " + DEVICE.type)
        self.to(DEVICE)
        criterion = nn.MSELoss()
        if self.optimizer is None:
            self.create_optimizer()
        scheduler = self.create_scheduler()
        best_error = float('inf')
        for epoch in range(self.parameters.epochs):
            self.last_epoch = epoch
            self.train()
            epoch_loss = 0.0
            n_samples = 0
            for batch_X, batch_y in self.train_loader:
                batch_X = batch_X.to(DEVICE)
                batch_y = batch_y.to(DEVICE)
                self.optimizer.zero_grad()

                output = self.forward(batch_X)
                self.last_loss = criterion(output, batch_y)
                if not torch.isfinite(self.last_loss):
                    log.error("Non-finite loss at epoch " + str(epoch + 1))
                    return -1.0
                self.last_loss.backward()
                # Clip every trainable layer before the optimizer updates its state.
                torch.nn.utils.clip_grad_norm_(
                    list(self.lstm.parameters()) + list(self.fc_stack.parameters()),
                    max_norm=1.0,
                )
                self.optimizer.step()
                epoch_loss += self.last_loss.item() * batch_X.size(0)
                n_samples += batch_X.size(0)
            if n_samples == 0:
                log.error("No training samples were processed")
                return -1.0
            avg_loss = epoch_loss / n_samples
            self.save_loss(epoch, avg_loss)
            log.info(f"Epoch [{epoch + 1}/{self.parameters.epochs}], Loss: {avg_loss:.4f}")
            if epoch % self.parameters.epochs_between_validations == 0 and epoch != 0:
                error, mape = self.validation_step()
                if error is None:
                    return -1.0
                if not math.isfinite(error):
                    log.error("Non-finite validation error at epoch " + str(epoch + 1))
                    return -1.0
                if error < best_error:
                    best_error = error
                    best_fc_stack = copy.deepcopy(self.fc_stack)
                    best_lstm = copy.deepcopy(self.lstm)
                    best_model = [best_lstm, best_fc_stack]
                    if len(self.best_models) == self.parameters.num_best_models:
                        self.best_models = dict(sorted(self.best_models.items()))
                        self.best_models.popitem()
                    self.best_models[error] = best_model
                    log.info("Saved best model for error " + str(error))
                log.info("Validation error: "  + str(error))
                log.info("Validation MAPE: " + str(mape * 100) + "%")
                scheduler.step(error)
                early_exit = scheduler.is_stuck_finally()
                if early_exit:
                    log.info("Early exit at epoch " + str(epoch))
                    break
        error, mape = self.validation_step()
        if error is None:
            return -1.0
        log.info("Finish train with validation error: " + str(error))
        log.info("Finish train with validation MAPE: " + str(mape * 100) + "%")
        log.info("Finish train with best error: " + str(min(self.best_models)))
        return error

    def create_scheduler(self):
        if self.parameters.optimizer == 'adam':
            return LossSchedulerAdam(self.optimizer, self.parameters.factor_on_divergence,
                                     self.parameters.min_learn_rate,
                                     self.parameters.loss_min_change_perc,
                                     self.parameters.max_stuck_events)
        elif self.parameters.optimizer == 'grad':
            return LossSchedulerSGD(self.optimizer, self.parameters.factor_on_improvements,
                                     self.parameters.factor_on_divergence,
                                     self.parameters.min_learn_rate, self.parameters.learning_rate,
                                     self.parameters.loss_min_change_perc,
                                     self.parameters.max_stuck_events)
        return None

    def create_optimizer(self):
        all_params = list(self.lstm.parameters()) + list(self.fc_stack.parameters())
        if self.parameters.optimizer == 'adam':
            self.optimizer = torch.optim.AdamW(
                all_params, lr=self.parameters.learning_rate, weight_decay=1e-4
            )
        elif self.parameters.optimizer == 'grad':
            self.optimizer = torch.optim.SGD(all_params, self.parameters.learning_rate, self.parameters.momentum)
        else:
            raise ValueError("Unsupported optimizer: " + str(self.parameters.optimizer))

    def validation_step(self):
        self.eval()
        val_output = self.forward(self.x_tensor_valid.to(DEVICE))
        val_actual, val_forecast = self.back_transform_values_arrays(val_output)
        if len(val_forecast) != len(self.t_array_valid):
            log.error("Wrong number of validation samples")
            return None, None
        error = 0.0
        mape = 0.0
        n_eval = 0
        for i in range(len(val_forecast)):
            for j in range(len(val_forecast[i])):
                actual = val_actual[i][j]
                forecast = val_forecast[i][j]
                error += (forecast - actual) * (forecast - actual)
                if actual != 0.0:
                    mape += math.fabs(forecast - actual) / math.fabs(actual)
                n_eval += 1
        if n_eval == 0:
            log.error("No validation samples after applying validation gap")
            return None, None
        error = error / n_eval
        mape = mape / n_eval
        error = math.sqrt(error)
        return error, mape

    def back_transform_values_arrays(self, val_output):
        val_forecast = self.train_scaler.inverse_transform(val_output.to('cpu').detach())
        val_actual = self.train_scaler.inverse_transform(self.y_tensor_valid)
        return val_actual, val_forecast

    def scale_data(self, features, train_values):
        train_val_2d = convert_values_2d(train_values)
        self.train_scaler.fit(train_val_2d)
        transformed_train_2d = self.train_scaler.transform(train_val_2d)
        self.features_scaler.fit(features)
        transformed_features = self.features_scaler.transform(features)
        transformed_train_values = []
        for val in transformed_train_2d:
            transformed_train_values.append(val[0])
        return transformed_features, transformed_train_values

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)  # h_n: (num_layers, batch, hidden)
        last_hidden = h_n[-1]  # take last layer’s hidden state fc_in = out[:, -1, :]
        # last_hidden = out[:, -1, :]
        return self.fc_stack(last_hidden)  # output shape: (batch, predict_len)

    def save_model(self, file_name: str):
        torch.save({
            'epoch': self.last_epoch,
            'optimizer': self.optimizer,
            'loss': self.last_loss,
            'lstm': self.lstm,
            'full_conn_layer': self.fc_stack,
            'best_models': self.best_models,
            'n_features': self.n_features,
            'train_scaler': self.train_scaler,
            'features_scaler': self.features_scaler
        }, file_name)

    def create_from_pretrained_model(self, file_name:str) -> bool:
        checkpoint = torch.load(file_name, weights_only = False)
        self.optimizer = checkpoint['optimizer']
        log.info("Last loss: " + str(checkpoint['loss']))
        log.info("Last epoch: " + str(checkpoint['epoch']))
        self.train_scaler = checkpoint['train_scaler']
        self.features_scaler = checkpoint['features_scaler']
        self.n_features = checkpoint['n_features']
        self.lstm = checkpoint['lstm']
        self.fc_stack = checkpoint['full_conn_layer']
        self.best_models = checkpoint['best_models']
        return True

    def forecast(self, values: list, features: list) -> list:
        if len(features) != len(values):
            log.error("Wrong number of features")
            return []
        flatten_features = self.parameters.flatten_features
        transformed_features = self.features_scaler.transform(features)
        val_2d = convert_values_2d(values)
        transformed_values = self.train_scaler.transform(val_2d)
        #add consumption
        if flatten_features:
            features_final = [[]]
        else:
            features_final = []
        for i in range(len(transformed_features)):
            line = [transformed_values[i][0]]
            line.extend(transformed_features[i])
            if flatten_features:
                features_final[0].extend(line)
            else:
                features_final.append(line)
        features_final = torch.tensor([features_final], dtype=torch.float)
        self.eval()
        curr_lstm = self.lstm
        curr_fc_stack = self.fc_stack
        sum_errors = 0
        forecast = [0] * self.parameters.predict_len
        for err in self.best_models.keys():
            model = self.best_models[err]
            self.lstm = model[0]
            self.fc_stack = model[1]
            val_output = self.forward(features_final.to(DEVICE)).to('cpu').detach()
            forecast_model = self.train_scaler.inverse_transform(val_output)[0]
            for i in range(self.parameters.predict_len):
                forecast[i] = forecast_model[i] * err + forecast[i]
            sum_errors += err
        for i in range(len(forecast)):
            forecast[i] = forecast[i] / sum_errors
        self.lstm = curr_lstm
        self.fc_stack = curr_fc_stack
        return forecast

    def save_loss(self, epoch: int, avg_loss: float):
        if self.parameters.loss_out_file_name == 'null':
            return
        if epoch == 0:
            with open(self.parameters.loss_out_file_name, "w") as f:
                f.write("Epoch,Loss\n")
        with open(self.parameters.loss_out_file_name, "a") as f:
            f.write(str(epoch) + "," + str(avg_loss) + "\n")
        return

def create_prediction_gap_filler(parameters: RunParams):
    match parameters.prediction_gap_filler:
        case "uniform":
            return PredictionGapFillerUniform()
    return None
