import logging
import sys

from lstmlib.loggersetup import setup_logging
from lstmlib.lstmmodelcreator import LstmModelCreator
from lstmlib.runparams import RunParams

def parse_data(data_path):
    with open(data_path) as f:
        lines = f.readlines()
    features = []
    time_stamps = []
    train_values = []
    for line in lines[1:]:
        data_line_str = line.split('\n')[0].split(',')
        time_stamps.append(data_line_str[0])
        train_values.append(float(data_line_str[1]))
        data_line = []
        for field in data_line_str[2:]:
            data_line.append(float(field))
        features.append(data_line)
    return time_stamps, train_values, features

def main():
    #runlstm.py params.txt BT_instances_model_short.csv BT_instances_forecast_2.1_short.csv out.csv
    setup_logging("loggerConfig.json")
    log = logging.getLogger(__name__)
    log.info("Starting LSTM model creation")
    if len(sys.argv) != 5:
        log.error('Wrong number of arguments: ' + str(len(sys.argv)))
        return
    params_file = sys.argv[1]
    data_file = sys.argv[2]
    forecast_file = sys.argv[3]
    out_file = sys.argv[4]
    params = RunParams.from_file(params_file)
    time_stamps, train_values, features = parse_data(data_file)
    lstm = LstmModelCreator(params)
    if lstm.prepare_data(time_stamps, train_values, features) < 0:
        return
    lstm.init_lstm()
    lstm.train_model()
    lstm.save_model('/home/bmi/model.pth')
    log.info('==================Finished==================')
    trained_lstm = LstmModelCreator(params)
    if not trained_lstm.create_from_pretrained_model('/home/bmi/model.pth'):
        return
    # if trained_lstm.prepare_data(time_stamps, train_values, features) < 0:
    #     return
    # lstm.train_model()
    time_stamps, values, features = parse_data(forecast_file)
    if len(values) != len(features) or len(values) != len(time_stamps) or len(time_stamps) != (params.predict_len + params.recursion_len):
        log.error("Wrong number of values and time stamps")
        return
    actual_values = values[-params.predict_len:]
    actual_time_stamps = time_stamps[-params.predict_len:]
    last_value = values[params.recursion_len - 1]
    for i in range(params.recursion_len, len(values)):
        values[i] = last_value
    res = trained_lstm.forecast(values, features)
    if len(res) != params.predict_len:
        log.error("Wrong number of values in prediction")
        return
    with open(out_file, "w") as f:
        f.write("Time, Actual, Predicted\n")
        for i in range(len(actual_time_stamps)):
            f.write(str(actual_time_stamps[i]) + "," + str(actual_values[i]) + "," + str(res[i]) + "\n")
    return

if __name__ == "__main__":
    main()