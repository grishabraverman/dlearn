import logging
import sys

from lstmlib.loggersetup import setup_logging
from lstmlib.lstmmodelcreator import LstmModelCreator
from lstmlib.runparams import RunParams
from runlstm import parse_data


def main():
    #makeForecast.py params.txt BT_instances_forecast_2.1_short.csv model.pth result.csv 
    setup_logging("loggerConfig.json")
    log = logging.getLogger(__name__)
    log.info("Starting LSTM model creation")
    if len(sys.argv) != 5:
        log.error('Wrong number of arguments: ' + str(len(sys.argv)))
        return
    params_file = sys.argv[1]
    forecast_file = sys.argv[2]
    model_file = sys.argv[3]
    out_file = sys.argv[4]
    params = RunParams.from_file(params_file)
    trained_lstm = LstmModelCreator(params)
    if not trained_lstm.create_from_pretrained_model(model_file):
        log.error("Error creating model from " + model_file)
        return
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