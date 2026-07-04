import logging
import sys

from lstmlib.loggersetup import setup_logging
from lstmlib.lstmmodelcreator import LstmModelCreator
from lstmlib.runparams import RunParams
from runlstm import parse_data

def main():
    #createModel.py params.txt BT_instances_model_short.csv model.pth
    setup_logging("loggerConfig.json")
    log = logging.getLogger(__name__)
    log.info("Starting LSTM model creation")
    if len(sys.argv) != 4:
        log.error('Wrong number of arguments: ' + str(len(sys.argv)))
        return
    params_file = sys.argv[1]
    data_file = sys.argv[2]
    model_file = sys.argv[3]
    params = RunParams.from_file(params_file)
    time_stamps, train_values, features = parse_data(data_file)
    lstm = LstmModelCreator(params)
    if lstm.prepare_data(time_stamps, train_values, features) < 0:
        return
    lstm.init_lstm()
    lstm.train_model()
    lstm.save_model(model_file)
    log.info("LSTM model saved")

if __name__ == "__main__":
    main()