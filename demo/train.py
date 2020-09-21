from jsonml import start
import pandas as pd
import sys


if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    pd.set_option('max_colwidth', 500)

    date = '' if len(sys.argv) < 3 else sys.argv[2]

    config_path = "train_config.json"
    if len(sys.argv) >= 2 and "test" == sys.argv[1]:
        config_path = "test_config.json"
    elif len(sys.argv) >= 2 and "predict" == sys.argv[1]:
        config_path = "predict_config.json"
    elif len(sys.argv) >= 2:
        config_path = sys.argv[1]

    start.start(config_path, date=date)

