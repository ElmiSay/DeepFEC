

import yaml
import os


class Config(object):
    def __init__(self, config_path):
        self.config_name = os.path.basename(config_path).split('.')[0]
        #base = 
        #self.config_name = os.path.splitext(config_path)[0]
        f = open(config_path, "r")
        conf = yaml.load(f)
        self.data_path = conf["data_path"]
        self.observe_length = conf["observe_length"]
        self.predict_length = conf["predict_length"]
        self.model_path = conf["model_path"]
        self.test_ratio = conf["test_ratio"]
        self.time_window = conf["time_window"]
        self.batch_size = conf["batch_size"]
        self.epochs = conf["epochs"]
        self.early_stopping = conf["early_stopping"]
        self.learning_rate = conf["learning_rate"]
        self.use_cache_model = conf["use_cache_model"]
        self.use_loopup = conf["use_loopup"]
        self.model_name = conf["model_name"]
        self.start_hour = conf["start_hour"]
        self.end_hour = conf["end_hour"]
        self.use_externel = conf["use_externel"]
        self.use_matrix_fuse = conf["use_matrix_fuse"]
        if "observe_p" in conf:
            self.observe_p = conf["observe_p"]
        else:
            self.observe_p = 0
        if "observe_t" in conf:
            self.observe_t = conf["observe_t"]
        else:
            self.observe_t = 0
