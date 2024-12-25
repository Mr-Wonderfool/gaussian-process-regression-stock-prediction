import yaml
import numpy as np

class Config():
    def __init__(self, yaml_path, ):
        self.path = yaml_path
        self.config = self._load_yaml()
    def _load_yaml(self, ):
        with open(self.path, 'r') as file:
            return yaml.safe_load(file)
    def query(self, class_name, key):
        return self.config[class_name][key]

def company_data(data_, company, start_time, end_time):
    """ extract specific column containing company name """
    column_list = ['Date']
    column_list.append(company)
    specific_data = data_[column_list]
    return specific_data[(specific_data['Date'] > start_time) 
        & (specific_data['Date'] < end_time)]
    
def data_processing(data, seq_length):
    X_train, y_train = [], []
    for i in range(seq_length, len(data)):
        # X_train contain groups of #seq_length numbers
        X_train.append(data[i-seq_length:i, 0])
        # y_train contains the data right after X_train
        y_train.append(data[i, 0])
    return np.array(X_train), np.array(y_train)