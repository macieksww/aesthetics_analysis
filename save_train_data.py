import csv
import pandas as pd

class data_saver:
    def __init__(self):

        self.model_params_fieldnames = ['id', 'optimizer', 'loss', 'metrics', 'epochs', 'batch_size']
        self.layers_neurons_fieldnames = ['id', 'IL', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'OL']
        self.act_fun_fieldnames = ['id', 'IL', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'OL']

        self.model_params_path="/home/bdroix/bdroix/aesthetics_analysis/dane do analizy/model_params.csv"
        self.layers_neurons_path="/home/bdroix/bdroix/aesthetics_analysis/dane do analizy/layers_neurons.csv"
        self.act_fun_path="/home/bdroix/bdroix/aesthetics_analysis/dane do analizy/activation_functions.csv"


    def save_data(self, data_type, data):

        if data_type=="model_params":
            path=self.model_params_path
            fieldnames=self.model_params_fieldnames
        
        elif data_type=="act_fun_params":
            path=self.act_fun_path
            fieldnames=self.act_fun_fieldnames

        else:
            path=self.layers_neurons_path
            fieldnames=self.layers_neurons_fieldnames

        print("PATH: "+path)
        df = pd.read_csv(path, delim_whitespace=True)

        line_count = 0
        with open(path, 'r', encoding='UTF8', newline='') as f:
            line_count = list(enumerate(f))[-1][0]
            data = [str(line_count)] + data
            # print("NUM OF LINES")
            # print(line_count)


        if df.empty:
            with open(path, 'w', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(fieldnames)
                writer.writerow(data)
        else:
            with open(path, 'a', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(data)


    # def prepare_data(self, data_type, data):
    #     if data_type=="model_params":
    #         fieldnames=self.model_params_fieldnames
        
    #     elif data_type=="act_fun_params":
    #         fieldnames=self.act_fun_fieldnames

    #     else:
    #         fieldnames=self.layers_neurons_fieldnames
        
        # data_dict={}

        # for fieldname in fieldnames:

a = data_saver()