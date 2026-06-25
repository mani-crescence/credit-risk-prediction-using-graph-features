import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sys, os, pickle
import numpy as np

def main(data, label, db_name, normalization_label, sub):
    
    directory = "engine/" + normalization_label + "/discretization/" + db_name + "/sub" + sub + "/"
    os.makedirs(directory, exist_ok=True)
    numeric_data = data.select_dtypes('float')
    discretized_data = pd.DataFrame(index=data.index)
    
    
    for col in numeric_data.columns:
        with open(directory + 'mdlp_engine_' + col , 'rb') as file:
            mdlp_engine = pickle.load(file)
        
        splits = mdlp_engine.splits 
        bins = np.digitize(numeric_data[col], splits)
        discretized_data[col] = bins
    
    new_data = data.drop(columns=numeric_data.columns)
    data = pd.concat([new_data, discretized_data], axis=1)
    
    
    directory = "data/" + normalization_label + "/discretized/" + db_name + "/sub" + sub + "/"
    os.makedirs(directory, exist_ok=True)  
    data.to_feather(directory + 'discretized_' + label + '_data_sup.feather')

if __name__ == "__main__":
    args = sys.argv[1:]    
    db_name = args[0]
    target = args[1]
    path = args[2]
    type_of_normalization = args[3]
    type_of_set = args[4]
    sub = args[5]

    partial_preprocessed_data  = pd.read_feather(path)
    
    partial_preprocessed_data[target] = partial_preprocessed_data[target].astype("object")
    
    main(partial_preprocessed_data, type_of_set, db_name, type_of_normalization, sub)
    












