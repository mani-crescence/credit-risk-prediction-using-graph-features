import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sys, os, pickle

def main(data, label, db_name, normalization_label):
    
    directory = "engine/" + normalization_label + "/discretization/" + db_name+ "/"
    os.makedirs(directory, exist_ok=True)
    numeric_data = data.select_dtypes('float')
    
    
    os.makedirs(directory, exist_ok=True)
    with open(directory + 'tree_discretization_engine' , 'rb') as file:
        tree_engine = pickle.load(file)
        
    discretized_data = tree_engine.transform(numeric_data)   
    new_data = data.drop(columns=numeric_data.columns)
    data = pd.concat([new_data, discretized_data], axis=1)
    
    
    directory = "data/" + normalization_label + "/discretized/" + db_name+ "/"
    os.makedirs(directory, exist_ok=True)  
    data.to_feather(directory + 'discretized_' + label + '_data_sup.feather')

if __name__ == "__main__":
    args = sys.argv[1:]    
    db_name = args[0]
    target = args[1]
    path = args[2]
    type_of_normalization = args[3]
    type_of_set = args[4]

    partial_preprocessed_data  = pd.read_feather(path)
    
    partial_preprocessed_data[target] = partial_preprocessed_data[target].astype("object")
    
    main(partial_preprocessed_data, type_of_set, db_name, type_of_normalization)
    












