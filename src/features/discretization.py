import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sys, os, pickle

def main(data, label, discretization_type, db_name):
    
    directory = "engine/discretization/" + db_name+ "/"
    os.makedirs(directory, exist_ok=True)
    numeric_data = data.select_dtypes('float')
    
    
    if discretization_type == "UNS":
        discretized_data = pd.DataFrame()
        for col in numeric_data.columns:
            
            att = data[col].values
            att = att.reshape(-1, 1)    
            
            with open(directory + 'kmeans_engine_'+ col , 'rb') as file:
                kmeans_engine = pickle.load(file) 
            discretized_data[col] = kmeans_engine['engine'].predict(att).astype("int").astype("object")    
            
        new_data = data.drop(columns=numeric_data.columns)
        data = pd.concat([new_data, discretized_data], axis=1)    
            
        directory = "data/discretized/" + db_name+ "/"
        os.makedirs(directory, exist_ok=True)  
        data.to_csv(directory + 'discretized_' + label + '_data_' + discretization_type.lower() + '.csv')  
         
    else:
        os.makedirs(directory, exist_ok=True)
        with open(directory + 'tree_discretization_engine' , 'rb') as file:
           tree_engine = pickle.load(file)
           
        discretized_data = tree_engine.transform(numeric_data)   
        new_data = data.drop(columns=numeric_data.columns)
        data = pd.concat([new_data, discretized_data], axis=1)
        
        directory = "data/discretized/" + db_name+ "/"
        os.makedirs(directory, exist_ok=True)  
        data.to_csv(directory + 'discretized_' + label + '_data_' + discretization_type.lower() + '.csv')

if __name__ == "__main__":
    args = sys.argv[1:]
    target = args[3]
    db_name = args[0]
    discretization_type = args[1]
    label = args[2]
    

    partial_preprocessed_data  = pd.read_csv("data/preprocessed/"+ db_name +"/partial_preprocessed_data_" + label + ".csv", keep_default_na=False)
    partial_preprocessed_data.drop(columns=['Unnamed: 0'], inplace=True)
    partial_preprocessed_data[target] = partial_preprocessed_data[target].astype("object")
    
    main(partial_preprocessed_data,  label,  discretization_type, db_name)
    












