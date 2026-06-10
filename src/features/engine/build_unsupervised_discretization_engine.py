import pandas as pd
from ...tools.preprocessing import kmeans_discretization_engine
import sys, os
import pickle


if __name__ == "__main__":
    args = sys.argv[1:]
    db_name = args[0]
    target = args[1]
    path = args[2]
    dir_= args[3]
    
    directory = dir_ + db_name+ "/"
    os.makedirs(directory, exist_ok=True)
    
    partial_preprocessed_data  = pd.read_feather(path)
     
    # try:
    #     partial_preprocessed_data.drop(columns=['Unnamed: 0'], inplace = True)
    # except:
    #     print("Column 'Unnamed: 0' not existed!")    
    
    numeric_data = partial_preprocessed_data.select_dtypes('float')
    partial_preprocessed_data[target] = partial_preprocessed_data[target].astype("object")
    
    
    for col in numeric_data.columns:
        att = numeric_data[col].values
        att = att.reshape(-1, 1)
        discretization_engine = kmeans_discretization_engine(att)
        
        with open(directory + 'kmeans_engine_'+ col , 'wb') as file:
            pickle.dump(discretization_engine, file)
    
        
   

    
    
    
    