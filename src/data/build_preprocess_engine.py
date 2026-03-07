import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from ..tools.training import *
from ..tools.preprocessing import * 
from ..tools.cleaning import *
import sys
import os
from dotenv import load_dotenv
import pickle


load_dotenv()

def preprocess_main(data, db_name, target):
    data[target]  = data[target].astype('object') 
    ############# STANDARDIZATION  ###############
    int_attributes = data.select_dtypes('int').columns
    numerical_data = convert_int_to_float(data, int_attributes)
    numerical_attributes = numerical_data.select_dtypes('float').columns.tolist()
    
    for col in numerical_attributes:
        parameters = standardization_engine(numerical_data, col, 0.25, 0.75)
        
        directory = "engine/preprocessing/" + db_name+ "/"
        os.makedirs(directory, exist_ok=True)
        with open(directory + col + '_params_stan.txt' , 'w') as file:
            file.write(str(parameters))
            
    ############# ENCODING ############### 
    categorical_data = data.select_dtypes('object').drop(columns=[target]) 
    
    if not categorical_data.empty :
       engine = one_hot_encoder(categorical_data)
       
       with open(directory + '/one_hot_encoder_engine', 'wb') as file:
        pickle.dump(engine, file)
    
          

if __name__== "__main__":
    args = sys.argv[1:]
    db_name = args[0]
    path = args[1]
    target = args[2]
    
    data = pd.read_csv(path, low_memory=False)
 
    preprocess_main(data, db_name, target)



    
    
    
 