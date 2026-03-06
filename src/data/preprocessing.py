import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tools.training import *
from tools.preprocessing import * 
from tools.cleaning import *
import sys
import os
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

load_dotenv()

def preprocess_main(data, target, db_name, attributes_for_manual_encoding = None, values_for_manual_encoding = None  ):
   
    ############# STANDARDIZATION  ###############
    int_attributes = data.select_dtypes('int').columns
    numerical_data = convert_int_to_float(data, int_attributes)
    numerical_attributes = numerical_data.select_dtypes('float').columns.tolist()
    
    for col in numerical_attributes:
        with open("engine/preprocessing/" + db_name+ "/" + col + "_params_stan.txt", "w") as file:
            parameters = ast.literal_eval(file.read())
            
        numerical_data = standardization(parameters['cmax'], parameters['sup'], parameters['inf'], parameters['iqr'], numerical_data, col)
    partial_preprocessed_data = numerical_data.copy()

        ############# ENCODING #################  
    boolean_values = ast.literal_eval(os.getenv('BOOLEAN_VALUES') )    
    numerical_data = bool_encoder(numerical_data, boolean_values )
    numerical_data[target]  = numerical_data[target].astype('float')

          ######### ORDINAL ENCODING #######   
    if attributes_for_manual_encoding is not None and values_for_manual_encoding is not None:
         numerical_data = manual_encoder(numerical_data, [attributes_for_manual_encoding, values_for_manual_encoding])
         
          ####### ONE HOT ENCODING #######
        
    columns_for_one_hot_encoding = numerical_data.select_dtypes(include=['object']).columns.tolist()
    
    with open("engine/preprocessing/" + db_name+ "/one_hot_encoder_engine", "rb") as file:
        encoder = pickle.load(file) 
         
    one_hot_encoded_data = encoder.transform(numerical_data[columns_for_one_hot_encoding])
    
    preprocessed_data = pd.concat([numerical_data, one_hot_encoded_data], axis=1)
    preprocessed_data = preprocessed_data.drop(columns_for_one_hot_encoding, axis=1)

    # partial_preprocessed_data = partial_preprocessed_data.sample(50)
    # index_t = partial_preprocessed_data.index
    # data_preprocessed= data_preprocessed.loc[index_t]
    
    directory='data/preprocessed/'+ db_name +'/'
    os.makedirs(directory, exist_ok=True)
    data_preprocessed.to_csv(directory+'/preprocessed_data.csv')
    partial_preprocessed_data.to_csv(directory+'/partial_preprocessed_data.csv')

if __name__== "__main__":
    args = sys.argv[1:]
    db_name = args[0]
    path = args[1]
    target = args[2]
    unuseful_attributes = args[3]
    # target_values = args[4]
    unuseful_attributes = ast.literal_eval(unuseful_attributes)
    # target_values = ast.literal_eval(target_values)
    data = pd.read_csv(path, low_memory=False)


    if len(args) > 6:
        attributes_for_manual_encoding = args[5]
        values_for_manual_encoding = args[6]

        attributes_for_manual_encoding = ast.literal_eval(attributes_for_manual_encoding)
        values_for_manual_encoding = ast.literal_eval(values_for_manual_encoding)
        preprocess_main(data, target,  db_name, attributes_for_manual_encoding, values_for_manual_encoding)
    else:
        preprocess_main(data, target,  db_name, None, None)



    
    
    
 