import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tools.training import *
from tools.preprocessing import * 
from tools.cleaning import *
import sys
import os
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

load_dotenv()

def preprocess_main(data, unuseful_attributes, target, db_name, target_values = None, attributes_for_manual_encoding = None, values_for_manual_encoding = None  ):
    directory='outputs/'+db_name+'/preprocessed_sets/'
    os.makedirs(directory, exist_ok=True)
    
    # if cost_attributes is not None and cost_attributes != []:
    #     cost_data = data[cost_attributes]
    #     with open(directory+'/cost_data', 'wb') as file:
    #         pickle.dump(cost_data, file)
    
    ########### DELETION OF UNUSEFUL ATTRIBUTES #########
    del_un_data = delete_attribute(data, unuseful_attributes)
        
    ############ TARGET ANALYSIS ################
    if target_values is not None and target_values != {}:
        del_un_data[target] = del_un_data[target].map(target_values)
        del_un_data = del_un_data[del_un_data[target].isin(['0','1'])]

    ############# MISSING VALUES ANALYSIS ###############
    unnan_data = del_un_data[del_un_data.columns[(del_un_data.isnull().sum() / del_un_data.shape[0])*100 < 80]]
    print(f'The number of duplicated rows is : {unnan_data.duplicated().sum()} \n')
    unnan_data = unnan_data.drop_duplicates()
    unnan_data = fill_nan_attribute(unnan_data)

    ############# STANDARDIZATION  ###############
    int_attributes = unnan_data.select_dtypes('int').columns
    int_float_data = convert_int_to_float(unnan_data, int_attributes)
    quantitative_attributes = int_float_data.select_dtypes('float').columns.tolist()
    stan_norm_data = standardization(int_float_data, quantitative_attributes, 0.25, 0.75)
    partial_preprocessed_data = stan_norm_data.copy()

        ############# ENCODING #################  
    boolean_values = ast.literal_eval(os.getenv('BOOLEAN_VALUES') )    
    stan_norm_data = bool_encoder(stan_norm_data, boolean_values )
    stan_norm_data[target]  = stan_norm_data[target].astype('float')

          ######### ORDINAL ENCODING #######   
    if attributes_for_manual_encoding is not None and values_for_manual_encoding is not None:
         stan_norm_data = manual_encoder(stan_norm_data, [attributes_for_manual_encoding, values_for_manual_encoding])
            ####### ONE HOT ENCODING #######
        
    columns_for_one_hot_encoding = stan_norm_data.select_dtypes(include=['object']).columns.tolist()
    data_one_hot_encoded = one_hot_encoder(stan_norm_data,columns_for_one_hot_encoding)
    data_preprocessed = pd.concat([stan_norm_data, data_one_hot_encoded ], axis=1)
    data_preprocessed = data_preprocessed.drop(columns_for_one_hot_encoding, axis=1)

    # partial_preprocessed_data = partial_preprocessed_data.sample(50)
    # index_t = partial_preprocessed_data.index
    # data_preprocessed= data_preprocessed.loc[index_t]

    data_preprocessed.to_csv(directory+'/preprocessed_data.csv')
    partial_preprocessed_data.to_csv(directory+'/partial_preprocessed_data.csv')

    trainset, testset = train_test_split(data_preprocessed, test_size=0.2, random_state=42, shuffle=True)
    directory = 'outputs/' + db_name + '/data/classic/'
    os.makedirs(directory, exist_ok=True)
    trainset.to_csv(directory + 'trainset.csv')
    testset.to_csv(directory + 'testset.csv')

if __name__== "__main__":
    # sys.excepthook = my_exception_hook
    args = sys.argv[1:]
    db_name = args[0]
    path = args[1]
    target = args[2]
    unuseful_attributes = args[3]
    target_values = args[4]
    unuseful_attributes = ast.literal_eval(unuseful_attributes)
    target_values = ast.literal_eval(target_values)
    data = pd.read_csv(path, low_memory=False)


    if len(args) > 6:
        attributes_for_manual_encoding = args[5]
        values_for_manual_encoding = args[6]

        attributes_for_manual_encoding = ast.literal_eval(attributes_for_manual_encoding)
        values_for_manual_encoding = ast.literal_eval(values_for_manual_encoding)
        preprocess_main(data, unuseful_attributes, target,  db_name, target_values, attributes_for_manual_encoding, values_for_manual_encoding)
    else:
        preprocess_main(data, unuseful_attributes, target,  db_name, target_values, None, None)



    
    
    
 