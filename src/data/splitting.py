import ast
from sklearn.model_selection import train_test_split
import os, sys
import pandas as pd

from ..tools.cleaning import delete_attribute
from ..tools.preprocessing import fill_nan_attribute


if __name__== "__main__":
    args = sys.argv[1:]
    db_name = args[0]
    path = args[1]
    target = args[2]
    unuseful_attributes = args[3]
    target_values = args[4]
    unuseful_attributes = ast.literal_eval(unuseful_attributes)
    target_values = ast.literal_eval(target_values)

    data = pd.read_csv(path, low_memory=False, na_values=['?'])
    
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
    
    
    
    unnan_data[target]  = unnan_data[target].astype('object')
    
    
    trainset, testset = train_test_split(unnan_data, test_size=0.2, random_state=42, shuffle=True) 
    
    # print(trainset.isnull().sum())
    # exit()
       
    directory = 'data/raw/train/'
    os.makedirs(directory, exist_ok=True)
    trainset.to_csv(directory + 'trainset_'+ db_name +'.csv', index=False)

    directory = 'data/raw/test/'
    os.makedirs(directory, exist_ok=True)
    testset.to_csv(directory + 'testset_'+ db_name +'.csv', index=False)
    