import pandas as pd
from tools.preprocessing import *  
from tools.entropy_based_discretization import entropy_discretization


def discretize(data, target, type, threshold = 0.5, bin_nb = 5):
    numeric_data = data.select_dtypes('float')
    if type == "UNS":
        disc_data = KMeans_discretisation(numeric_data)
        return disc_data
    else :
        numeric_target_data = pd.concat([numeric_data, data[target]], axis=1)
        disc_data = entropy_discretization(numeric_target_data, target)
        return  disc_data
    
def build_discretized_attributes(disc_data, p_type):
    new_names_train = {}
    for col in disc_data.columns:
        new_names_train[col] = col+'_DISC_'+ p_type
    disc_data.rename(columns=new_names_train, inplace=True)
    return disc_data
