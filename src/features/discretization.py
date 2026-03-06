import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from features.numeric_attributes_discretization import *
import sys, os

def main(partial_preprocessed_data, target, train_index, test_index, ptype, db_name):
    
    discretized_data = discretize(partial_preprocessed_data, target, ptype)
    
    
    
    discretized_data_train = discretized_data.loc[train_index]
    discretized_data_test = discretized_data.loc[test_index]
    
    directory='outputs/'+ db_name +'/data/discretized/graph'
    os.makedirs(directory, exist_ok=True)
    discretized_data_train.to_csv(directory + '/discretized_train_' + ptype.lower() + '.csv')
    discretized_data_test.to_csv(directory + '/discretized_test_' + ptype.lower() + '.csv')

    new_discretized_data_train = build_discretized_attributes(discretized_data.loc[train_index], ptype)
    new_discretized_data_test = build_discretized_attributes(discretized_data.loc[test_index], ptype)
    directory='outputs/'+ db_name +'/data/discretized'
    new_discretized_data_train.to_csv(directory+'/new_discretized_train_'+ptype.lower()+ '.csv')
    new_discretized_data_test.to_csv(directory+'/new_discretized_test_'+ptype.lower()+ '.csv')
       

if __name__ == "__main__":
    args = sys.argv[1:]
    target = args[0]
    db_name = args[1]
    ptype = args[2]

    partial_preprocessed_data  = pd.read_csv("outputs/"+db_name+"/preprocessed_sets/partial_preprocessed_data.csv", keep_default_na=False)
    partial_preprocessed_data.set_index('Unnamed: 0', inplace=True)
    partial_preprocessed_data.index.name = None


    trainset = pd.read_csv('outputs/' + db_name + '/data/classic/trainset.csv')
    trainset.set_index('Unnamed: 0', inplace=True)
    trainset.index.name = None
    testset = pd.read_csv('outputs/' + db_name + '/data/classic/testset.csv')
    testset.set_index('Unnamed: 0', inplace=True)
    testset.index.name = None
    train_index = trainset.index
    test_index = testset.index

    partial_preprocessed_data[target] = partial_preprocessed_data[target].astype("object")
    main(partial_preprocessed_data, target, train_index, test_index, ptype, db_name)
    












