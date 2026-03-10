from ...tools.execute import *
import sys
import pickle

def main(target, db_name, alpha, graph_type, label, discretization_type = None):
    # partial_preprocessed_data  = pd.read_csv("outputs/"+db_name+"/preprocessed_sets/partial_preprocessed_data.csv", keep_default_na=False)
    # partial_preprocessed_data.set_index('Unnamed: 0', inplace=True) 
    # partial_preprocessed_data.index.name = None
    
    
    discretized_data  = pd.read_csv("data/discretized/"+ db_name +"/discretized_" + label + "_data_"+ discretization_type +".csv", dtype='object')
    discretized_data.drop(columns='Unnamed: 0', inplace=True)
    
    # exit(discretized_data.head())
    
    # discretized_data_test = pd.read_csv("outputs/" + db_name + "/data/discretized/graph/discretized_test_" + discretization_type + ".csv", index_col=0, dtype='int')
    # train_index = discretized_data_train.index
    # partial_preprocessed_data_train = partial_preprocessed_data.loc[train_index]
    # test_index = discretized_data_test.index
    # partial_preprocessed_data_test = partial_preprocessed_data.loc[test_index]
    # target_data_train = partial_preprocessed_data_train[target]
    # categorical_data_train= partial_preprocessed_data_train.select_dtypes("object")
    # train_data = pd.concat([discretized_data_train, categorical_data_train, target_data_train], axis=1)
    
    # target_data_test = partial_preprocessed_data_test[target]
    # categorical_data_test= partial_preprocessed_data_test.select_dtypes("object")
    # test_data = pd.concat([discretized_data_test, categorical_data_test, target_data_test], axis=1)
    
    with open("graph/"+db_name+"/graph_"+ graph_type.lower() + '_' + discretization_type,"rb" ) as f:
        graph_data = pickle.load(f)

    build_graph_attributes(discretized_data, graph_data["graph"], graph_data["descriptors"], target, db_name, alpha,  graph_type, label, discretization_type)
    # build_graph_attributes(graph_data["graph"], graph_data["descriptors"], target, db_name, alpha, "test", graph_type, None, test_data, discretization_type)
    # else:
    #     trainset  = pd.read_csv("outputs/"+db_name+"/data/classic/trainset.csv")
    #     testset = pd.read_csv("outputs/" + db_name + "/data/classic/testset.csv")
    #     trainset.set_index('Unnamed: 0', inplace=True)
    #     testset.set_index('Unnamed: 0', inplace=True)
    #     trainset.index.name = None
    #     testset.index.name = None
    #     train_index = trainset.index
    #     test_index = testset.index
        
    #     partial_preprocessed_data_train = partial_preprocessed_data.loc[train_index]
    #     partial_preprocessed_data_test = partial_preprocessed_data.loc[test_index]
    #     with open("outputs/"+db_name+"/graph/graph_"+ graph_type.lower() , "rb" ) as f:
    #         graph_data = pickle.load(f)

    #     build_graph_attributes(graph_data["graph"], graph_data["descriptors"], target, db_name, alpha, "train", graph_type, partial_preprocessed_data_train, None, discretization_type)
    #     build_graph_attributes(graph_data["graph"], graph_data["descriptors"], target, db_name, alpha, "test", graph_type, partial_preprocessed_data_train, partial_preprocessed_data_test, discretization_type)
        
        
if __name__ == "__main__":
    args = sys.argv[1:]
    target = args[0]
    db_name = args[1]
    alpha = args[2]
    alpha = float(alpha)
    graph_type = args[3].lower()
    discretization_type = args[4].lower()
    label = args[5]
    
    main(target, db_name, alpha, graph_type, label, discretization_type)
           
      
      
      
      
      
      
      
      
      
      
      
      
      
 
 
 
 












