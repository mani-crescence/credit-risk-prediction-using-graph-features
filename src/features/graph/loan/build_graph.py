import sys, os, pickle
import networkx as nx
import pandas as pd, numpy as np
import gower
from itertools import combinations

pd.set_option('display.max_columns', None)
  

def main(train_data, test_data, target, _dir, sub):
    
    graph = nx.Graph()
    
    ### TRAINSET PROCESSING
    for i, row in train_data.iterrows():
        w = int(row[target])
        graph.add_edge(str(i), target + '_loan_' + str(w), weight=1)
   
    train_gower = gower.gower_matrix(train_data)
    test_gower  = gower.gower_matrix(test_data)

    train_indices = list(train_data.index)
    test_indices  = list(test_data.index)
    
        ### TRAINSET — pairwise edges
    for idx_a, idx_b in combinations(range(len(train_indices)), 2):
        i, j = train_indices[idx_a], train_indices[idx_b]
        graph.add_edge(str(i),  str(j),
                    weight=train_gower[idx_a, idx_b])

    ### TESTSET — pairwise edges
    for idx_a, idx_b in combinations(range(len(test_indices)), 2):
        i, j = test_indices[idx_a], test_indices[idx_b]
        graph.add_edge( str(i), str(j),
                   weight=test_gower[idx_a, idx_b])
        
        
    ### CROSS TRAIN/TEST edges
    combined     = pd.concat([train_data, test_data], axis=0)
    combined.drop(columns=[target], inplace=True) 
    cross_gower  = gower.gower_matrix(combined)
    
    n_train = len(train_indices)
    for idx_a, i in enumerate(train_indices):
        for idx_b, j in enumerate(test_indices):
            weight = cross_gower[idx_a, n_train + idx_b]
            graph.add_edge( str(i), str(j), weight=weight)    
        
    mst = nx.minimum_spanning_tree(graph, algorithm='prim') 
    graph_data = {"graph": mst, "descriptors" : [target + '_loan_0', target + '_loan_1']}     
       
    directory = _dir + db_name + '/sub' + sub + '/' 
    os.makedirs(directory, exist_ok=True)
    with open(directory + 'graph_' + graph_type.lower(), 'wb') as file:
        pickle.dump(graph_data, file)    



if __name__ == "__main__":
    args =  sys.argv[1:]   
    db_name = args[0]
    target = args[1]
    graph_type = args[2] 
    train_path = args[3]
    test_path = args[4]
    _dir = args[5]
    sub = args[6]
    
    trainset  = pd.read_feather(train_path)
    testset  = pd.read_feather(test_path)
    testset.drop(columns=[target], inplace=True) 
    
  
    main(trainset, testset, target, _dir, sub)
    