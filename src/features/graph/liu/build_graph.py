import sys, os, pickle
import pandas as pd
import networkx as nx
from itertools import combinations
from sklearn.metrics.pairwise import euclidean_distances
import gower

def main(df):
    G = nx.Graph()

    for index, _ in df.iterrows():
        G.add_node(index)

    distances = gower.gower_matrix(df)

    threshold = distances.mean()

    for i in range(distances.shape[0]):
        for j in range(i+1, distances.shape[0]): 
            if distances[i, j] < threshold:
                weight = 1.0 / distances[i, j] 
                G.add_edge(i, j, weight=weight)
                
    mst = nx.minimum_spanning_tree(G, algorithm='prim')  
    return mst          
    

if __name__ == "__main__":
    args =  sys.argv[1:]   
    db_name = args[0]
    target = args[1]
    graph_type = args[2] 
    discretization_type = args[3]
    train_path = args[4]
    test_path = args[5]
    _dir = args[6]
     
    trainset  = pd.read_feather('data/preprocessed/' + db_name + '/preprocessed_data_train.feather')
    
    testset  = pd.read_feather('data/preprocessed/' + db_name + '/preprocessed_data_test.feather')
    testset.drop(columns=[target], inplace=True)
    
    df = pd.concat([trainset, testset], axis=0)
    graph  = main(df)
    
    directory = _dir  + db_name + '/'  
    os.makedirs(directory, exist_ok=True) 
    with open(directory + 'graph_' + graph_type.lower() , 'wb') as file:
        pickle.dump(graph, file)
  