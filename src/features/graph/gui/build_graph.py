import sys, os, pickle
import pandas as pd
import networkx as nx
from sklearn.metrics.pairwise import *

def main(df):
    
   # Create a graph
    G = nx.Graph()

    # Add nodes to the graph
    for index, row in df.iterrows():
        G.add_node(index)
        

    # Compute pairwise Euclidean distances
    # distances = euclidean_distances(df.values)
    distances = nan_euclidean_distances(df.values)
    
    threshold = distances.mean()
    
    labels = df.index.tolist()

    for i in range(distances.shape[0]):
        for j in range(i+1, distances.shape[0]): 
            
            if distances[i, j] < threshold:
                weight = 1.0 / distances[i, j] 
                G.add_edge(labels[i], labels[j], weight=weight)
                
                  
    mst = nx.minimum_spanning_tree(G, algorithm='prim')  
    
    # def edge_filter(u, v):
    #     return (u, v) in mst.edges()

    # # Create a filtered subgraph based on the edge filter
    # filtered_graph = G.edge_subgraph((u, v) for u, v in G.edges() if edge_filter(u, v))   
    
              
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
    sub = args[7]
    
     
    trainset  = pd.read_feather('data/preprocessed/' + db_name + '/preprocessed_data_train_' + sub + '.feather')
    testset  = pd.read_feather('data/preprocessed/' + db_name + '/preprocessed_data_test_' + sub + '.feather')
    testset.drop(columns=[target], inplace=True)
    
    df = pd.concat([trainset, testset], axis=0)
    
    graph  = main(df)
    
    directory = _dir  + db_name + '/sub' + sub + '/'  
    os.makedirs(directory, exist_ok=True)
    with open(directory + 'graph_' + graph_type.lower(), 'wb') as file:
        pickle.dump(graph, file)
  