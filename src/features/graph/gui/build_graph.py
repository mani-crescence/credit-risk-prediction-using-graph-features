import sys, os, pickle
import pandas as pd
import networkx as nx
from sklearn.metrics.pairwise import nan_euclidean_distances

def main(df):
    
   # Create a graph
    G = nx.Graph()

    # Add nodes to the graph
    for index, row in df.iterrows():
        G.add_node(index)

    # Compute pairwise Euclidean distances
    distances = nan_euclidean_distances(df)
    
    threshold = distances.mean()

    for i in range(distances.shape[0]):
        for j in range(i+1, distances.shape[0]): 
            
            if distances[i, j] < threshold:
                weight = 1.0 / distances[i, j] 
                G.add_edge(i, j, weight=weight)
                
    mst = nx.minimum_spanning_tree(G)            
            
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
    
     
    trainset  = pd.read_csv('data/preprocessed/' + db_name + '/preprocessed_data_train.csv', dtype='object', keep_default_na=False, na_values=[""], index_col=0)
    
    testset  = pd.read_csv('data/preprocessed/' + db_name + '/preprocessed_data_test.csv', dtype='object', keep_default_na=False, na_values=[""], index_col=0)
    testset.drop(columns=[target], inplace=True)
    
    df = pd.concat([trainset, testset], axis=0)
    
    
    graph  = main(df)
    
    directory = _dir  + db_name + '/'  
    os.makedirs(directory, exist_ok=True)
    with open(directory + 'graph_' + graph_type.lower(), 'wb') as file:
        pickle.dump(graph, file)
  