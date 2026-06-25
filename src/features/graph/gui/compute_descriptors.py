import sys, os, pickle 
import pandas as pd
import networkx as nx
          

def main(G, train_index, test_index, db_name, sub, df_indexes):
   
    pagerank = nx.pagerank(G, weight='weight')
    
    for node_id in G.nodes():
        G.nodes[node_id]['pagerank'] = pagerank[node_id]

    df_node_attributes = pd.json_normalize(list(dict(G.nodes(data=True)).values()))
    
    df_node_attributes.index = df_indexes
    
    train = df_node_attributes.loc[train_index]
    test = df_node_attributes.loc[test_index]
    
    directory = _dir + db_name + '/sub' + sub + '/' + graph_type
    os.makedirs(directory, exist_ok=True)
    train.to_feather(directory + '/new_features_train.feather')

    directory = _dir + db_name + '/sub' + sub + '/' + graph_type
    os.makedirs(directory, exist_ok=True)
    test.to_feather(directory + '/new_features_test.feather')


if __name__ == "__main__":
    args = sys.argv[1:]
    target = args[0]
    db_name = args[1]
    graph_type = args[2].lower()
    train_path = args[3]
    test_path = args[4]
    _dir = args[5]
    _graph_dir = args[6]
    sub = args[7]
    
    trainset = pd.read_feather(train_path)
    testset  = pd.read_feather(test_path)
    
    df = pd.concat([trainset, testset], axis=0)
    
    with open(_graph_dir + db_name + '/sub' + sub  + "/graph_"+ graph_type.lower() ,"rb" ) as f:
        graph = pickle.load(f)
    

    main(graph, trainset.index, testset.index, db_name, sub, df.index)
    
    
    