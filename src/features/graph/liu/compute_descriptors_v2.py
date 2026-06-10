import sys, os, pickle, ast
import pandas as pd
import networkx as nx
from xgboost import data
from ....tools.execute import pagerank_personalized, compute_gx_class
          

def main(G, train_index, test_index, db_name):
   
    # Compute centrality measures
    degree_centrality = nx.degree_centrality(G)
    closeness_centrality = nx.closeness_centrality(G, distance='weight')
    betweenness_centrality = nx.betweenness_centrality(G, weight='weight')
    eigenvector_centrality = nx.eigenvector_centrality(G, weight='weight')

    # Compute more centrality measures
    pagerank = nx.pagerank(G, weight='weight')
    # HITS algorithm returns two dictionaries keyed by node containing hub scores and authority scores
    hubs, authorities = nx.hits(G, max_iter=1000)


    # Add these as node attributes
    for node_id in G.nodes():
        G.nodes[node_id]['degree_centrality'] = degree_centrality[node_id]
        G.nodes[node_id]['closeness_centrality'] = closeness_centrality[node_id]
        G.nodes[node_id]['betweenness_centrality'] = betweenness_centrality[node_id]
        G.nodes[node_id]['eigenvector_centrality'] = eigenvector_centrality[node_id]

        G.nodes[node_id]['pagerank'] = pagerank[node_id]
        G.nodes[node_id]['hub_score'] = hubs[node_id]
        G.nodes[node_id]['authority_score'] = authorities[node_id]
   
    
    df_node_attributes = pd.json_normalize(list(dict(G.nodes(data=True)).values()))
    
    df_node_attributes = df_node_attributes[df_node_attributes.columns[df_node_attributes.isnull().sum() == 0]]
    
    train = df_node_attributes.loc[train_index]
    test = df_node_attributes.loc[test_index]
    
    directory = _dir + db_name + '/' + graph_type
    os.makedirs(directory, exist_ok=True)
    train.to_feather(directory + '/new_features_train.feather')

    directory = _dir + db_name + '/' + graph_type
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
    
    trainset = pd.read_feather('data/preprocessed/bondora/preprocessed_data_train.feather')
    
    testset  = pd.read_feather('data/preprocessed/bondora/preprocessed_data_test.feather')
    testset.drop(columns=target, inplace=True)
    
    with open(_graph_dir + db_name + "/graph_liu","rb" ) as f:
        graph = pickle.load(f)
    

    main(graph, trainset.index, testset.index, db_name)
    
    
    