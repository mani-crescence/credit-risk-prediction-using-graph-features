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
    # eigenvector_centrality = nx.eigenvector_centrality(G, weight='weight')

    # Compute more centrality measures
    pagerank = nx.pagerank(G, weight='weight')
    # HITS algorithm returns two dictionaries keyed by node containing hub scores and authority scores
    hubs, authorities = nx.hits(G, max_iter=1000)


    # Add these as node attributes
    for node_id in G.nodes():
        G.nodes[node_id]['degree_centrality'] = degree_centrality[node_id]
        G.nodes[node_id]['closeness_centrality'] = closeness_centrality[node_id]
        G.nodes[node_id]['betweenness_centrality'] = betweenness_centrality[node_id]
        # G.nodes[node_id]['eigenvector_centrality'] = eigenvector_centrality[node_id]

        G.nodes[node_id]['pagerank'] = pagerank[node_id]
        G.nodes[node_id]['hub_score'] = hubs[node_id]
        G.nodes[node_id]['authority_score'] = authorities[node_id]

    df_node_attributes = pd.json_normalize(list(dict(G.nodes(data=True)).values()))
    
    train = df_node_attributes.loc[train_index]
    test = df_node_attributes.loc[test_index]
    
    directory = _dir + db_name + '/' + graph_type + '/train'
    os.makedirs(directory, exist_ok=True)
    train.to_csv(directory + '/new_features_' +  str(alpha)+'.csv')

    directory = _dir + db_name + '/' + graph_type + '/test'
    os.makedirs(directory, exist_ok=True)
    test.to_csv(directory + '/new_features_' +  str(alpha)+'.csv')


if __name__ == "__main__":
    args = sys.argv[1:]
    target = args[0]
    db_name = args[1]
    graph_type = args[2].lower()
    alpha = args[3]
    alpha = float(alpha) 
    discretization_type = args[4].lower()
    train_path = args[5]
    test_path = args[6]
    _dir = args[7]
    _graph_dir = args[8]
    
    trainset = pd.read_csv('data/preprocessed/bondora/preprocessed_data_train.csv', dtype='object', keep_default_na=False, na_values=[""])
    trainset.drop(columns='Unnamed: 0', inplace=True)
    
    testset  = pd.read_csv('data/preprocessed/bondora/preprocessed_data_test.csv', dtype='object', keep_default_na=False, na_values=[""])
    testset.drop(columns='Unnamed: 0', inplace=True)
    
    with open(_graph_dir + db_name + "/graph_"+ graph_type.lower() + '_' + discretization_type,"rb" ) as f:
        graph_data = pickle.load(f)
    

    main(graph_data['graph'], trainset.index, testset.index)
    
    
    