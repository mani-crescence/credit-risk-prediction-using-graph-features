import sys, os, pickle, ast
import pandas as pd
import networkx as nx
from xgboost import data
from ....tools.execute import pagerank_personalized
from .build_graph import main as build_graph


def main(train_data, test_data, graph, descriptors, db_name, alpha,  graph_type, _dir):
    
    print(f'############## processing  with  alpha ==>{alpha} ######################')
    
    graph_descriptors = pd.DataFrame()
    
    for i , _ in train_data.iterrows():
        
        graph_copy = graph.copy() 
        
        pagerank_attributes = pagerank_personalized(graph_copy, alpha, ['tr_u'+ str(i)], 'weight', descriptors)
        
        graph_descriptors.loc[i, list(pagerank_attributes.keys())] = list(pagerank_attributes.values())
        
        print('tr_u'+ str(i), ' done!')
        
    graph_descriptors = graph_descriptors.astype(float)
    
    directory = _dir + db_name + '/' + graph_type +'/train'
    os.makedirs(directory, exist_ok=True)
    graph_descriptors.to_feather(directory + '/new_features_' +  str(alpha)+'.feather')
    
    graph_descriptors = pd.DataFrame()   
    for i , _ in test_data.iterrows():
        graph_copy = graph.copy()
        
        pagerank_attributes = pagerank_personalized(graph_copy, alpha, ['ts_u'+ str(i)], "weight", descriptors)
        
        print('ts_u'+ str(i), ' done!')
                   
        graph_descriptors.loc[i, list(pagerank_attributes.keys())] = list(pagerank_attributes.values())
        
    graph_descriptors = graph_descriptors[descriptors]
    graph_descriptors = graph_descriptors.astype(float)
    
    directory = _dir + db_name + '/' + graph_type +'/test'
    os.makedirs(directory, exist_ok=True)
    graph_descriptors.to_feather(directory + '/new_features_' +  str(alpha)+'.feather')
    
    print(f"finish processed ===>  with alpha {alpha} ")

if __name__ == "__main__":
    args = sys.argv[1:]
    target = args[0]
    db_name = args[1]
    graph_type = args[2].lower()
    alpha = args[3]
    alpha = float(alpha) 
    train_path = args[4]
    test_path = args[5]
    _graph_dir = args[6]
    _dir = args[7]
    
    trainset  = pd.read_feather(train_path)
    testset  = pd.read_feather(test_path)
    testset.drop(columns=[target], inplace=True)
    
    with open(_graph_dir + db_name + "/graph_" + graph_type.lower(),"rb" ) as f:
        graph_data = pickle.load(f)
    
    
    main(trainset, testset, graph_data["graph"], graph_data["descriptors"],  db_name, alpha, graph_type, _dir)
    
    
    