import sys, os, pickle, ast
import pandas as pd
import networkx as nx
from xgboost import data
from ....tools.execute import pagerank_personalized
from .build_graph import main as build_graph


def main(train_data, test_data, graph, descriptors, db_name, alpha,  graph_type):
    
    print(f'############## processing  with  alpha ==>{alpha} ######################')
    
    graph_descriptors = pd.DataFrame()
    
    for row in train_data.itertuples():
        
        graph_copy = graph.copy() 
        
        dict_row = row._asdict()
        pagerank_attributes = pagerank_personalized(graph_copy, alpha, ['tr_u'+ str(dict_row['Index'])], 'weight', descriptors)
        
        graph_descriptors.loc[row.Index, list(pagerank_attributes.keys())] = list(pagerank_attributes.values())
        
        print('tr_u'+ str(dict_row['Index']), ' done!')
        
    graph_descriptors = graph_descriptors.astype(float)
    
    directory='data/graph_features/'+ db_name + '/' + graph_type +'/train'
    os.makedirs(directory, exist_ok=True)
    graph_descriptors.to_csv(directory + '/new_features_' +  str(alpha)+'.csv')
    
    graph_descriptors = pd.DataFrame()   
    for row in test_data.itertuples():
        dict_row = row._asdict()
        graph_copy = graph.copy()
        
        pagerank_attributes = pagerank_personalized(graph_copy, alpha, ['ts_u'+ str(dict_row['Index'])], "weight", descriptors)
        
        
        print('ts_u'+ str(dict_row['Index']), ' done!')
                   
        graph_descriptors.loc[row.Index, list(pagerank_attributes.keys())] = list(pagerank_attributes.values())
        
    graph_descriptors = graph_descriptors[descriptors]
    graph_descriptors = graph_descriptors.astype(float)
    
    directory='data/graph_features/' + db_name+'/' + graph_type +'/test'
    os.makedirs(directory, exist_ok=True)
    graph_descriptors.to_csv(directory + '/new_features_' +  str(alpha)+'.csv')
    
    print(f"finish processed ===>  with alpha {alpha} ")

if __name__ == "__main__":
    args = sys.argv[1:]
    target = args[0]
    db_name = args[1]
    graph_type = args[2].lower()
    alpha = args[3]
    alpha = float(alpha) 
    
    trainset  = pd.read_csv("data/preprocessed/"+ db_name +"/preprocessed_data_train.csv", keep_default_na=False, na_values=[""])
    trainset.drop(columns=['Unnamed: 0'], inplace=True)
    
    testset  = pd.read_csv("data/preprocessed/"+ db_name +"/preprocessed_data_test.csv", keep_default_na=False, na_values=[""])
    testset.drop(columns=['Unnamed: 0', target], inplace=True)
    
    with open("graph/" + db_name + "/graph_" + graph_type.lower(),"rb" ) as f:
        graph_data = pickle.load(f)
    
    
    main(trainset, testset, graph_data["graph"], graph_data["descriptors"],  db_name, alpha,  graph_type)
    
    
    