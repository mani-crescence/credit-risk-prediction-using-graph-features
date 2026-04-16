import sys, os, pickle, ast
import pandas as pd
from utils import math
import networkx as nx


def euclidian_distance(u, v, attributes):
    sum = 0
    for i in attributes:
        sum += (u[i] - v[i])**2

    return  math.sqrt(sum)

def complete_graph_gui(trainset, testset, target): 
    graph = nx.Graph()
    
    cols = trainset.columns
    
    train_dicts = {i: row.to_dict() for i, row in trainset.iterrows()}
    test_dicts  = {j: row.to_dict() for j, row in testset.iterrows()}
    
    for i in  trainset.index:
        graph.add_node('tr_u' + str(i), type='train')
        
    for i, loan1 in train_dicts.items():
        for j, loan2 in train_dicts.items():
            graph.add_edge(
                    'tr_u' + str(i), 
                    'tr_u' + str(j), 
                    weight=euclidian_distance(loan1, loan2, cols)
                )
                    
    
    cols = cols.drop(target)
                
    for i in  testset.index:
        graph.add_node('ts_u' + str(i), type='test')
        
    for i, loan1 in test_dicts.items():
        for j, loan2 in test_dicts.items():
            graph.add_edge(
                'ts_u' + str(i), 
                'ts_u' + str(j), 
                weight=euclidian_distance(loan1, loan2, cols))    
        
    
    for i, loan1 in train_dicts.items():
        for j, loan2 in test_dicts.items():
            graph.add_edge(
                'tr_u' + str(i), 
                'ts_u' + str(j), 
                weight = euclidian_distance(loan1, loan2, cols))
                
    mst = nx.minimum_spanning_tree(graph, algorithm="prim")            
                            
    return mst           
      
def main(data, graph_type, db_name, target, discretization_type=None, testset = None):
    directory='graph/'+ db_name + '/'  
    os.makedirs(directory, exist_ok=True)
    
    original_graph = complete_graph_gui(trainset, testset, target)
    descriptors_attributes = ["pagerank"]     
    graph_data = {"graph": original_graph, "descriptors": descriptors_attributes}    
    
    with open(directory + 'complete_gui_graph', 'wb') as file:
        pickle.dump(graph_data, file)   
        
if __name__ == "__main__":
    args =  sys.argv[1:]   
    db_name = args[0]
    target = args[1]
    graph_type = args[2] 
    
    trainset = pd.read_csv("data/preprocessed/"+ db_name + "/preprocessed_data_train.csv", keep_default_na=False, na_values=[""])
    testset = pd.read_csv("data/preprocessed/"+ db_name + "/preprocessed_data_test.csv", keep_default_na=False, na_values=[""])
        
    trainset.drop(columns=['Unnamed: 0'], inplace=True)
    testset.drop(columns=['Unnamed: 0'], inplace=True)
        
    
    main(trainset,  graph_type, db_name, target, None, testset)

    
            