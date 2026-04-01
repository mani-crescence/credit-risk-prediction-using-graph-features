from ...tools.graph import *
import sys, os, pickle, ast
import pandas as pd

def main(data, graph_type, db_name, target, discretization_type=None, testset = None):
    directory='graph/'+ db_name + '/'  
    os.makedirs(directory, exist_ok=True)
    
    if graph_type == 'BIP':
        original_graph = graph_bipartite_modality(None, data, None, discretization_type)
        descriptors_attributes = [node for node, data_ in original_graph.nodes(data=True) if data_['type'] == 'attribute']
        graph_data = {"graph": original_graph, "descriptors": descriptors_attributes}  
        
        with open(directory + 'graph_' + graph_type.lower() + '_' + discretization_type, 'wb') as file:
            pickle.dump(graph_data, file)
            
    elif graph_type == 'MOD':
        original_graph = graph_modality(None, data, None, discretization_type)
        descriptors_attributes = [node for node, data_ in original_graph.nodes(data=True) if data_['type'] == 'attribute']
        graph_data = {"graph": original_graph, "descriptors": descriptors_attributes}  
        
        with open(directory + 'graph_' + graph_type.lower() + '_' + discretization_type, 'wb') as file:
            pickle.dump(graph_data, file)
            
    elif graph_type == "COM":
         original_graph = complete_graph_parallel(trainset, testset, target)
         descriptors_attributes = ["deg0", "deg1"]     
         graph_data = {"graph": original_graph, "descriptors": descriptors_attributes}    
         
         with open(directory + 'complete_graph', 'wb') as file:
             pickle.dump(graph_data, file)
             
    elif graph_type == "GUI":
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
    
    
    
    if len(args) > 3:
        discretization_type = args[3].lower()
        
        if discretization_type is not None:
            data_file ='data/discretized/'+ db_name +'/discretized_train_data_'+ discretization_type + '.csv'
            trainset  = pd.read_csv(data_file, dtype='object', keep_default_na=False, na_values=[""])
           
            trainset.drop(columns=['Unnamed: 0'], inplace=True)
            main(trainset,  graph_type, db_name, target, discretization_type, None)
    else:
                 
        trainset = pd.read_csv("data/preprocessed/"+ db_name +"/preprocessed_data_train.csv", keep_default_na=False, na_values=[""])
        testset = pd.read_csv("data/preprocessed/"+ db_name +"/preprocessed_data_test.csv", keep_default_na=False, na_values=[""])
        
        trainset.drop(columns=['Unnamed: 0'], inplace=True)
        testset.drop(columns=['Unnamed: 0'], inplace=True)
         
        
        main(trainset,  graph_type, db_name, target, None, testset)
   
    
    
    
    
    