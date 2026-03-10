from ...tools.graph import *
import sys, os, pickle, ast
import pandas as pd

def main(data, graph_type, db_name, target, discretization_type):
    directory='graph/'+ db_name + '/'  
    os.makedirs(directory, exist_ok=True)
    # if discretization_type is not None:
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
    # else:
    #   descriptors_attributes = []
    #   original_graph = graph_loans(None, data, target, None)
    #   target_values = data[target].unique()
    #   for i in target_values:
    #     descriptors_attributes.append(target+"_loan_"+str(i))
        
    #   graph_data = {"graph": original_graph, "descriptors": descriptors_attributes}  
    #   directory='outputs/'+db_name+'/graph/'  
    #   os.makedirs(directory, exist_ok=True)
    #   with open(directory + 'graph_' + graph_type.lower() , 'wb') as file:
    #      pickle.dump(graph_data, file)
     
    
if __name__ == "__main__":
    args =  sys.argv[1:]   
    db_name = args[0]
    target = args[1]
    discretization_type = args[2].lower()
    graph_type = args[3] 
    
    print(f"Discretization type: {discretization_type}")
   
    data_file ='data/discretized/'+ db_name +'/discretized_train_data_'+ discretization_type + '.csv'
    discretized_train_data  = pd.read_csv(data_file, dtype='object')
    discretized_train_data.drop(columns=['Unnamed: 0'], inplace=True)
    
    main(discretized_train_data,  graph_type, db_name, target, discretization_type)
   
    
    
    
    
    