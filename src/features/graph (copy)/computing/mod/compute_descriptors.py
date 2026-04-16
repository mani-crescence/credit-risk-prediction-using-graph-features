import sys, os, pickle, ast
import pandas as pd
import networkx as nx
from tools.execute import pagerank_personalized
from build_graph import main as build_graph


def main(data, graph, descriptors, target, bd_name, alpha,  graph_type, label, discretization_type ):
    
    print(f'############## processing {discretization_type} with  alpha ==>{alpha} ######################')
    
    graph_descriptors = pd.DataFrame()
    
    if label == "train":
        for row in data.itertuples():
            
            graph_copy = graph.copy()
            
            nodes_for_personalization = []
            dict_row = row._asdict()
            del dict_row['Index']
            
            if graph_type == 'mod':
                
                for k, w in dict_row.items():
                    nodes_for_personalization.append(str(k) + '_' + str(w) + '_' + discretization_type + '_' + graph_type)
                
                pagerank_attributes = pagerank_personalized(graph_copy, alpha, nodes_for_personalization, 'weight', descriptors)
                
            
            graph_descriptors.loc[row.Index, list(pagerank_attributes.keys())] = list(pagerank_attributes.values())
            
        
        graph_descriptors = graph_descriptors.astype(float)
        
        directory='data/graph_features/'+bd_name+'/'+ discretization_type + '/' + graph_type +'/'+ label
        os.makedirs(directory, exist_ok=True)
        graph_descriptors.to_csv(directory + '/new_features_' +  str(alpha)+'.csv')
        
    elif label == "test":
        
        for row in data.itertuples():
            
            nodes_for_personalization = []
            dict_row = row._asdict()
            
            graph_copy = graph.copy()
    
            if graph_type == "mod":
                
                del dict_row[target]
                del dict_row['Index']
                
                for k, w in dict_row.items():
                   nodes_for_personalization.append(str(k) + '_' + str(w) + '_' + discretization_type + '_' + graph_type.lower())
                   
                   
                augmented_graph = build_graph(graph_copy, None, dict_row, discretization_type)
                pagerank_attributes = pagerank_personalized(augmented_graph, alpha, nodes_for_personalization, "weight", descriptors)
                
                      
            graph_descriptors.loc[row.Index, list(pagerank_attributes.keys())] = list(pagerank_attributes.values())
           
       

        graph_descriptors = graph_descriptors[descriptors]
        graph_descriptors = graph_descriptors.astype(float)
        
        directory='data/graph_features/'+bd_name+'/'+ discretization_type + '/' + graph_type +'/'+ label
        os.makedirs(directory, exist_ok=True)
        graph_descriptors.to_csv(directory + '/new_features_' +  str(alpha)+'.csv')
      
    print(f"finish processed ===> {discretization_type} with alpha {alpha} ")

if __name__ == "__main__":
    args = sys.argv[1:]
    target = args[0]
    db_name = args[1]
    graph_type = args[2].lower()
    alpha = args[3]
    alpha = float(alpha) 
    discretization_type = args[4].lower()
    label = args[5]
    
    discretized_data  = pd.read_csv("data/discretized/"+ db_name +"/discretized_" + label + "_data_"+ discretization_type +".csv", 
                                        dtype='object', keep_default_na=False, na_values=[""])
    discretized_data.drop(columns='Unnamed: 0', inplace=True)
    
    with open("graph/"+db_name+"/graph_"+ graph_type.lower() + '_' + discretization_type,"rb" ) as f:
        graph_data = pickle.load(f)

    main(discretized_data, graph_data["graph"], graph_data["descriptors"], target, db_name, alpha,  graph_type, label, discretization_type)
    
    
    