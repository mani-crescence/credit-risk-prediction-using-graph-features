import sys, os, pickle
import pandas as pd
import networkx as nx
from itertools import combinations

def main(graph, data, set_type, discretization_type):
    if set_type == "train":
        graph = nx.Graph()
        
        for row in data.itertuples(index=False):
            nodes = [f"{col}_{val}_{discretization_type}_mod"
                    for col, val in zip(data.columns, row)]

            for node in nodes:
                graph.add_node(node, type="attribute")

            for i, n1 in enumerate(nodes[:-1]):
                for n2 in nodes[i+1:]:
                    if graph.has_edge(n1, n2):
                        graph[n1][n2]["weight"] += 1
                    else:
                        graph.add_edge(n1, n2, weight=1)
        
      
        return graph
    
   

if __name__ == "__main__":
    args =  sys.argv[1:]   
    db_name = args[0]
    target = args[1]
    graph_type = args[2] 
    discretization_type = args[3].lower()
    
         
    data_file ='data/discretized/'+ db_name +'/discretized_train_data_'+ discretization_type + '.csv'
    trainset  = pd.read_csv(data_file, dtype='object', keep_default_na=False, na_values=[""])
    
    trainset.drop(columns=['Unnamed: 0'], inplace=True)
    
    graph  = main(None, trainset, "train", discretization_type)
    
    descriptors_attributes = [node for node, data_ in graph.nodes(data=True) if data_['type'] == 'attribute']
    graph_data = {"graph": graph, "descriptors": descriptors_attributes}  
    
    directory='graph/'+ db_name + '/'  
    os.makedirs(directory, exist_ok=True)
    with open(directory + 'graph_' + graph_type.lower() + '_' + discretization_type, 'wb') as file:
        pickle.dump(graph_data, file)
  