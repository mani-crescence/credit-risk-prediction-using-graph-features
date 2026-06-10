import sys, os, pickle
import pandas as pd
import networkx as nx
from itertools import combinations

def main(train_data, test_data, discretization_type):
    
    graph = nx.Graph()
    
    for row in train_data.itertuples(index=False):
        nodes = [f"{col}_{val}_{discretization_type}_mod"
                for col, val in zip(train_data.columns, row)]

        for node in nodes:
            graph.add_node(node, type="attribute")

        for i, n1 in enumerate(nodes[:-1]):
            for n2 in nodes[i+1:]:
                if graph.has_edge(n1, n2):
                    graph[n1][n2]["weight"] += 1
                else:
                    graph.add_edge(n1, n2, weight=1)
    
    
    for row in test_data.itertuples(index=False):
        dict_row = row._asdict()
        items = list(dict_row.items())
        
        nodes = [f"{k}_{w}_{discretization_type}_mod" for k, w in items]
            
        for node in nodes:
            if not graph.has_node(node):
                graph.add_node(node, type="attribute")
        
        for n1, n2 in combinations(nodes, 2):
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
    discretization_type = args[3]
    train_path = args[4]
    test_path = args[5]
    _dir = args[6]
     
    trainset  = pd.read_feather(train_path)
    
    testset  = pd.read_feather(test_path)
    testset.drop(columns=[target], inplace=True)
    
    graph  = main(trainset, testset, discretization_type)
    
    descriptors_attributes = [node for node, data_ in graph.nodes(data=True) if data_['type'] == 'attribute']
    graph_data = {"graph": graph, "descriptors": descriptors_attributes}  
    
    directory = _dir  + db_name + '/'  
    os.makedirs(directory, exist_ok=True)
    with open(directory + 'graph_' + graph_type.lower() + '_' + discretization_type, 'wb') as file:
        pickle.dump(graph_data, file)
  