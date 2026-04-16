import sys, os, pickle, ast
import pandas as pd
import networkx as nx
from itertools import combinations

def build_graph(graph, data, new_loan = None, discretization_type = None):
    if graph is None:
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
    
    else:
        items = list(new_loan.items())
        
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
    discretization_type = args[3].lower()
    
         
    data_file ='data/discretized/'+ db_name +'/discretized_train_data_'+ discretization_type + '.csv'
    trainset  = pd.read_csv(data_file, dtype='object', keep_default_na=False, na_values=[""])
    
    trainset.drop(columns=['Unnamed: 0'], inplace=True)
    
    build_graph(None, trainset, None, discretization_type)