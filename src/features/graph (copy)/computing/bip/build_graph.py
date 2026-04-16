import sys, os, pickle, ast
import pandas as pd
import networkx as nx
from tools.execute import pagerank_personalized


def main(graph, data, new_loan = None, discretization_type = None):
    if graph is None:
        graph = nx.Graph()

        for i , row in data.iterrows():
            graph.add_node('tr_u'+ str(i), type='loan', bipartite=0)
            
            for j, w in row.items():
                if not graph.has_node(str(j) + '_' + str(w) + '_' + discretization_type + '_bip'):
                    graph.add_node(str(j)+ '_'+ str(w)+'_'+discretization_type+'_bip', type='attribute', bipartite=1)
                    
                graph.add_edge('tr_u'+ str(i), str(j)+ '_'+ str(w)+'_'+discretization_type+'_bip')

        return graph
        
    else:
        edges = []
        nodes = [] 
        new_node = 'ts_'+str(new_loan['Index'])
        
        graph.add_node(new_node, type = 'loan')
        nodes.append(new_node)
        
        del new_loan['Index']
        
        for j, w in new_loan.items():
            if graph.has_node(str(j)+ '_' + str(w) + '_' + discretization_type + '_bip') is False:
                graph.add_node(str(j)+ '_'+ str(w)+'_'+discretization_type+'_bip', type = 'attribute')
            graph.add_edge(new_node, str(j)+ '_'+ str(w)+'_'+discretization_type+'_bip')
            edges.append((new_node, str(j)+ '_'+ str(w)+'_'+discretization_type+'_bip'))
        
        return graph     

if __name__ == "__main__":
    args =  sys.argv[1:]   
    db_name = args[0]
    target = args[1]
    graph_type = args[2] 
    discretization_type = args[3].lower()
    
    # STEP1: build the graph
        
    data_file ='data/discretized/'+ db_name +'/discretized_train_data_'+ discretization_type + '.csv'
    trainset  = pd.read_csv(data_file, dtype='object', keep_default_na=False, na_values=[""])
    
    trainset.drop(columns=['Unnamed: 0'], inplace=True)
    
    main(None, trainset, None, discretization_type)
    