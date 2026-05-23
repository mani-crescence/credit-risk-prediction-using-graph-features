import sys, os, pickle, ast
import pandas as pd
import networkx as nx



def build(graph, data, set_type, discretization_type = None):
    if set_type == "train":
        graph = nx.Graph()

        for i , row in data.iterrows():
            graph.add_node('tr_u'+ str(i), type='loan', bipartite=0)
            
            for j, w in row.items():
                if not graph.has_node(str(j) + '_' + str(w) + '_' + discretization_type + '_bip'):
                    graph.add_node(str(j)+ '_'+ str(w)+'_'+discretization_type+'_bip', type='attribute', bipartite=1)
                    
                graph.add_edge('tr_u'+ str(i), str(j)+ '_'+ str(w)+'_'+discretization_type+'_bip')

        return graph
        
    else:
        for i , row in data.iterrows():
            graph.add_node('ts_u'+ str(i), type='loan', bipartite=0)
            
            for j, w in row.items():
                if not graph.has_node(str(j) + '_' + str(w) + '_' + discretization_type + '_bip'):
                    graph.add_node(str(j)+ '_'+ str(w)+'_'+discretization_type+'_bip', type='attribute', bipartite=1)
                    
                graph.add_edge('ts_u'+ str(i), str(j)+ '_'+ str(w)+'_'+discretization_type+'_bip')
        
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
    
    trainset  = pd.read_csv(train_path, dtype='object', keep_default_na=False, na_values=[""])
    testset  = pd.read_csv(test_path, dtype='object', keep_default_na=False, na_values=[""])
    
    trainset.drop(columns=['Unnamed: 0'], inplace=True)
    testset.drop(columns=['Unnamed: 0', target], inplace=True)
    
    inter_graph = build(None, trainset, "train", discretization_type)
    
    graph = build(inter_graph, testset, "test", discretization_type) 
    
    descriptors_attributes = [node for node, data_ in inter_graph.nodes(data=True) if data_['type'] == 'attribute']
    graph_data = {"graph": graph, "descriptors": descriptors_attributes}  
    
    directory = _dir + db_name + '/'  
    os.makedirs(directory, exist_ok=True)
    with open(directory + 'graph_' + graph_type.lower() + '_' + discretization_type, 'wb') as file:
        pickle.dump(graph_data, file)
    
  