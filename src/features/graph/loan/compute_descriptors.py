import sys, os, pickle, ast
import pandas as pd
import networkx as nx
from xgboost import data
from ....tools.execute import pagerank_personalized
from .build_graph import main as build_graph


def compute_gx_class(pagerank_indices_attributes, indices_value_proportion):
        
    gx_ind_paid = 0     
    for k1, v1 in pagerank_indices_attributes.items(): 
        gx_ind_paid += float(v1)* float(indices_value_proportion[int(k1[4:])][0])
     
    
    gx_ind_unpaid = 0     
    for k1, v1 in pagerank_indices_attributes.items() : 
        gx_ind_paid += float(v1)* float(indices_value_proportion[int(k1[4:])][1])  
        
    return  gx_ind_paid, gx_ind_unpaid        
             
    
 

def main(train_data, test_data, graph, descriptors, db_name, alpha,  graph_type, _dir, indices_value_proportion):
    
    print(f'############## processing  with  alpha ==>{alpha} ######################')
    
    graph_descriptors = pd.DataFrame()
    graph_descriptors_for_gy = pd.DataFrame()
    
    for i , _ in train_data.iterrows():
        
        graph_copy = graph.copy() 
        
        pagerank_attributes = pagerank_personalized(graph_copy, alpha, ['tr_u'+ str(i)], 'weight', graph_copy.nodes)
        
        pagerank_indices_attributes = {key : value for key, value in pagerank_attributes.items()  if key not in descriptors}
        
        gx_paid, gx_unpaid = compute_gx_class(pagerank_indices_attributes, indices_value_proportion)
        
        graph_descriptors.loc[i, ['gx_paid', 'gx_unpaid']] = [gx_paid, gx_unpaid]
        graph_descriptors_for_gy.loc[i, list(pagerank_attributes.keys())] = list(pagerank_attributes.values())
        
        # graph_descriptors.loc[i, list(pagerank_attributes.keys())] = list(pagerank_attributes.values())
        
        # print('tr_u'+ str(i), ' done!')
     
    graph_descriptors_for_gy = graph_descriptors_for_gy[descriptors] 
    graph_descriptors_for_gy = graph_descriptors_for_gy.astype(float)
    graph_descriptors["gy"] = (graph_descriptors_for_gy[target +'_' + graph_type + '_1'] > 
                               graph_descriptors_for_gy[target +'_' + graph_type + '_0']).astype("int8")    

    directory = _dir + db_name + '/' + graph_type +'/train'
    os.makedirs(directory, exist_ok=True)
    graph_descriptors.to_feather(directory + '/new_features_' +  str(alpha)+'.feather')
    
    graph_descriptors = pd.DataFrame()   
    graph_descriptors_for_gy = pd.DataFrame()
    
    for i , _ in test_data.iterrows():
        graph_copy = graph.copy()
        
        pagerank_attributes = pagerank_personalized(graph_copy, alpha, ['ts_u'+ str(i)], "weight", graph_copy.nodes)
        
        pagerank_indices_attributes = {key : value for key, value in pagerank_attributes.items()  if key not in descriptors}
        
        gx_paid, gx_unpaid = compute_gx_class(pagerank_indices_attributes, indices_value_proportion)
        
        graph_descriptors.loc[i, ['gx_paid', 'gx_unpaid']] = [gx_paid, gx_unpaid]
        graph_descriptors_for_gy.loc[i, list(pagerank_attributes.keys())] = list(pagerank_attributes.values())

        
        
    graph_descriptors_for_gy = graph_descriptors_for_gy[descriptors]
    graph_descriptors_for_gy = graph_descriptors_for_gy.astype(float)
    graph_descriptors["gy"] = (graph_descriptors_for_gy[target +'_' + graph_type + '_1'] > 
                               graph_descriptors_for_gy[target +'_' + graph_type + '_0']).astype("int8") 
    
    directory = _dir + db_name + '/' + graph_type +'/test'
    os.makedirs(directory, exist_ok=True)
    graph_descriptors.to_feather(directory + '/new_features_' +  str(alpha)+'.feather')
    
    print(f"finish processed ===>  with alpha {alpha} ")

if __name__ == "__main__":
    args = sys.argv[1:]
    target = args[0]
    db_name = args[1]
    graph_type = args[2].lower()
    alpha = args[3]
    alpha = float(alpha) 
    train_path = args[4]
    test_path = args[5]
    _graph_dir = args[6]
    _dir = args[7]
    
    trainset  = pd.read_feather(train_path)
    testset  = pd.read_feather(test_path)
    full_df = pd.concat([trainset, testset], axis=0)
    testset.drop([target], axis=1, inplace=True)
    indices_value_proportion={}
    
    for index, row in full_df.iterrows():
        indices_value_proportion[index]={}
    
    for index, row in full_df.iterrows():
        if row[target] == 0:
            indices_value_proportion[index][0] = 1
            indices_value_proportion[index][1] = 0
            
        if row[target] == 1:
            indices_value_proportion[index][0] = 0
            indices_value_proportion[index][1] = 1  
    
    
    with open(_graph_dir + db_name + "/graph_" + graph_type.lower(),"rb" ) as f:
        graph_data = pickle.load(f)
    
    # print(graph_data["graph"].nodes)    
    # exit()
    
    main(trainset, testset, graph_data["graph"], graph_data["descriptors"],  db_name, 
         alpha, graph_type, _dir, indices_value_proportion)
    
    
    