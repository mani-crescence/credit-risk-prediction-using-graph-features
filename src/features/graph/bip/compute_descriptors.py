import sys, os, pickle, ast
import pandas as pd
import networkx as nx
from ....tools.execute import pagerank_personalized


def compute_gx_class(pagerank_columns_attributes, pagerank_indices_attributes, graph_type, discretization_type, indices_value_proportion, 
                                              columns_value_proportion, target, label):
    couples = {}
    pagerank_attributes_copy = pagerank_columns_attributes.copy()
    columns_value_proportion_copy = columns_value_proportion.copy()
    
    if label == 'test':
        del pagerank_attributes_copy['st_0_' + discretization_type + '_' + graph_type]
        del pagerank_attributes_copy['st_1_' + discretization_type + '_' + graph_type]
        del columns_value_proportion_copy[target]

    for key, value in pagerank_attributes_copy.items():
        sub = "_" + discretization_type + "_" + graph_type
        couple = key.replace(sub, "")
        couple = couple.split("_", 1)
        
        if couple[0] not in couples:
            couples[couple[0]] = {} 
        couples[couple[0]][couple[1]]  = value   
        
    
    gx_paid = 0     
    for k1, v1 in columns_value_proportion_copy.items(): 
        for ka, va in v1.items():
            gx_paid += float(va[0])* float(couples[k1][str(ka)])
   
      
    gx_unpaid = 0     
    for k1, v1 in columns_value_proportion_copy.items() : 
        for ka, va in v1.items():
            gx_unpaid += float(va[1])*float(couples[k1][str(ka)])   
    
        
    gx_ind_paid = 0     
    for k1, v1 in pagerank_indices_attributes.items(): 
        try :
            gx_ind_paid += float(v1)* float(indices_value_proportion[int(k1[4:])][0])
        except:
            # print("node ", k1, "not in trainset !")    
            pass
     
    
    gx_ind_unpaid = 0     
    for k1, v1 in pagerank_indices_attributes.items() : 
        try :
            gx_ind_paid += float(v1)* float(indices_value_proportion[int(k1[4:])][1])  
        except:
            # print("node ", k1, "not in trainset !")
            pass
                
        
    return (gx_paid + gx_ind_paid), (gx_unpaid + gx_ind_unpaid )         
             
    
 


def main(train_data, test_data, graph, descriptors, target, bd_name, alpha,  graph_type, discretization_type, 
         indices_value_proportion, columns_value_proportion, _dir, sub):
    
    print(f'############## processing {discretization_type} with  alpha ==>{alpha} ######################')
    
    graph_descriptors = pd.DataFrame()
    graph_descriptors_for_gy = pd.DataFrame()
     
    for row in train_data.itertuples():
        
        graph_copy = graph.copy()     
            
        pagerank_attributes = pagerank_personalized(graph_copy, alpha, ['tr_u' + str(row.Index)], None, graph_copy.nodes)
        
        
        pagerank_columns_attributes = {key : value for key, value in pagerank_attributes.items()  if key in descriptors}
        
        pagerank_indices_attributes = {key : value for key, value in pagerank_attributes.items()  if key not in descriptors}
        
        
        gx_paid, gx_unpaid = compute_gx_class(pagerank_columns_attributes, pagerank_indices_attributes, graph_type, discretization_type, indices_value_proportion, 
                                              columns_value_proportion, target, 'train')
        
        graph_descriptors_for_gy.loc[row.Index, list(pagerank_attributes.keys())] = list(pagerank_attributes.values())
        graph_descriptors.loc[row.Index, ['gx_paid', 'gx_unpaid']] = [gx_paid, gx_unpaid]
        
    graph_descriptors_for_gy = graph_descriptors_for_gy.astype(float)
    graph_descriptors["gy"] = (graph_descriptors_for_gy[target + '_1'+ '_' + discretization_type + '_' +graph_type] > graph_descriptors_for_gy[target + '_0'+ '_' + discretization_type + '_' + graph_type]).astype("int8")    

    
    
    directory = _dir + bd_name + '/sub' + sub + '/' + graph_type + '/' + discretization_type +'/train'
    os.makedirs(directory, exist_ok=True)
    graph_descriptors.to_feather(directory + '/new_features_' +  str(alpha)+'.feather')
        
    graph_descriptors = pd.DataFrame()
    graph_descriptors_for_gy = pd.DataFrame()
    
    for row in test_data.itertuples():
        
        graph_copy = graph.copy()
        
        pagerank_attributes  = pagerank_personalized(graph_copy, alpha, ['ts_u' + str(row.Index)], None, graph_copy.nodes)
        graph_descriptors_for_gy.loc[row.Index, list(pagerank_attributes.keys())] = list(pagerank_attributes.values())
        
        pagerank_columns_attributes = {key : value for key, value in pagerank_attributes.items()  if key in descriptors}
        pagerank_indices_attributes = {key : value for key, value in pagerank_attributes.items()  if key not in descriptors}
        
        gx_paid, gx_unpaid = compute_gx_class(pagerank_columns_attributes, pagerank_indices_attributes, graph_type, discretization_type, indices_value_proportion, 
                                              columns_value_proportion, target, 'test')
        
        graph_descriptors.loc[row.Index, ['gx_paid', 'gx_unpaid']] = [gx_paid, gx_unpaid]
        graph_descriptors_for_gy.loc[row.Index, list(pagerank_attributes.keys())] = list(pagerank_attributes.values())
        
   
    # COMPUTATION OF GY
    graph_descriptors_for_gy = graph_descriptors_for_gy[descriptors]
    graph_descriptors_for_gy = graph_descriptors_for_gy.astype(float)
    graph_descriptors["gy"] = (graph_descriptors_for_gy[target + '_1'+ '_' + discretization_type + '_' + graph_type] > 
                               graph_descriptors_for_gy[target + '_0'+ '_' + discretization_type + '_' + graph_type]).astype("int8")    

    
    directory = _dir + bd_name + '/sub' + sub + '/' + graph_type + '/' + discretization_type +'/test'
    os.makedirs(directory, exist_ok=True)
    graph_descriptors.to_feather(directory + '/new_features_' +  str(alpha)+'.feather')
      
    print(f"finish processed ===> {discretization_type} with alpha {alpha} ")

if __name__ == "__main__":
    args = sys.argv[1:]
    target = args[0]
    db_name = args[1]
    graph_type = args[2].lower()
    alpha = args[3]
    alpha = float(alpha) 
    discretization_type = args[4].lower()
    train_path = args[5]
    test_path = args[6]
    _dir = args[7]
    _graph_dir = args[8]
    sub = args[9]
   
    train_discretized_data  = pd.read_feather(train_path)
    test_discretized_data  = pd.read_feather(test_path)
    test_discretized_data.drop([target], axis=1, inplace=True)
    
    columns_value_proportion = {}
    indices_value_proportion = {}
    
    for index, row in train_discretized_data.iterrows():
        indices_value_proportion[index]={}
    
    for index, row in train_discretized_data.iterrows():
        if row[target] == 0:
            indices_value_proportion[index][0] = 1
            indices_value_proportion[index][1] = 0
            
        if row[target] == 1:
            indices_value_proportion[index][0] = 0
            indices_value_proportion[index][1] = 1    
            
 
    for i in train_discretized_data.columns:
         proportion = pd.crosstab(train_discretized_data[i], train_discretized_data[target], normalize='index')
         columns_value_proportion[i] = proportion.to_dict(orient='index')
   
    with open( _graph_dir + db_name + '/sub' + sub  + "/graph_"+ graph_type.lower() + '_' + discretization_type,"rb" ) as f:
        graph_data = pickle.load(f)
        
   
    main(train_discretized_data, test_discretized_data, graph_data["graph"], graph_data["descriptors"], target, db_name, alpha,  graph_type, 
          discretization_type, indices_value_proportion, columns_value_proportion,  _dir, sub)
    
 