import sys, os, pickle, ast
import pandas as pd
import networkx as nx
from ....tools.execute import pagerank_personalized, compute_gx_class


def main(train_data, test_data, graph, descriptors, target, bd_name, alpha,  graph_type, discretization_type, 
         paid_proportion_of_columns, unpaid_proportion_of_columns, number_of_paid_items, number_of_unpaid_items):
    
    print(f'############## processing {discretization_type} with  alpha ==>{alpha} ######################')
    
    graph_descriptors = pd.DataFrame()
    graph_descriptors_for_gy = pd.DataFrame()
     
    for row in train_data.itertuples():
        
        graph_copy = graph.copy()     
            
        pagerank_attributes = pagerank_personalized(graph_copy, alpha, ['tr_u' + str(row.Index)], None, descriptors)
        
        
        gx_paid, gx_unpaid = compute_gx_class(pagerank_attributes, graph_type, discretization_type, paid_proportion_of_columns,
                                              unpaid_proportion_of_columns, number_of_paid_items, number_of_unpaid_items, target)
        
        graph_descriptors_for_gy.loc[row.Index, list(pagerank_attributes.keys())] = list(pagerank_attributes.values())
        graph_descriptors.loc[row.Index, ['gx_paid', 'gx_unpaid']] = [gx_paid, gx_unpaid]
        
        
    
    graph_descriptors_for_gy = graph_descriptors_for_gy.astype(float)
    graph_descriptors["gy"] = (graph_descriptors_for_gy[target + '_1'+ '_' + discretization_type + '_' +graph_type] > graph_descriptors_for_gy[target + '_0'+ '_' + discretization_type + '_' + graph_type]).astype("int8")    

    
    directory='data/graph_features/' + bd_name+'/'+ graph_type + '/' + discretization_type +'/train'
    os.makedirs(directory, exist_ok=True)
    graph_descriptors.to_csv(directory + '/new_features_' +  str(alpha)+'.csv')
        
    graph_descriptors = pd.DataFrame()
    graph_descriptors_for_gy = pd.DataFrame()
    for row in test_data.itertuples():
        
        graph_copy = graph.copy()
        
        pagerank_attributes  = pagerank_personalized(graph_copy, alpha, ['ts_u' + str(row.Index)], None, descriptors)
        graph_descriptors_for_gy.loc[row.Index, list(pagerank_attributes.keys())] = list(pagerank_attributes.values())
        
        gx_paid, gx_unpaid = compute_gx_class(pagerank_attributes, graph_type, discretization_type, paid_proportion_of_columns, unpaid_proportion_of_columns,
                                              number_of_paid_items, number_of_unpaid_items, target)
        graph_descriptors.loc[row.Index, ['gx_paid', 'gx_unpaid']] = [gx_paid, gx_unpaid]
        
    
    # COMPUTATION OF GY
    graph_descriptors_for_gy = graph_descriptors_for_gy[descriptors]
    graph_descriptors_for_gy = graph_descriptors_for_gy.astype(float)
    graph_descriptors["gy"] = (graph_descriptors_for_gy[target + '_1'+ '_' + discretization_type + '_' + graph_type] > graph_descriptors_for_gy[target + '_0'+ '_' + discretization_type + '_' + graph_type]).astype("int8")    

    
    directory='data/graph_features/' + bd_name+'/'+ graph_type + '/' + discretization_type +'/test'
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
    
    train_discretized_data  = pd.read_csv("data/discretized/"+ db_name +"/discretized_train_data_"+ discretization_type +".csv", 
                                        dtype='object', keep_default_na=False, na_values=[""])
    train_discretized_data.drop(columns='Unnamed: 0', inplace=True)
    
    test_discretized_data  = pd.read_csv("data/discretized/"+ db_name +"/discretized_test_data_"+ discretization_type +".csv", 
                                        dtype='object', keep_default_na=False, na_values=[""])
    test_discretized_data.drop(columns='Unnamed: 0', inplace=True)
    
    paid_columns_repartition = {}
    unpaid_columns_repartition = {}
    paid_discretized_data = train_discretized_data.loc[train_discretized_data[target] == '0']
    unpaid_discretized_data = train_discretized_data.loc[train_discretized_data[target] == '1']
    
    number_of_paid_items = 0
    for col in train_discretized_data.columns.drop(target):
        paid_columns_repartition[col]= {}
        for key, value in paid_discretized_data[col].value_counts(normalize=True).items():
            paid_columns_repartition[col][key] = value
            number_of_paid_items += 1
      
    number_of_unpaid_items = 0
    for col in train_discretized_data.columns.drop(target):
        unpaid_columns_repartition[col]= {}
        for key, value in unpaid_discretized_data[col].value_counts(normalize=True).items():
            unpaid_columns_repartition[col][key] = value  
            number_of_unpaid_items += 1
            

    with open("graph/"+db_name+"/graph_"+ graph_type.lower() + '_' + discretization_type,"rb" ) as f:
        graph_data = pickle.load(f)
        
   
    main(train_discretized_data, test_discretized_data, graph_data["graph"], graph_data["descriptors"], target, db_name, alpha,  graph_type, 
          discretization_type, paid_columns_repartition, unpaid_columns_repartition, number_of_paid_items, number_of_unpaid_items)
    