import sys, os, pickle, ast
import pandas as pd
import networkx as nx
from xgboost import data
from ....tools.execute import pagerank_personalized
from .build_graph import main as build_graph


def compute_gx_class(pagerank_indices_attributes, train_data):
    df = pd.DataFrame.from_dict(pagerank_indices_attributes, orient="index", columns=["pagerank"])
    df = df.loc[train_data.index]
    df = pd.concat([df, train_data[target]], axis=1)   
     
    overall_mean = df["pagerank"].mean()
    mean_class0 = df.loc[df[target] == False, "pagerank"].mean()
    mean_class1 = df.loc[df[target] == True, "pagerank"].mean()
    
    score0 = mean_class0 / overall_mean
    score1 = mean_class1 / overall_mean
    score = score0 + score1
    gx_paid_loan = score0 / score
    gx_unpaid_loan = score1 / score
        
    return  gx_paid_loan, gx_unpaid_loan        
             
    
 

def main(train_data, test_data, graph, descriptors, db_name, alpha,  graph_type, _dir,  sub, paid_size, unpaid_size):
    
    print(f'############## processing  with  alpha ==>{alpha} ######################')
    
    graph_descriptors = pd.DataFrame()
    graph_descriptors_for_gy = pd.DataFrame()
    
    for i , _ in train_data.iterrows():
        
        graph_copy = graph.copy() 
        
        pagerank_attributes = pagerank_personalized(graph_copy, alpha, [str(i)], 'weight', graph_copy.nodes)
        
        pagerank_indices_attributes = {int(key) : value for key, value in pagerank_attributes.items()  if key not in descriptors}
        
        gx_paid, gx_unpaid = compute_gx_class(pagerank_indices_attributes, train_data)
        
        graph_descriptors.loc[i, ['gx_paid', 'gx_unpaid']] = [gx_paid, gx_unpaid]
        graph_descriptors_for_gy.loc[i, list(pagerank_attributes.keys())] = list(pagerank_attributes.values())
      
    graph_descriptors_for_gy = graph_descriptors_for_gy[descriptors] 
    graph_descriptors_for_gy = graph_descriptors_for_gy.astype(float)
    graph_descriptors["gy"] = (graph_descriptors_for_gy[target +'_' + graph_type + '_1'] / unpaid_size > 
                               graph_descriptors_for_gy[target +'_' + graph_type + '_0'] / paid_size).astype("int8")    

    directory = _dir + db_name + '/sub' + sub + '/' + graph_type +'/train'
    os.makedirs(directory, exist_ok=True)
    graph_descriptors.to_feather(directory + '/new_features_' +  str(alpha)+'.feather')
    
    graph_descriptors = pd.DataFrame()   
    graph_descriptors_for_gy = pd.DataFrame()
    
    for i , _ in test_data.iterrows():
        graph_copy = graph.copy()
        
        pagerank_attributes = pagerank_personalized(graph_copy, alpha, [str(i)], "weight", graph_copy.nodes)
        
        pagerank_indices_attributes = {int(key) : value for key, value in pagerank_attributes.items()  if key not in descriptors}
        
        gx_paid, gx_unpaid = compute_gx_class(pagerank_indices_attributes, train_data)
        
        graph_descriptors.loc[i, ['gx_paid', 'gx_unpaid']] = [gx_paid, gx_unpaid]
        graph_descriptors_for_gy.loc[i, list(pagerank_attributes.keys())] = list(pagerank_attributes.values())

        
        
    graph_descriptors_for_gy = graph_descriptors_for_gy[descriptors]
    graph_descriptors_for_gy = graph_descriptors_for_gy.astype(float)
    graph_descriptors["gy"] = (graph_descriptors_for_gy[target +'_' + graph_type + '_1'] / unpaid_size > 
                               graph_descriptors_for_gy[target +'_' + graph_type + '_0'] / paid_size).astype("int8") 
    
    directory = _dir + db_name + '/sub' + sub +'/' + graph_type +'/test'
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
    sub = args[8]
   
    
    trainset  = pd.read_feather(train_path)
    testset  = pd.read_feather(test_path)
    testset.drop([target], axis=1, inplace=True) 
    train_paid = trainset.loc[trainset[target] == False]
    train_unpaid = trainset.loc[trainset[target] == True]
    
    
    
    with open(_graph_dir + db_name + '/sub' + sub + "/graph_" + graph_type.lower(),"rb" ) as f:
        graph_data = pickle.load(f)
    
    main(trainset, testset, graph_data["graph"], graph_data["descriptors"],  db_name, 
         alpha, graph_type, _dir,  sub, train_paid.shape[0], train_unpaid.shape[0])
    
    
    