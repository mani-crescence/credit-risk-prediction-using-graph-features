import sys, os, pickle, ast
import pandas as pd
import networkx as nx
from ....tools.execute import pagerank_personalized


def compute_gx_class(pagerank_columns_attributes,  graph_type, discretization_type, 
                                              columns_value_proportion, target, label, df, train_data):
    couples = {}
    pagerank_attributes_copy = pagerank_columns_attributes.copy()
    columns_value_proportion_copy = columns_value_proportion.copy()
    
    
    overall_mean = df["pagerank"].mean()
    mean_class0 = df.loc[df[target] == False, "pagerank"].mean()
    mean_class1 = df.loc[df[target] == True, "pagerank"].mean()
    
    score0 = mean_class0 / overall_mean
    score1 = mean_class1 / overall_mean
    score = score0 + score1
    gx_paid_loan = score0 / score
    gx_unpaid_loan = score1 / score

    if label == 'test':
        del pagerank_attributes_copy['st_0_' + discretization_type + '_' + graph_type]
        del pagerank_attributes_copy['st_1_' + discretization_type + '_' + graph_type]

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
            
    return (gx_paid_loan + gx_paid)/2, (gx_unpaid_loan + gx_unpaid)/2      
             

def main(train_data, test_data, graph, descriptors, target, bd_name, alpha,  graph_type, discretization_type, 
        columns_value_proportion, _dir, sub):
    
    print(f'############## processing {discretization_type} with  alpha ==>{alpha} ######################')
    
    graph_descriptors = pd.DataFrame()
    graph_descriptors_for_gy = pd.DataFrame()
    
     
    for row in train_data.itertuples():
        
        graph_copy = graph.copy()     
            
        pagerank_attributes = pagerank_personalized(graph_copy, alpha, [str(row.Index)], None, graph_copy.nodes)
        
        
        pagerank_columns_attributes = {key : value for key, value in pagerank_attributes.items()  if key in descriptors}
        
        pagerank_indices_attributes = {int(key) : value for key, value in pagerank_attributes.items()  if key not in descriptors}
        
        df = pd.DataFrame.from_dict(pagerank_indices_attributes, orient="index", columns=["pagerank"])
        df = df.loc[train_data.index]
        df = pd.concat([df, train_data[target]], axis=1)
    
        
        gx_paid, gx_unpaid = compute_gx_class(pagerank_columns_attributes, graph_type, discretization_type, 
                                              columns_value_proportion, target, 'train', df, train_data)
        
        graph_descriptors_for_gy.loc[row.Index, list(pagerank_attributes.keys())] = list(pagerank_attributes.values())
        graph_descriptors.loc[row.Index, ['gx_paid', 'gx_unpaid']] = [gx_paid, gx_unpaid ]
        
        
    graph_descriptors_for_gy = graph_descriptors_for_gy.astype(float)
    graph_descriptors["gy"] = (graph_descriptors_for_gy[target + '_1'+ '_' + discretization_type + '_' +graph_type] / df.loc[df[target] == True].shape[0] > 
                               graph_descriptors_for_gy[target + '_0'+ '_' + discretization_type + '_' + graph_type] / df.loc[df[target] == False].shape[0]).astype("int8")    

    
    
    directory = _dir + bd_name + '/sub' + sub + '/' + graph_type + '/' + discretization_type +'/train'
    os.makedirs(directory, exist_ok=True)
    graph_descriptors.to_feather(directory + '/new_features_' +  str(alpha)+'.feather')
        
    graph_descriptors = pd.DataFrame()
    graph_descriptors_for_gy = pd.DataFrame()
    
    for row in test_data.itertuples():
        
        graph_copy = graph.copy()
        
        pagerank_attributes  = pagerank_personalized(graph_copy, alpha, [str(row.Index)], None, graph_copy.nodes)
        graph_descriptors_for_gy.loc[row.Index, list(pagerank_attributes.keys())] = list(pagerank_attributes.values())
        
        pagerank_columns_attributes = {key : value for key, value in pagerank_attributes.items()  if key in descriptors}
        pagerank_indices_attributes = {int(key) : value for key, value in pagerank_attributes.items()  if key not in descriptors}
        
        df = pd.DataFrame.from_dict(pagerank_indices_attributes, orient="index", columns=["pagerank"])
        df = df.loc[train_data.index]
        df = pd.concat([df, train_data[target]], axis=1)
        
        gx_paid, gx_unpaid = compute_gx_class(pagerank_columns_attributes, graph_type, discretization_type,
                                              columns_value_proportion, target, 'test', df, train_data)
        
        graph_descriptors.loc[row.Index, ['gx_paid', 'gx_unpaid']] = [gx_paid, gx_unpaid]
        graph_descriptors_for_gy.loc[row.Index, list(pagerank_attributes.keys())] = list(pagerank_attributes.values())
        
   
    # COMPUTATION OF GY
    graph_descriptors_for_gy = graph_descriptors_for_gy[descriptors]
    graph_descriptors_for_gy = graph_descriptors_for_gy.astype(float)
    graph_descriptors["gy"] = (graph_descriptors_for_gy[target + '_1'+ '_' + discretization_type + '_' +graph_type] / df.loc[df[target] == True].shape[0] > 
                               graph_descriptors_for_gy[target + '_0'+ '_' + discretization_type + '_' + graph_type] / df.loc[df[target] == False].shape[0]).astype("int8")    

    
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
    
    train_paid = train_discretized_data.loc[train_discretized_data[target] == False]
    train_unpaid = train_discretized_data.loc[train_discretized_data[target] == True]
    
    prob_train_paid = train_paid.shape[0] / train_discretized_data.shape[0]
    prob_train_unpaid = train_unpaid.shape[0] / train_discretized_data.shape[0] 
    
    proportions = {}
    
    for col in train_discretized_data.columns.drop(target): 
        probs = ( train_discretized_data.groupby(col)[target] 
                 .value_counts(normalize=True) 
                 .unstack(fill_value=0) )
        
        probs[0] = probs[0]/prob_train_paid
        probs[1] = probs[1]/prob_train_unpaid
        s = probs[0] + probs[1]
        probs[0] = probs[0]/ s
        probs[1] = probs[1]/ s
        
        proportions[col] = {}
        
        for value, row in probs.iterrows():
            proportions[col][value] = {
                cls : prob
                for cls, prob in row.items()
            }
    
   
    with open( _graph_dir + db_name + '/sub' + sub  + "/graph_"+ graph_type.lower() + '_' + discretization_type,"rb" ) as f:
        graph_data = pickle.load(f)
        
   
    main(train_discretized_data, test_discretized_data, graph_data["graph"], graph_data["descriptors"], target, db_name, alpha,  graph_type, 
          discretization_type, proportions,  _dir, sub)
    
 