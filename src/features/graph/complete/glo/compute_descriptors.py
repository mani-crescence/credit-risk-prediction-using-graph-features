import sys, os, pickle, ast, re
import pandas as pd
import networkx as nx
from xgboost import data
from .....tools.execute import pagerank_personalized

def main(target, db_name, alpha,  graph_type):
    
    # print(f'############## processing {discretization_type} with  alpha ==>{alpha} ######################')
    
    graph_descriptors = pd.DataFrame()
    
    with open('graph/'+ db_name + '/complete_graph' ,"rb" ) as f:
        graph_data = pickle.load(f)
    
    # PROCESSING OF TRAINING DATA             
    trainset = pd.read_csv("data/preprocessed/"+ db_name +"/preprocessed_data_train.csv", keep_default_na=False, na_values=[""])
    trainset.drop(columns=['Unnamed: 0'], inplace=True)
    _compute(trainset, graph_data, target, "tr_u", "train", alpha, graph_type, db_name, graph_descriptors)
    
    
    # PROCESSING OF TEST DATA  
    testset = pd.read_csv("data/preprocessed/"+ db_name +"/preprocessed_data_test.csv", keep_default_na=False, na_values=[""])
    testset.drop(columns=['Unnamed: 0'], inplace=True)
    _compute(testset, graph_data, target, "ts_u", "test", alpha, graph_type, db_name, graph_descriptors)
    
    
    
    
def _compute(data, graph_data, target, node_label, set_type, alpha, graph_type, db_name, graph_descriptors):
    
    graph_copy = graph_data["graph"].copy()  
    
    pagerank_df = pd.DataFrame(columns=['class0', 'class1'])
    
    for row in data.itertuples():
        
        dict_row = row._asdict()
        
        centralities_df = pd.DataFrame(columns=['pg'])
        
        pagerank_attributes = pagerank_personalized(graph_copy, alpha, [node_label + str(dict_row['Index'])], None, graph_data["graph"].nodes())
        
        for key, value in pagerank_attributes.items():
            if key.startswith('tr_u'):
                number = (re.findall(r'\d+',  key))[0]
                centralities_df.loc[int(number)] = value
                
                
        full_df =  pd.concat([centralities_df, data[target]], axis=1)    
        transformed_df = full_df.groupby(target).sum().sum(axis=1).to_frame().T 
        
        
        ratio_df = transformed_df.div(transformed_df.sum(axis=1), axis=0)
        pagerank_df.loc[dict_row['Index']] = [ratio_df.loc[0,0.0] ,ratio_df.loc[0,1.0]]
   
    pagerank_df = pagerank_df.astype(float)
    
    directory='data/graph_features/' + db_name +'/' + graph_type +'/'+ set_type
    os.makedirs(directory, exist_ok=True)
    pagerank_df.to_csv(directory + '/new_features_' +  str(alpha)+'.csv')


if __name__ == "__main__":
    args = sys.argv[1:]
    target = args[0]
    db_name = args[1]
    graph_type = args[2].lower()
    alpha = args[3]
    alpha = float(alpha)
    

    main(target, db_name, alpha,  graph_type)
    
    
    