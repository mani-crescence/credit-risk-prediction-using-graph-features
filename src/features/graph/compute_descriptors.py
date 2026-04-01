from ...tools.execute import *
import sys
import pickle

def main(target, db_name, graph_type, alpha, label, discretization_type = None):
    
    if discretization_type is not None:
   
        discretized_data  = pd.read_csv("data/discretized/"+ db_name +"/discretized_" + label + "_data_"+ discretization_type +".csv", 
                                        dtype='object', keep_default_na=False, na_values=[""])
        discretized_data.drop(columns='Unnamed: 0', inplace=True)
     
        
        with open("graph/"+db_name+"/graph_"+ graph_type.lower() + '_' + discretization_type,"rb" ) as f:
            graph_data = pickle.load(f)

        build_graph_attributes(discretized_data, graph_data["graph"], graph_data["descriptors"], target, db_name, alpha,  graph_type, label, discretization_type)
        
    else:
        if graph_type == "COM":
            with open('graph/'+ db_name + '/complete_graph' ,"rb" ) as f:
                graph_data = pickle.load(f)
            
            trainset = pd.read_csv("data/preprocessed/"+ db_name +"/preprocessed_data_train.csv", keep_default_na=False, na_values=[""])
            testset = pd.read_csv("data/preprocessed/"+ db_name +"/preprocessed_data_test.csv", keep_default_na=False, na_values=[""])
            trainset.drop(columns=['Unnamed: 0'], inplace=True)
            testset.drop(columns=['Unnamed: 0'], inplace=True)
                
            compute_degree_centralities(graph_data["graph"], trainset, testset, db_name, graph_type, target)    
        else:
            with open('graph/'+ db_name + '/complete_graph' ,"rb" ) as f:
                graph_data = pickle.load(f)
                
            trainset = pd.read_csv("data/preprocessed/"+ db_name +"/preprocessed_data_train.csv", keep_default_na=False, na_values=[""])
            testset = pd.read_csv("data/preprocessed/"+ db_name +"/preprocessed_data_test.csv", keep_default_na=False, na_values=[""])
            trainset.drop(columns=['Unnamed: 0'], inplace=True)
            testset.drop(columns=['Unnamed: 0'], inplace=True)    
                
                 
        
if __name__ == "__main__":
    args = sys.argv[1:]
    target = args[0]
    db_name = args[1]
    graph_type = args[2].lower()
    
    if len(args) > 3:
        alpha = args[3]
        alpha = float(alpha) 
        discretization_type = args[4].lower()
        label = args[5]
        
        
        
        main(target, db_name, graph_type, alpha, label, discretization_type)
        
    else:     
        main(target, db_name, graph_type, None, None, None)
    
   