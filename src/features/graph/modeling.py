from tools.graph import *
import sys, os, pickle, ast
import pandas as pd

def main(data, graph_type, db_name, target, discretization_type = None):
    if discretization_type is not None:
        if graph_type == 'BIP':
            original_graph = graph_bipartite_modality(None, data, None, discretization_type)
            descriptors_attributes = [node for node, data_ in original_graph.nodes(data=True) if data_['type'] == 'attribute']
            graph_data = {"graph": original_graph, "descriptors": descriptors_attributes}  
            directory='outputs/'+db_name+'/graph/'  
            os.makedirs(directory, exist_ok=True)
            with open(directory + 'graph_' + graph_type.lower() + '_' + discretization_type, 'wb') as file:
                pickle.dump(graph_data, file)
                
        elif graph_type == 'MOD':
            original_graph = graph_modality(None, data, None, discretization_type)
            descriptors_attributes = [node for node, data_ in original_graph.nodes(data=True) if data_['type'] == 'attribute']
            graph_data = {"graph": original_graph, "descriptors": descriptors_attributes}  
            directory='outputs/'+db_name+'/graph/'  
            os.makedirs(directory, exist_ok=True)
            with open(directory + 'graph_' + graph_type.lower() + '_' + discretization_type, 'wb') as file:
                pickle.dump(graph_data, file)
    else:
      descriptors_attributes = []
      original_graph = graph_loans(None, data, target, None)
      target_values = data[target].unique()
      for i in target_values:
        descriptors_attributes.append(target+"_loan_"+str(i))
        
      graph_data = {"graph": original_graph, "descriptors": descriptors_attributes}  
      directory='outputs/'+db_name+'/graph/'  
      os.makedirs(directory, exist_ok=True)
      with open(directory + 'graph_' + graph_type.lower() , 'wb') as file:
         pickle.dump(graph_data, file)
     
    
if __name__ == "__main__":
    args =  sys.argv[1:]   
    db_name = args[0]
    target = args[1]
    discretization_type = args[2].lower()
    graph_type = args[3]
    
    print(f"Discretization type: {discretization_type}")
    
    partial_preprocessed_data  = pd.read_csv("outputs/"+db_name+"/preprocessed_sets/partial_preprocessed_data.csv", keep_default_na=False)
    partial_preprocessed_data.set_index('Unnamed: 0', inplace=True)
    partial_preprocessed_data.index.name = None
    
    if discretization_type == 'None':
        discretization_type = ast.literal_eval(discretization_type)
        trainset = pd.read_csv("outputs/" + db_name + "/data/classic/trainset.csv")
        trainset.set_index('Unnamed: 0', inplace=True)
        trainset.index.name = None
        index = trainset.index
        data = partial_preprocessed_data.loc[index]
        main(data, graph_type, db_name, target, discretization_type)
    else:
        data_file ='outputs/'+ db_name +'/data/discretized/graph/discretized_train_' + discretization_type + '.csv'
        discretized_data_train  = pd.read_csv(data_file, dtype='int')
        discretized_data_train.set_index('Unnamed: 0', inplace=True)
        discretized_data_train.index.name = None
        index = discretized_data_train.index
        partial_preprocessed_data = partial_preprocessed_data.loc[index]
        target_data = partial_preprocessed_data[target]
        categorical_data = partial_preprocessed_data.select_dtypes("object")
        data = pd.concat([discretized_data_train, categorical_data, target_data], axis=1)
        main(data,  graph_type, db_name, target, discretization_type)
   
    
    
    
    
    