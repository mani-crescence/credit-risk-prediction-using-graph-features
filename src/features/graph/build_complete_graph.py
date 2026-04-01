import pandas as pd 
import sys , os
import networkx as nx


edges_for_same_subset = []
edges_for_different_subset = []

def gower_distance(u, v, attributes, max = None, min = None):
    sum = 0
    attributes_length = len(u)
    for i in attributes:
        if type(u[i]) is str:
            
            if u[i] == v[i]:
                sum += 0
            else:
                sum += 1
        else:
            if (max[i] - min[i]) == 0:
                sum += abs(u[i] - v[i])
            else:
                sum += abs(u[i] - v[i]) / (max[i] - min[i])

    return sum / attributes_length

def build_sub_graph(data_dicts, col_max, col_min, cols, src, dst, 
                    directory, start, end):
    edges = []
    data = {}
        
    for i, loan1 in data_dicts.items():
        for j, loan2 in data_dicts.items():
            if i!=j:
                w = gower_distance(loan1, loan2, cols, col_max, col_min)
                edges.append((src + str(i), dst + str(j), w))
            
    edges_for_same_subset.append(edges)
    
    data["edges"] = edges_for_same_subset
    data["start"] = start
    data["end"] = end
    
    
    with open(directory + "edges" + str(start) + "_" + str(end), "w") as file:
        file.write(str(data))
    
def relate_graphs(data_dicts1, data_dicts2, src, dst):
    edges = []
    for i, loan1, j, loan2  in zip(data_dicts1.items(), data_dicts2.items()):
        w = gower_distance(loan1, loan2, cols, col_max, col_min)
        edges.append(( src + str(i), dst + str(j), w))
        
    edges_for_different_subset.append(edges)
   

if __name__  == "__main__":
    
    args =  sys.argv[1:]   
    db_name = args[0]
    start = int(args[1])
    end = int(args[2])
    type_of_set = args[3] 
    
    directory = 'graph/'+ db_name + '/subsets/' + type_of_set + '/' 
    os.makedirs(directory, exist_ok=True)
                
    
    if(type_of_set == 'train'):
        trainset = pd.read_csv("data/preprocessed/"+ db_name +"/preprocessed_data_train.csv", keep_default_na=False, na_values=[""])
        trainset.drop(columns=['Unnamed: 0'], inplace=True)
        col_max = trainset.max()
        col_min = trainset.min()
        cols = trainset.columns
        trainset = trainset.iloc[start:end-1] 
        train_dicts = {i: row.to_dict() for i, row in trainset.iterrows()} 
        
        build_sub_graph(train_dicts, col_max, col_min, cols, 'tr_u', 'tr_u', 
                    directory, start, end)
    else:
        testset = pd.read_csv("data/preprocessed/"+ db_name +"/preprocessed_data_test.csv", keep_default_na=False, na_values=[""])
        testset.drop(columns=['Unnamed: 0'], inplace=True)
        col_max = testset.max()
        col_min = testset.min()
        cols = testset.columns
        testset = testset.iloc[start:end-1] 
        test_dicts = {i: row.to_dict() for i, row in testset.iterrows()} 
        
        build_sub_graph(test_dicts, col_max, col_min, cols, 'ts_u', 'ts_u', 
                    directory, start, end)
        
    
    
    
        
        
        
    
