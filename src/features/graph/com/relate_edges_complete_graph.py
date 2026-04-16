import pandas as pd 
import sys , os
import networkx as nx
from itertools import islice 




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

def relate_graphs(data_dicts1, data_dicts2, src, dst, start1, end1, start2, end2, directory):
    edges = []
    data = {}
        
    for _, (i, loan1)  in islice(enumerate(data_dicts1.items()), 0, len(data_dicts1)):
        for _, (j, loan2) in islice(enumerate(data_dicts2.items()), 0, len(data_dicts2)):
            
            if src + str(i) != dst + str(j):
                w = gower_distance(loan1, loan2, cols, col_max, col_min)
                edges.append(( src + str(i), dst + str(j), w))
            
    
    data["edges"] = edges
    
    with open(directory + "edges" + str(start1) + "_" + str(end2), "w") as file:
        file.write(str(data))
    
if __name__  == "__main__":
    args =  sys.argv[1:]   
    db_name = args[0]
    start1 = int(args[1])
    end1 = int(args[2])
    start2 = int(args[3])
    end2 = int(args[4])
    type_of_set = args[5]
    path1 = args[6]
    path2 = args[7]
 
    
    if (type_of_set  == 'train'):
        directory = 'graph/'+ db_name + '/related/train/'  
        os.makedirs(directory, exist_ok=True)
        trainset = pd.read_csv("data/preprocessed/"+ db_name +"/preprocessed_data_train.csv", keep_default_na=False, na_values=[""])
        trainset.drop(columns=['Unnamed: 0'], inplace=True)
        col_max = trainset.max()
        col_min = trainset.min()
        cols = trainset.columns
        set1 = trainset.iloc[start1:end1] 
        set2 = trainset.iloc[start2:end2] 
        
        data_dicts1 = {i: row.to_dict() for i, row in set1.iterrows()} 
        data_dicts2 = {i: row.to_dict() for i, row in set2.iterrows()} 
        
        
        relate_graphs(data_dicts1, data_dicts2, 'tr_u', 'tr_u', start1, end1, start2, end2, directory)
        
    elif (type_of_set  == 'test'):
        directory = 'graph/'+ db_name + '/related/test/'  
        os.makedirs(directory, exist_ok=True)
        testset = pd.read_csv("data/preprocessed/"+ db_name +"/preprocessed_data_test.csv", keep_default_na=False, na_values=[""])
        testset.drop(columns=['Unnamed: 0'], inplace=True)
        col_max = testset.max()
        col_min = testset.min()
        cols = testset.columns
        set1 = testset.iloc[start1:end1] 
        set2 = testset.iloc[start2:end2] 
        
        data_dicts1 = {i: row.to_dict() for i, row in set1.iterrows()} 
        data_dicts2 = {i: row.to_dict() for i, row in set2.iterrows()} 
        
        relate_graphs(data_dicts1, data_dicts2, 'ts_u', 'ts_u', start1, end1, start2, end2, directory)
        
    else:
        directory = 'graph/'+ db_name + '/related/mix/'  
        os.makedirs(directory, exist_ok=True)
        trainset = pd.read_csv("data/preprocessed/"+ db_name +"/preprocessed_data_train.csv", keep_default_na=False, na_values=[""])
        trainset.drop(columns=['Unnamed: 0'], inplace=True)
        testset = pd.read_csv("data/preprocessed/"+ db_name +"/preprocessed_data_test.csv", keep_default_na=False, na_values=[""])
        testset.drop(columns=['Unnamed: 0'], inplace=True)
        col_max = testset.max()
        col_min = testset.min()
        cols = testset.columns
        set1 = trainset.iloc[start1:end1] 
        set2 = testset.iloc[start2:end2] 
        
        data_dicts1 = {i: row.to_dict() for i, row in set1.iterrows()} 
        data_dicts2 = {i: row.to_dict() for i, row in set2.iterrows()} 
        
        relate_graphs(data_dicts1, data_dicts2, 'tr_u', 'ts_u', start1, end1, start2, end2, directory)
        
        
        
    
        
        
    
