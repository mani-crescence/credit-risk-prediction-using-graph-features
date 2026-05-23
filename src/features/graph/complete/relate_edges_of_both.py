import pandas as pd 
import sys , os, shutil
from itertools import islice 
from ....tools.graph import gower_distance



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
    train_path = args[5]
    test_path = args[6]
    _dir = args[7]
 
    
    directory = _dir + db_name + '/related/both/' 
    os.makedirs(directory, exist_ok=True)
    
    trainset = pd.read_csv(train_path, keep_default_na=False, na_values=[""])
    trainset.drop(columns=['Unnamed: 0', 'st'], inplace=True, errors='ignore')
    testset = pd.read_csv(test_path, keep_default_na=False, na_values=[""])
    testset.drop(columns=['Unnamed: 0', 'st'], inplace=True, errors='ignore')
    col_max = testset.max()
    col_min = testset.min()
    cols = testset.columns
    set1 = trainset.iloc[start1:end1] 
    set2 = testset.iloc[start2:end2] 
    
    data_dicts1 = {i: row.to_dict() for i, row in set1.iterrows()} 
    data_dicts2 = {i: row.to_dict() for i, row in set2.iterrows()} 
    
    relate_graphs(data_dicts1, data_dicts2, 'tr_u', 'ts_u', start1, end1, start2, end2, directory)
        
        
        
    
        
        
    
