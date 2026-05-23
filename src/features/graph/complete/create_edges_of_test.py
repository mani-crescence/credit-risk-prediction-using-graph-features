import pandas as pd 
import sys , os, shutil
from itertools import islice 
from ....tools.graph import gower_distance



def create_edges(data_dicts, col_max, col_min, cols, src, dst, 
                    directory, start, end):
    edges = []
    data = {}
    N = len(data_dicts) 
        
    for k , (i, loan1) in islice(enumerate(data_dicts.items()), 0, N-1):
        for _,  (j, loan2) in islice(enumerate(data_dicts.items()), k + 1, N):
            if i!=j:
                w = gower_distance(loan1, loan2, cols, col_max, col_min)
                edges.append((src + str(i), dst + str(j), w))
   
    data["edges"] = edges
    data["start"] = start
    data["end"] = end
    
    print(directory + "edges" + str(start) + "_" + str(end))
    with open(directory + "edges" + str(start) + "_" + str(end), "w") as file:
        file.write(str(data))

if __name__  == "__main__":
    args =  sys.argv[1:]   
    db_name = args[0]
    start = int(args[1])
    end = int(args[2])
    _path = args[3]
    _dir = args[4]
    
    directory = _dir + db_name + '/subsets/test/' 
    
    os.makedirs(directory, exist_ok=True)
                
    
    testset = pd.read_csv(_path, keep_default_na = False, na_values = [""])
    testset.drop(columns=['Unnamed: 0', 'st'], inplace=True, errors='ignore')
    col_max = testset.max()
    col_min = testset.min()
    cols = testset.columns
    testset = testset.iloc[start:end] 
    test_dicts = {i: row.to_dict() for i, row in testset.iterrows()} 
    
    create_edges(test_dicts, col_max, col_min, cols, 'ts_u', 'ts_u', 
                directory, start, end)
        
    
    
    
        
        
        
    
