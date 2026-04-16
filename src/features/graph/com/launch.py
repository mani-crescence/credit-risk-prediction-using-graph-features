import subprocess
import sys, ast, os
from dotenv import load_dotenv
import pandas as pd 
from itertools import combinations, product
from concurrent.futures import ProcessPoolExecutor

def build_complete_graph(db_name):
    trainset = pd.read_csv("data/preprocessed/"+ db_name +"/preprocessed_data_train.csv", keep_default_na=False, na_values=[""])
    trainset.drop(columns=['Unnamed: 0'], inplace=True)
    
    
    cpu_count = os.cpu_count()
    size = trainset.shape[0]
    step = int(size / cpu_count)
    end = step
    start = 0
    
    commands = []    
    
    while start < size - 1:
        if step + start >= size - 1 :
            end = size
         
        commands.append(""" make run_build_edges_com_{0} DB_NAME={1} START={2} END={3} TYPE={4} """.
                            format(*[db_name.lower(), db_name.lower(), start, end, 'train']))
       
        end += step
        start += step
      
   
    processes = []
    
    for cmd in commands:
        process = subprocess.Popen(cmd, shell=True)
        processes.append(process)
        
    for p in processes:
        p.wait() 
         
  
    testset = pd.read_csv("data/preprocessed/"+ db_name +"/preprocessed_data_test.csv", keep_default_na=False, na_values=[""])    
    testset.drop(columns=['Unnamed: 0'], inplace=True)
    
   
    commands = []    
    
    size = testset.shape[0]
    step = int(size / cpu_count)
    end = step
    start = 0
    
    if(cpu_count > step):
        print("start =>", 0, ", end =>", size)
        commands.append(""" make run_build_edges_com_{0} DB_NAME={1}  START={2} END={3} TYPE={4} """.
                            format(*[db_name.lower(), db_name.lower(), 0, size, 'test']))
    else:
        
        while start < size - 1:
            
            if step + start >= size - 1 :
                end = size
                
            print("start =>", start, ", end =>", end)    
            commands.append(""" make run_build_edges_com_{0} DB_NAME={1}  START={2} END={3} TYPE={4} """.
                            format(*[db_name.lower(), db_name.lower(), start, end, 'test']))
            
        
            end += step
            start += step

    processes = []
    
    for cmd in commands:
        process = subprocess.Popen(cmd, shell=True)
        processes.append(process)
        
    for p in processes:
        p.wait()   
  
def launch_relate_edges(db_name):
    
    # CONNECTIONS BETWEEN TRAINING EDGES
    directory = 'graph/'+ db_name + '/subsets/train/' 
      
    commands = []  
    subset_train_data = {}
    for item in os.listdir(directory):
        path = os.path.join(directory, item)
        with open(path, 'r') as file:
            data = ast.literal_eval(file.read())
        subset_train_data[path] = (data['start'], data['end'])
  
    count = 0
        
    for (path1, (start1, end1)), (path2, (start2, end2)) in combinations(subset_train_data.items(), 2):
        commands.append(""" make run_relate_edges_com_{0} DB_NAME={1} START1={2} END1={3} START2={4} END2={5} TYPE={6} PATH1={7} PATH2={8}""".
        format(db_name.lower(), db_name.lower(), start1, end1, start2, end2, 'train', path1, path2)) 
        # print(path1, path2)
        count += 1
        
        if (count % 8 == 0):
            
            processes = []    
                
            for cmd in commands:
                process = subprocess.Popen(cmd, shell=True)
                processes.append(process)
                
            for p in processes:
                p.wait()     
                print(f"Discretization command '{p}' completed.")  
            
            commands = []
        
   
    
    processes = []    
                
    for cmd in commands:
        process = subprocess.Popen(cmd, shell=True)
        processes.append(process)
        
    for p in processes:
        p.wait()     
        print(f"modeling train-train command '{p}' completed.")          
       
        
    # CONNECTIONS BETWEEN TEST EDGES    
    commands = []  
    subset_test_data = {}
    
    directory = 'graph/'+ db_name + '/subsets/test/'
    
    for item in os.listdir(directory):
        path = os.path.join(directory, item)
        with open(path, 'r') as file:
            data = ast.literal_eval(file.read())
        subset_test_data[path] = (data['start'], data['end'])

    if (len(subset_test_data) == 1):
        path, (start, end) = list(subset_test_data.items())[0]
        commands.append(""" make run_relate_edges_com_{0} DB_NAME={1} START1={2} END1={3} START2={4} END2={5} TYPE={6} PATH1={7} PATH2={8}""".
         format(db_name.lower(), db_name.lower(), start, end, start, end, 'test', path, path))
        
    else:     
        count = 0
        
        for (path1, (start1, end1)), (path2, (start2, end2)) in combinations(subset_test_data.items(), 2):
             commands.append(""" make run_relate_edges_com_{0} DB_NAME={1} START1={2} END1={3} START2={4} END2={5} TYPE={6} PATH1={7} PATH2={8}""".
             format(db_name.lower(), db_name.lower(), start1, end1, start2, end2, 'test', path1, path2))   
             
             count += 1  
             
             if (count % 8 == 0):
                processes = []    
                    
                for cmd in commands:
                    process = subprocess.Popen(cmd, shell=True)
                    processes.append(process)
                    
                for p in processes:
                    p.wait()     
                    print(f"Discretization command '{p}' completed.")  
                
                commands = []
   
    processes = []    
                
    for cmd in commands:
        process = subprocess.Popen(cmd, shell=True)
        processes.append(process)
        
    for p in processes:
        p.wait()     
        print(f"modeling test-test command '{p}' completed.")         
    
   
   # CONNECTIONS BETWEEN TRAINING EDGES AND TEST EDGES 
    count = 0
    commands = []
    for ((path1,(start1, end1)), (path2,(start2, end2))) in product(subset_train_data.items(), subset_test_data.items()):
        
         commands.append(""" make run_relate_edges_com_{0} DB_NAME={1} START1={2} END1={3} START2={4} END2={5} TYPE={6} PATH1={7} PATH2={8}""".
         format(db_name.lower(), db_name.lower(), start1, end1, start2, end2, 'mix', '\"' + str(path1) + '\"', '\"'+ str(path2)+ '\"' ))
         
         count += 1  
             
         if (count % 8 == 0):
                processes = []    
                    
                for cmd in commands:
                    process = subprocess.Popen(cmd, shell=True)
                    processes.append(process)
                    
                for p in processes:
                    p.wait()     
                    print(f"Discretization command '{p}' completed.")  
                
                commands = []
   
    processes = [] 
           
    for cmd in commands:
        process = subprocess.Popen(cmd, shell=True)
        processes.append(process)
        
    for p in processes:
        p.wait()     
        print(f"modeling train-test command '{p}' completed.")         

def launch_silm(db_name):
    command = """ make run_compute_descriptors_com_{0}  BD_NAME={1} GRAPH_TYPE={2}  """.format(*[db_name.lower(), db_name.lower(), "com"])
    
    process = subprocess.Popen(command, shell=True)

    process.wait()


if __name__ == "__main__":
    args = sys.argv[1:]
    db_name = args[0]
    
    build_complete_graph(db_name)
    launch_relate_edges(db_name)
    launch_silm(db_name)