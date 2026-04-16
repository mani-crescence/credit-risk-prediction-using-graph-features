import subprocess
import sys, ast, os
from dotenv import load_dotenv
import numpy as np
import pandas as pd 
import logging
from itertools import combinations, product
from concurrent.futures import ProcessPoolExecutor

 

discretization_types =  ["UNS", "SUP"]#, "SUP"]#, ]#, "SUP"]
discretization_for_attributes_types = ["UNS_", "SUP_"]
alphas = [0.1] #[0.2, 0.3, 0.4]  #[0.2, 0.3, 0.4] #[0.2, 0.3, 0.4]    # [0.2, 0.3, 0.4] # #  [0.5, 0.6, 0.7] ## [0.5, 0.6, 0.7]  [0.1] # [0.8, 0.85, 0.9]  # #[0.5, 0.6, 0.7] #[0.8, 0.85, 0.9] # [0.1]## # [0.1]## [0.2, 0.3, 0.4 ]  #, 0.8, 0.85, 0.9]#, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9]#, [0.1, 0.2] # 0.2,  0.3]# 0.1, 0.2, 0.3, 0.4, 0.5]#, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7
process_type_prediction = ["UNS", "SUP", "SUP_", "UNS_"]
plot_type = ["UNS","SUP"] 
pagerank_type = ["PER", "GLO"]
graph_types = ["GUI"]#, "MOD", "BIP"] #"LOAN", "MOD", "BIP", "COM" 
graph_types1 = ["BIP", "MOD", "GUI"]
graph_type_for_prediction = ["MOD", "BIP"]
graphs = ["bip", "bip", "mod", "mod", None, None]
discretizations = ["uns", "sup", "uns", "sup", "na", None]
models =[ "log", "svm", "rf", "dtree", "lda", "xgb"]#, "mlp"] #["log", "svm", ]
metrics = ["acc", "f1"]

load_dotenv()


def launch_split(db_name):
    commands = []
    commands.append(""" make run_splitting_{0} DB_NAME={1}""".format(*[db_name.lower(), db_name.lower()]))

    processes = []   
    for cmd in commands:
        process = subprocess.Popen(cmd, shell=True)
        processes.append(process)
        
    for p in processes:
        p.wait()

def launch_build_engine_for_preprocessing(db_name):
    commands = []
    commands.append(""" make run_engine_building_pre_{0} DB_NAME={1}""".format(*[db_name.lower(), db_name.lower()]))

    processes = []   
    for cmd in commands:
        process = subprocess.Popen(cmd, shell=True)
        processes.append(process)
        
    for p in processes:
        p.wait()
    
def launch_preprocess_train(db_name):
    commands = []
    commands.append(""" make run_preprocess_train_{0} DB_NAME={1}""".format(*[db_name.lower(), db_name.lower()]))

    processes = []   
    for cmd in commands:
        process = subprocess.Popen(cmd, shell=True)
        processes.append(process)
        
    for p in processes:
        p.wait()
             
def launch_preprocess_test(db_name):
    commands = []
    commands.append(""" make run_preprocess_test_{0} DB_NAME={1}""".format(*[db_name.lower(), db_name.lower()]))

    processes = []   
    for cmd in commands:
        process = subprocess.Popen(cmd, shell=True)
        processes.append(process)
        
    for p in processes:
        p.wait()        

def launch_sampling(db_name):
    commands = [""" make run_sampling_{0} DB_NAME={1}""".format(*[db_name.lower(), db_name.lower()])]
    processes = []
    for cmd in commands:
        process = subprocess.Popen(cmd, shell=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        processes.append(process)

    for p in processes:
        p.wait()

    # log_error(processes)

def launch_build_engine_for_discretization(db_name):
    commands = []
    
    for type in discretization_types:
            commands.append(""" make run_engine_building_disc_{0} DB_NAME={1} DISCRETIZATION_TYPE={2} """.format(*[db_name.lower(), db_name.lower(),  type]))

    processes = []   
    for cmd in commands:
        process = subprocess.Popen(cmd, shell=True)
        processes.append(process)
        
    for p in processes:
        p.wait()
   
def launch_disc(db_name):
    commands = []
    for label in ["train", "test"]:
         for discretization_type  in discretization_types:
             commands.append("""make run_discretization_{0} DB_NAME={1} DISCRETIZATION_TYPE={2} LABEL={3} """.format(*[db_name.lower(), db_name.lower(), discretization_type, label]))
   
    processes = []
    
    for cmd in commands:
        process = subprocess.Popen(cmd, shell=True)
        processes.append(process)
        
    for p in processes:
        p.wait()     
        print(f"Discretization command '{p}' completed.")

    # log_error(processes, db_name, "Discretization")


def launch_conf(db_name):
    commands = []

    commands.append("""make run_make_configurations_{0}  DISCRETIZATION_TYPE={1} GRAPH_TYPE={2}""".format(*[db_name.lower(), None, None]))

    for graph_type in graph_types:
        # for graph_type in graph_types:
         if graph_type == "COM" or graph_type == "GUI": 
            commands.append("""make run_make_configurations_{0}  DISCRETIZATION_TYPE={1} GRAPH_TYPE={2}""".format(*[db_name.lower(), None, graph_type]))

        #  for discretization_type in discretization_types:
        #     commands.append("""make run_make_configurations_{0}  DISCRETIZATION_TYPE={1} GRAPH_TYPE={2}""".format(*[db_name.lower(), discretization_type, graph_type]))

    processes = []
    for cmd in commands:
        process = subprocess.Popen(cmd, shell=True)
        processes.append(process)

    for p in  processes:
        p.wait()

def launch_predict(db_name):
    commands = []
    for graph_type in graph_types:
        if graph_type == "COM" or graph_type == "GUI":
            train_path = 'data/graph_features/'+db_name.lower()+'/'+graph_type.lower()+'/new_features_train.csv'
            test_path = 'data/graph_features/'+db_name.lower()+'/'+graph_type.lower()+'/new_features_test.csv'
            config_path = "data/configurations/" + db_name.lower()+"/configuration_" +graph_type.lower() + ".txt"

            commands.append("""make run_make_predictions_{0} DB_NAME={1} TRAIN_PATH={2} TEST_PATH={3}  DISCRETIZATION_TYPE={4} GRAPH_TYPE={5} CONFIG_PATH={6} ALPHA={7}""".format(*[db_name.lower(), db_name.lower(), train_path, test_path, None, graph_type, config_path, None]))
        else:
            for disc_type in discretization_types:
                train_directory = 'data/graph_features/'+ db_name.lower() +'/'+ disc_type.lower()+ "/" + graph_type .lower() + '/train'
                test_directory = 'data/graph_features/'+ db_name.lower() +'/'+ disc_type.lower()+ "/" + graph_type .lower() + '/test'
                
                config_path = "data/configurations/" + db_name.lower()+"/configuration_" +graph_type.lower()+"_"+disc_type.lower()+ ".txt"

                for alpha in alphas: 
                    train_path = train_directory + '/new_features_' + str(alpha) + '.csv'
                    test_path = test_directory + '/new_features_' + str(alpha) + '.csv'
                    commands.append("""make run_make_predictions_{0} DB_NAME={1} TRAIN_PATH={2} TEST_PATH={3}  DISCRETIZATION_TYPE={4} GRAPH_TYPE={5} CONFIG_PATH={6} ALPHA={7}""".format(*[db_name.lower(), db_name.lower(), train_path, test_path, disc_type, graph_type, config_path, alpha]))

    # for disc_type in discretization_types :
    #     train_path = 'outputs/'+db_name.lower()+'/data/discretized/new_discretized_train_'+disc_type.lower()+'.csv'
    #     test_path = 'outputs/'+db_name.lower()+'/data/discretized/new_discretized_test_'+disc_type.lower()+'.csv'
    #     config_path = "outputs/"+db_name.lower()+"/configurations/configuration_" + disc_type.lower() + ".txt"
    #     commands.append("""make run_make_predictions_{0}  DB_NAME={1} TRAIN_PATH={2} TEST_PATH={3} DISCRETIZATION_TYPE={4} CONFIG_PATH={5} GRAPH_TYPE={6} """.format(*[db_name.lower(), db_name.lower(), train_path, test_path, disc_type, config_path,  None]))


    processes = []

    for cmd in commands:
        process = subprocess.Popen(cmd, shell=True)
        processes.append(process)

    for p in  processes:
        p.wait()
    # log_error(processes, db_name, "Prediction With New Descriptors")

def launch_predict_classic(db_name):
    train_path = 'data/preprocessed/'+db_name.lower()+'/preprocessed_data_train.csv'
    test_path = 'data/preprocessed/'+db_name.lower()+'/preprocessed_data_test.csv'
    config_path = "data/configurations/"+ db_name+"/configuration_classic.txt" 
    command = """make run_make_predictions_{0} DB_NAME={1} TRAIN_PATH={2} TEST_PATH={3} DISCRETIZATION_TYPE={4} GRAPH_TYPE={5} CONFIG_PATH={6} """.format(*[db_name.lower(), db_name.lower(), train_path, test_path, None, None, config_path, "CLASSIC"])
    process = subprocess.run(command, shell=True)
    # log_error([process], db_name, "Prediction For Ordinary Attributes")

def launch_print(db_name):
    commands = ["""make run_print_{0} DB_NAME={1}  DISCRETIZATION_TYPE={2} GRAPH_TYPE={3} """.format(
        *[db_name.lower(), db_name.lower(), None, None]),
        # """make run_print_{0} DB_NAME={1}  DISCRETIZATION_TYPE={2} GRAPH_TYPE={3} """.format(
        #     *[db_name.lower(), db_name.lower(), 'na', None])
        ]

    for graph in graph_types:
        if graph == "COM"  or graph == "GUI":
            commands.append("""make run_print_{0} DB_NAME={1}  DISCRETIZATION_TYPE={2} GRAPH_TYPE={3} """.format(
                *[db_name.lower(), db_name.lower(), None, graph]))
        else:
            for discretization in discretization_types:
                commands.append(
                    """make run_print_{0} DB_NAME={1}  DISCRETIZATION_TYPE={2} GRAPH_TYPE={3} """.format(
                        *[db_name.lower(), db_name.lower(), discretization, graph]))


    processes = []
    
    for cmd in commands:
        process = subprocess.Popen(cmd, shell=True)
        processes.append(process)
        
    for p in processes: 
        p.wait()
    # log_error(processes, db_name, "Printing")
        
def launch_plot(db_name):
    commands = []

    # for model in models:
    #     train_descriptors_paths = []
    #     test_descriptors_paths = []
    #     best_alphas_path = 'outputs/'+ db_name +'/results/predictions/loan/best_alpha_values.txt'
    #     with open(best_alphas_path, "r") as file:
    #         alphas = ast.literal_eval(file.read())
    #     alpha = alphas[model]
    #     train_descriptors_paths.append('\'outputs/' + db_name + '/new_descriptors/LOAN/train/new_descriptors_data_LOAN_' + str(alpha) + '.csv\'')
    #     test_descriptors_paths.append('\'outputs/' + db_name + '/new_descriptors/LOAN/test/new_descriptors_data_LOAN_' + str(alpha) + '.csv\'')
    #     commands.append("""make run_plot_{0} DB_NAME={1} TRAIN_DESCRIPTORS_PATHS={2} TEST_DESCRIPTORS_PATHS={3} MODEL={4} DISCRETIZATION_TYPE={5} """.format(
    #             *[db_name.lower(), db_name.lower(), train_descriptors_paths, test_descriptors_paths, model, None]))

    for model in models:
        for discretization_type in discretization_types:
            train_descriptors_paths = []
            test_descriptors_paths = []
            for graph in ["BIP", "MOD"]:
                best_alphas_path = 'outputs/general_results/results/'+ db_name +'/percent/predictions/' + graph.lower()  + '/best_alpha_values_' + graph.lower() + '_' + discretization_type.lower() + '.txt'
                with open(best_alphas_path, "r") as file:
                    alphas = ast.literal_eval(file.read())
                # exit(alphas)
                alpha = alphas[model]
                train_descriptors_paths.append('outputs/' + db_name + '/new_descriptors/' + graph.lower() + '/train/new_descriptors_data' + '_' + discretization_type.lower() + '_' +  graph.lower() + '_' + str(alpha) + '.csv')
                test_descriptors_paths.append('outputs/' + db_name + '/new_descriptors/' + graph.lower() + '/test/new_descriptors_data' + '_' + discretization_type.lower()  + '_' + graph.lower() + '_' + str(alpha) + '.csv')
            # print(train_descriptors_paths, test_descriptors_paths)
            subprocess.run("""make run_plot_{0} DB_NAME={1} TRAIN_DESCRIPTORS_PATHS={2} TEST_DESCRIPTORS_PATHS={3}  MODEL={4} DISCRETIZATION_TYPE={5} """.format(
                *[db_name.lower(), db_name.lower(), '\"' + str(train_descriptors_paths) + '\"',
                  f'\"{test_descriptors_paths}\"', model, discretization_type]), shell=True)
            # commands.append("""make run_plot_{0} DB_NAME={1} TRAIN_DESCRIPTORS_PATHS={2} TEST_DESCRIPTORS_PATHS={3}  MODEL={4} DISCRETIZATION_TYPE={5} """.format(
            #     *[db_name.lower(), db_name.lower(),'\"' + str(train_descriptors_paths) + '\"', f'\"{test_descriptors_paths}\"', model, discretization_type]))
            # print("after launching ....", db_name)

    # processes = []
    #
    # for cmd in commands:
    #     process = subprocess.Popen(cmd, shell=True)#, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    #     processes.append(process)
    #
    # for p in processes:
    #     p.wait()
    # # log_error(processes, db_name, "Plot Making")

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
         
        commands.append(""" make run_build_edges_{0} DB_NAME={1} START={2} END={3} TYPE={4} """.
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
        commands.append(""" make run_build_edges_{0} DB_NAME={1}  START={2} END={3} TYPE={4} """.
                            format(*[db_name.lower(), db_name.lower(), 0, size, 'test']))
    else:
        
        while start < size - 1:
            
            if step + start >= size - 1 :
                end = size
                
            print("start =>", start, ", end =>", end)    
            commands.append(""" make run_build_edges_{0} DB_NAME={1}  START={2} END={3} TYPE={4} """.
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
        commands.append(""" make run_relate_edges_{0} DB_NAME={1} START1={2} END1={3} START2={4} END2={5} TYPE={6} PATH1={7} PATH2={8}""".
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
        print(f"Discretization command '{p}' completed.")          
       
        
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
        commands.append(""" make run_relate_edges_{0} DB_NAME={1} START1={2} END1={3} START2={4} END2={5} TYPE={6} PATH1={7} PATH2={8}""".
         format(db_name.lower(), db_name.lower(), start, end, start, end, 'test', path, path))
        
    else:     
        count = 0
        
        for (path1, (start1, end1)), (path2, (start2, end2)) in combinations(subset_test_data.items(), 2):
             commands.append(""" make run_relate_edges_{0} DB_NAME={1} START1={2} END1={3} START2={4} END2={5} TYPE={6} PATH1={7} PATH2={8}""".
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
        print(f"Discretization command '{p}' completed.")         
    
   
   # CONNECTIONS BETWEEN TRAINING EDGES AND TEST EDGES 
    count = 0
    commands = []
    for ((path1,(start1, end1)), (path2,(start2, end2))) in product(subset_train_data.items(), subset_test_data.items()):
        
         commands.append(""" make run_relate_edges_{0} DB_NAME={1} START1={2} END1={3} START2={4} END2={5} TYPE={6} PATH1={7} PATH2={8}""".
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
        print(f"Discretization command '{p}' completed.")         



if __name__ == "__main__":
    args = sys.argv[1:]
    db_name = args[0]
    # launch_split(db_name)
    # launch_build_engine_for_preprocessing(db_name)
    # launch_preprocess_train(db_name)
    # launch_preprocess_test(db_name)
    # launch_build_engine_for_discretization(db_name)
    # launch_disc(db_name)
    
    # build_complete_graph(db_name)
    # launch_relate_edges(db_name)
    
    # launch_graph_modeling(db_name)
    
    # launch_silm(db_name)
    
    # launch_conf(db_name)
    # launch_predict_classic(db_name)
    # launch_predict(db_name)
    # launch_print(db_name)
    # launch_plot(db_name)
 
 
    
# def launch_graph_modeling(db_name):
    
#     commands = []
#     for graph_type in graph_types:
#         if graph_type == "COM":  
#             commands.append("""make run_graph_modeling_{0} DB_NAME={1}  GRAPH_TYPE={2}  """.format(*[db_name.lower(), db_name.lower(),graph_type ]))
#         else:    
#             for discretization_type in discretization_types:
#                 commands.append("""make run_graph_modeling_{0} DB_NAME={1} GRAPH_TYPE={2} DISCRETIZATION_TYPE={3} """.format(*[db_name.lower(), db_name.lower(), graph_type, discretization_type]))
    
#     processes = []
    
#     for cmd in commands:
        
#         process = subprocess.Popen(cmd, shell=True)
#         processes.append(process)
        
#     for p in processes:
#         p.wait()

# def launch_silm(db_name):
#     commands = []
    
#     for graph_type in graph_types:
#         if graph_type == "COM" or graph_type == "GUI":  
#            commands.append(""" make run_compute_descriptors_{0}  BD_NAME={1} GRAPH_TYPE={2}  """.format(*[db_name.lower(), db_name.lower(), graph_type]))
#         # else:
#         #     for label in ["train", "test"]:#, "test"]: ,
#         #         for discretization_type in discretization_types:
#         #             for alpha in alphas:
#         #                 commands.append(""" make run_compute_descriptors_{0}  BD_NAME={1} GRAPH_TYPE={2} ALPHA={3}  DISCRETIZATION_TYPE={4} LABEL={5} """.format(*[db_name.lower(), db_name.lower(), graph_type, alpha, discretization_type, label]))
#     processes = []
    
#     for cmd in commands:
#         process = subprocess.Popen(cmd, shell=True)
#         processes.append(process)
        
#     for p in processes:
#         p.wait()

#     # log_error(processes, db_name, "New Descriptors Building")


