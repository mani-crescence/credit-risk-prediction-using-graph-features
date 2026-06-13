import shutil
import subprocess
import sys, ast, os
from dotenv import load_dotenv
import pandas as pd 
from itertools import combinations, product

discretization_types =  ["UNS", "SUP"] 
discretization_for_attributes_types = ["UNS_", "SUP_"]
alphas = [0.1] # [0.15, 0.5, 0.85]#, 0.3, 0.5, 0.7, 0.85, 0.9]    
process_type_prediction = ["UNS", "SUP", "SUP_", "UNS_"]
plot_type = ["UNS","SUP"] 
pagerank_type = ["PER", "GLO"]
graph_types = ["LIU"]  #"MOD", "BIP", "LOAN"]#, "MOD", "LOAN"]#, "MOD", "BIP"] 
graph_type_for_prediction = ["MOD", "BIP"]
graphs = ["bip", "bip", "mod", "mod", None, None]
discretizations = ["uns", "sup", "uns", "sup", "na", None]
models =[ "log", "svm", "rf", "dtree", "lda", "xgb"]
metrics = ["acc", "f1"]
state_of_art_graphs = ["LIU_V2", "GUI", "LIU_V1"]#, "LIU_V2",  
standard_proposed_graphs = ["MOD"]#, "MOD"] #,  
proposed_complete_graph = ["LOAN"]


load_dotenv()

def launch_preprocess(db_name):
    command = """ make run_preprocess_{0} DB_NAME={1}""".format(*[db_name.lower(), db_name.lower()])

    subprocess.run(command, shell=True)

def launch_build_engine_for_unsupervized_discretization(db_name):
    
    _path = 'data/preprocessed/'+ db_name +'/partial_preprocessed_data_train.feather' 
    _dir = 'engine/with_normalization/discretization/'
    
    command = """ make run_engine_building_unsupervised_discretization_{0} DB_NAME={1} _PATH={2} _DIR={3} """.format(
        *[db_name.lower(), db_name.lower(), _path, _dir])

    subprocess.run(command, shell=True)
 
def launch_build_engine_for_supervized_discretization(db_name):
    
    _path = 'data/preprocessed/'+ db_name +'/partial_preprocessed_data_train.feather'
    _dir = "engine/with_normalization/discretization/"
    
    command = """ make run_engine_building_supervised_discretization_{0} DB_NAME={1} _PATH={2} _DIR={3} """.format(
        *[db_name.lower(), db_name.lower(), _path, _dir ] )

    subprocess.run(command, shell=True)
       
def launch_unsupervised_discretization(db_name):
    commands = []
    for label in ["train", "test"]:
        _path = 'data/preprocessed/'+ db_name +'/partial_preprocessed_data_' + label + '.feather'
        
        commands.append("""make run_unsupervised_discretization_{0} DB_NAME={1} _PATH={2} NORMALIZATION_LABEL={3}  DATA_LABEL={4} """.
                        format(*[db_name.lower(), db_name.lower(), _path, "with_normalization", label]))
   
    processes = []
    
    for cmd in commands:
        process = subprocess.Popen(cmd, shell=True)
        processes.append(process)
        
    for p in processes:
        p.wait()     
        print(f"Unsupervised discretization launch  completed.")        

def launch_supervised_discretization(db_name):
    commands = []
    for label in ["train", "test"]:
        _path = 'data/preprocessed/'+ db_name +'/partial_preprocessed_data_' + label + '.feather'
        
        commands.append("""make run_supervised_discretization_{0} DB_NAME={1} _PATH={2} NORMALIZATION_LABEL={3}  DATA_LABEL={4} """.
                        format(*[db_name.lower(), db_name.lower(), _path, "with_normalization", label]))
   
    processes = []
    
    for cmd in commands:
        process = subprocess.Popen(cmd, shell=True)
        processes.append(process)
        
    for p in processes:
        p.wait()     
        print(f"Supervized discretization launch  completed.")

def launch_graph_modeling(db_name):
    commands = []
    _dir = 'graph/with_normalization/' 
    
    for graph_type in standard_proposed_graphs:
        for discretization_type in discretization_types:
            train_path = "data/with_normalization/discretized/" + db_name.lower() + "/discretized_train_data_" + discretization_type.lower() + ".feather"
            test_path = "data/with_normalization/discretized/" + db_name.lower() + "/discretized_test_data_" + discretization_type.lower() + ".feather"
           
            
            commands.append("""make run_graph_modeling_""" + graph_type.lower() + """_{0} DB_NAME={1} GRAPH_TYPE={2} DISCRETIZATION_TYPE={3}  TRAIN_PATH={4} TEST_PATH={5} _DIR={6} """.
                        format(*[db_name.lower(), db_name.lower(), graph_type.lower(), discretization_type.lower(), train_path, test_path, _dir]))
    
    for graph_type in proposed_complete_graph:
         train_path = "data/preprocessed/"+ db_name +"/partial_preprocessed_data_train.feather"
         test_path = "data/preprocessed/"+ db_name +"/partial_preprocessed_data_test.feather"
         
         commands.append("""make run_graph_modeling_""" + graph_type.lower() + """_{0} DB_NAME={1} GRAPH_TYPE={2} TRAIN_PATH={3} TEST_PATH={4} _DIR={5} """.
                         format(*[db_name.lower(), db_name.lower(), graph_type.lower(), train_path, test_path, _dir]))
 

    for graph_type in ["LIU", "GUI"]: 
            commands.append("""make run_graph_modeling_""" + graph_type.lower() + """_{0} DB_NAME={1} GRAPH_TYPE={2} DISCRETIZATION_TYPE={3}  TRAIN_PATH={4} TEST_PATH={5} _DIR={6} """.
                        format(*[db_name.lower(), db_name.lower(), graph_type.lower(), None, None, None,  _dir]))
       
            
    processes = [] 
    
    for cmd in commands:
        
        process = subprocess.Popen(cmd, shell=True) 
        processes.append(process)
        
    for p in processes:
        p.wait()
        
def launch_compute_descriptors(db_name):
    
    _dir = 'data/with_normalization/graph_features/'
    _graph_dir = 'graph/with_normalization/'
    
    commands = []
    
    for graph_type in standard_proposed_graphs:
        for discretization_type in discretization_types:
            train_path = "data/with_normalization/discretized/" + db_name.lower() + "/discretized_train_data_" + discretization_type.lower() + ".feather"
            test_path = "data/with_normalization/discretized/" + db_name.lower() + "/discretized_test_data_" + discretization_type.lower() + ".feather"
           
            for alpha in alphas:
                commands.append(""" make run_compute_descriptors_""" + graph_type.lower() + """_{0}  BD_NAME={1} GRAPH_TYPE={2} ALPHA={3} DISCRETIZATION_TYPE={4} TRAIN_PATH={5} TEST_PATH={6} _DIR={7} GRAPH_DIR={8} """.
                                format(*[db_name.lower(), db_name.lower(), graph_type.lower(), alpha, discretization_type.lower(), train_path, test_path, _dir, _graph_dir]))
     
     
    # train_path = "data/preprocessed/"+ db_name +"/preprocessed_data_train.feather"
    # test_path = "data/preprocessed/"+ db_name +"/preprocessed_data_test.feather"
    # for graph_type in proposed_complete_graph:
    #      for alpha in alphas:
    #         commands.append(""" make run_compute_descriptors_loan_{0}  BD_NAME={1} GRAPH_TYPE={2} ALPHA={3}  TRAIN_PATH={4} TEST_PATH={5}  GRAPH_DIR={6} _DIR={7} """.
    #                     format(*[db_name.lower(), db_name.lower(), graph_type.lower(), alpha, train_path, test_path, _graph_dir, _dir]))
        
                     
    # for graph_type in state_of_art_graphs:  
    #     commands.append(""" make run_compute_descriptors_""" + graph_type.lower() + """_{0}  BD_NAME={1} GRAPH_TYPE={2} TRAIN_PATH={3} TEST_PATH={4} _DIR={5} GRAPH_DIR={6}  """.
    #                     format(*[db_name.lower(), db_name.lower(), graph_type.lower(), train_path, test_path, _dir, _graph_dir]))
        
    processes = []
    
    for cmd in commands:
        process = subprocess.Popen(cmd, shell=True)
        processes.append(process)
        
    for p in processes:
        p.wait()            
   
def launch_config_without_stepwise(db_name):
    
    classic_train_path = "data/preprocessed/"+ db_name +"/preprocessed_data_train.feather"
    save_dir = 'data/with_normalization/without_stepwise/configurations'
      
    commands = []

    commands.append("""make run_make_configurations_{0}  SAVE_DIR={1} DISCRETIZATION_TYPE={2} GRAPH_TYPE={3} CLASSIC_TRAIN_PATH={4} NEW_DESCRIPTOR_TRAIN_PATH={5} """.
                    format(*[db_name.lower(), save_dir, None, None, classic_train_path, classic_train_path]))
  
    for graph_type in state_of_art_graphs:
        
        new_descriptor_train_path = "data/with_normalization/graph_features/" + db_name.lower() + "/" + graph_type.lower() + "/new_features_train.feather"
        
        commands.append("""make run_make_configurations_{0}  SAVE_DIR={1} DISCRETIZATION_TYPE={2} GRAPH_TYPE={3} CLASSIC_TRAIN_PATH={4} NEW_DESCRIPTOR_TRAIN_PATH={5}""".
                        format(*[db_name.lower(), save_dir, None, graph_type, classic_train_path, new_descriptor_train_path]))
         

    for graph_type in standard_proposed_graphs:    
        for discretization_type in discretization_types:
            new_descriptor_train_path = "data/with_normalization/graph_features/" + db_name.lower() + "/" + graph_type.lower()  +"/"+ discretization_type.lower()  + '/train/' + "new_features_0.1.feather"
            
            commands.append("""make run_make_configurations_{0}  SAVE_DIR={1} DISCRETIZATION_TYPE={2} GRAPH_TYPE={3} CLASSIC_TRAIN_PATH={4} NEW_DESCRIPTOR_TRAIN_PATH={5}""".
                            format(*[db_name.lower(), save_dir, discretization_type, graph_type, classic_train_path, new_descriptor_train_path]))
            
    for graph_type in proposed_complete_graph:
        new_descriptor_train_path = "data/with_normalization/graph_features/" + db_name.lower() + "/" + graph_type.lower() + "/train/new_features_0.1.feather"
        
        commands.append("""make run_make_configurations_{0}  SAVE_DIR={1} DISCRETIZATION_TYPE={2} GRAPH_TYPE={3} CLASSIC_TRAIN_PATH={4} NEW_DESCRIPTOR_TRAIN_PATH={5}""".
                        format(*[db_name.lower(), save_dir, None, graph_type, classic_train_path, new_descriptor_train_path]))
                
            

    processes = []
    for cmd in commands:
        process = subprocess.Popen(cmd, shell=True)
        processes.append(process)

    for p in  processes:
        p.wait()
        
def launch_config_with_stepwise(db_name):
    
    classic_train_path = "data/preprocessed/"+ db_name +"/preprocessed_data_train.feather"
    save_dir = 'data/with_normalization/with_stepwise/configurations'
      
    commands = []

    commands.append("""make run_make_configurations_with_stepwise_{0}  SAVE_DIR={1} DISCRETIZATION_TYPE={2} GRAPH_TYPE={3} CLASSIC_TRAIN_PATH={4} NEW_DESCRIPTOR_TRAIN_PATH={5} """.
                    format(*[db_name.lower(), save_dir, None, None, classic_train_path, classic_train_path]))
  
    for graph_type in state_of_art_graphs:
        
        new_descriptor_train_path = "data/with_normalization/graph_features/" + db_name.lower() + "/" + graph_type.lower() + "/new_features_train.feather"
        
        commands.append("""make run_make_configurations_with_stepwise_{0}  SAVE_DIR={1} DISCRETIZATION_TYPE={2} GRAPH_TYPE={3} CLASSIC_TRAIN_PATH={4} NEW_DESCRIPTOR_TRAIN_PATH={5}""".
                        format(*[db_name.lower(), save_dir, None, graph_type, classic_train_path, new_descriptor_train_path]))
         

    for graph_type in standard_proposed_graphs:    
        for discretization_type in discretization_types:
            new_descriptor_train_path = "data/with_normalization/graph_features/" + db_name.lower() + "/" + graph_type.lower()  +"/"+ discretization_type.lower()  + '/train/' + "new_features_0.1.feather"
            
            commands.append("""make run_make_configurations_with_stepwise_{0}  SAVE_DIR={1} DISCRETIZATION_TYPE={2} GRAPH_TYPE={3} CLASSIC_TRAIN_PATH={4} NEW_DESCRIPTOR_TRAIN_PATH={5}""".
                            format(*[db_name.lower(), save_dir, discretization_type, graph_type, classic_train_path, new_descriptor_train_path]))
            
    for graph_type in proposed_complete_graph:
        new_descriptor_train_path = "data/with_normalization/graph_features/" + db_name.lower() + "/" + graph_type.lower() + "/train/new_features_0.1.feather"
        
        commands.append("""make run_make_configurations_with_stepwise_{0}  SAVE_DIR={1} DISCRETIZATION_TYPE={2} GRAPH_TYPE={3} CLASSIC_TRAIN_PATH={4} NEW_DESCRIPTOR_TRAIN_PATH={5}""".
                        format(*[db_name.lower(), save_dir, None, graph_type, classic_train_path, new_descriptor_train_path]))
                
            

    processes = []
    for cmd in commands:
        process = subprocess.Popen(cmd, shell=True)
        processes.append(process)

    for p in  processes:
        p.wait()        

def launch_stepwise_selection(db_name):
    train_path = "data/preprocessed/"+ db_name +"/preprocessed_data_train.feather"
    save_dir = 'data/with_normalization/with_stepwise/configurations/'+ db_name.lower() 
    
    command = """make run_select_features_{0} DB_NAME={1} TRAIN_PATH={2} SAVE_DIR={3} """.format(
        *[db_name.lower(), db_name.lower(), train_path, save_dir])
    
    subprocess.run(command, shell=True)
   
def launch_predict(db_name):
    save_dir = 'reports/with_normalization/without_stepwise/'+ db_name + '/metrics'
    classic_train_path = "data/preprocessed/"+ db_name +"/preprocessed_data_train.feather"
    classic_test_path = "data/preprocessed/"+ db_name +"/preprocessed_data_test.feather"
    classic_config_path = "data/with_normalization/without_stepwise/configurations/"+ db_name + "/configuration_classic.txt" 
    
    commands = []
    
    for graph_type in state_of_art_graphs:
            train_path = 'data/with_normalization/graph_features/' + db_name.lower() + '/' + graph_type.lower() + '/new_features_train.feather'
            test_path = 'data/with_normalization/graph_features/' + db_name.lower() + '/' + graph_type.lower() + '/new_features_test.feather'
            config_path = "data/with_normalization/without_stepwise/configurations/" + db_name.lower()+"/configuration_" + graph_type.lower() + ".txt"

            commands.append("""make run_make_predictions_{0} DB_NAME={1} TRAIN_PATH={2} TEST_PATH={3}  DISCRETIZATION_TYPE={4} GRAPH_TYPE={5} CONFIG_PATH={6} SAVE_DIR={7}  CLASSIC_TRAIN_PATH={8} CLASSIC_TEST_PATH={9} ALPHA={10}  CLASSIC_CONFIG_PATH={11}  """.
                            format(*[db_name.lower(), db_name.lower(), train_path, test_path, None, graph_type, config_path, save_dir,
                                     classic_train_path, classic_test_path, None, classic_config_path]))
        
    for graph_type in proposed_complete_graph:
        train_directory = 'data/with_normalization/graph_features/' + db_name.lower() + "/" + graph_type .lower() + '/train'
        test_directory = 'data/with_normalization/graph_features/' + db_name.lower() +"/" + graph_type .lower() + '/test'
        
        config_path = "data/with_normalization/without_stepwise/configurations/" + db_name.lower() + "/configuration_" + graph_type.lower() + ".txt"

        for alpha in alphas: 
            train_path = train_directory + '/new_features_' + str(alpha) + '.feather'
            test_path = test_directory + '/new_features_' + str(alpha) + '.feather'
            commands.append("""make run_make_predictions_{0} DB_NAME={1} TRAIN_PATH={2} TEST_PATH={3}  DISCRETIZATION_TYPE={4} GRAPH_TYPE={5} CONFIG_PATH={6} SAVE_DIR={7}  CLASSIC_TRAIN_PATH={8} CLASSIC_TEST_PATH={9} ALPHA={10} CLASSIC_CONFIG_PATH={11} """.
                            format(*[db_name.lower(), db_name.lower(), train_path, test_path, None, graph_type, config_path, save_dir, classic_train_path, classic_test_path,
                                     alpha, classic_config_path]))

                
    for graph_type in standard_proposed_graphs:
        for disc_type in discretization_types:
            train_directory = 'data/with_normalization/graph_features/' + db_name.lower() + '/' + graph_type .lower() + "/" +  disc_type.lower() + '/train'
            test_directory = 'data/with_normalization/graph_features/' + db_name.lower() + '/' + graph_type .lower() + "/" + disc_type.lower() + '/test' 
            
            config_path = "data/with_normalization/without_stepwise/configurations/" + db_name.lower() + "/configuration_" + graph_type.lower() + "_" + disc_type.lower() + ".txt"

            for alpha in alphas: 
                train_path = train_directory + '/new_features_' + str(alpha) + '.feather'
                test_path = test_directory + '/new_features_' + str(alpha) + '.feather'
                commands.append("""make run_make_predictions_{0} DB_NAME={1} TRAIN_PATH={2} TEST_PATH={3}  DISCRETIZATION_TYPE={4} GRAPH_TYPE={5} CONFIG_PATH={6} SAVE_DIR={7}  CLASSIC_TRAIN_PATH={8} CLASSIC_TEST_PATH={9} ALPHA={10} CLASSIC_CONFIG_PATH={11} """.
                                format(*[db_name.lower(), db_name.lower(), train_path, test_path, disc_type, graph_type, config_path, save_dir, classic_train_path, classic_test_path,
                                         alpha, classic_config_path]))
                

    processes = []

    for cmd in commands:
        process = subprocess.Popen(cmd, shell=True)
        processes.append(process)

    for p in  processes:
        p.wait()

def launch_predict_classic(db_name):
    train_path = "data/preprocessed/"+ db_name +"/preprocessed_data_train.feather"
    test_path = "data/preprocessed/"+ db_name +"/preprocessed_data_test.feather"
    config_path = "data/with_normalization/without_stepwise/configurations/" + db_name.lower() + "/configuration_classic.txt" 
    save_dir = 'reports/with_normalization/without_stepwise/' + db_name.lower() + '/metrics/'
    
    command = """make run_make_predictions_{0} DB_NAME={1} TRAIN_PATH={2} TEST_PATH={3} DISCRETIZATION_TYPE={4} GRAPH_TYPE={5} CONFIG_PATH={6} SAVE_DIR={7}  """.format(
        *[db_name.lower(), db_name.lower(), train_path, test_path, None, None, config_path,  save_dir])
    
    subprocess.run(command, shell=True)

def launch_predict_with_stepwise(db_name):
    save_dir = 'reports/with_normalization/with_stepwise/'+ db_name + '/metrics'
    classic_train_path = "data/preprocessed/"+ db_name +"/preprocessed_data_train.feather"
    classic_test_path = "data/preprocessed/"+ db_name +"/preprocessed_data_test.feather"
    classic_config_path = "data/with_normalization/with_stepwise/configurations/"+ db_name+"/configuration_classic.txt" 
    
    commands = []
    
    for graph_type in state_of_art_graphs:
            train_path = 'data/with_normalization/graph_features/' + db_name.lower() + '/' + graph_type.lower() + '/new_features_train.feather'
            test_path = 'data/with_normalization/graph_features/' + db_name.lower() + '/' + graph_type.lower() + '/new_features_test.feather'
            config_path = "data/with_normalization/with_stepwise/configurations/" + db_name.lower()+"/configuration_" + graph_type.lower() + ".txt"

            commands.append("""make run_make_predictions_{0} DB_NAME={1} TRAIN_PATH={2} TEST_PATH={3}  DISCRETIZATION_TYPE={4} GRAPH_TYPE={5} CONFIG_PATH={6} SAVE_DIR={7} CLASSIC_TRAIN_PATH={8} CLASSIC_TEST_PATH={9} CLASSIC_CONFIG_PATH={10} ALPHA={11} """.
                            format(*[db_name.lower(), db_name.lower(), train_path, test_path, None, graph_type, config_path, save_dir, 
                                     classic_train_path, classic_test_path, classic_config_path, None]))
        
    for graph_type in proposed_complete_graph:
        train_directory = 'data/with_normalization/graph_features/' + db_name.lower() + "/" + graph_type .lower() + '/train'
        test_directory = 'data/with_normalization/graph_features/' + db_name.lower() +"/" + graph_type .lower() + '/test'
        
        config_path = "data/with_normalization/with_stepwise/configurations/" + db_name.lower() + "/configuration_" + graph_type.lower() + ".txt"

        for alpha in alphas: 
            train_path = train_directory + '/new_features_' + str(alpha) + '.feather'
            test_path = test_directory + '/new_features_' + str(alpha) + '.feather'
            commands.append("""make run_make_predictions_{0} DB_NAME={1} TRAIN_PATH={2} TEST_PATH={3}  DISCRETIZATION_TYPE={4} GRAPH_TYPE={5} CONFIG_PATH={6} SAVE_DIR={7} CLASSIC_TRAIN_PATH={8} CLASSIC_TEST_PATH={9} ALPHA={10} CLASSIC_CONFIG_PATH={11} """.
                            format(*[db_name.lower(), db_name.lower(), train_path, test_path, None, graph_type, config_path, save_dir, 
                                     classic_train_path, classic_test_path, alpha, classic_config_path]))

                
    for graph_type in standard_proposed_graphs:
        for disc_type in discretization_types:
            train_directory = 'data/with_normalization/graph_features/' + db_name.lower() + '/' + graph_type .lower() + "/" +  disc_type.lower() + '/train'
            test_directory = 'data/with_normalization/graph_features/' + db_name.lower() + '/' + graph_type .lower() + "/" + disc_type.lower() + '/test' 
            
            config_path = "data/with_normalization/with_stepwise/configurations/" + db_name.lower() + "/configuration_" + graph_type.lower() + "_" + disc_type.lower() + ".txt"

            for alpha in alphas: 
                train_path = train_directory + '/new_features_' + str(alpha) + '.feather'
                test_path = test_directory + '/new_features_' + str(alpha) + '.feather'
                commands.append("""make run_make_predictions_{0} DB_NAME={1} TRAIN_PATH={2} TEST_PATH={3}  DISCRETIZATION_TYPE={4} GRAPH_TYPE={5} CONFIG_PATH={6} SAVE_DIR={7} CLASSIC_TRAIN_PATH={8} CLASSIC_TEST_PATH={9} ALPHA={10} CLASSIC_CONFIG_PATH={11}""".
                                format(*[db_name.lower(), db_name.lower(), train_path, test_path, disc_type, graph_type, config_path, save_dir, 
                                         classic_train_path, classic_test_path, alpha, classic_config_path]))

    processes = []

    for cmd in commands:
        process = subprocess.Popen(cmd, shell=True)
        processes.append(process)

    for p in  processes:
        p.wait()

def launch_predict_classic_with_stepwise(db_name):
    train_path = "data/preprocessed/"+ db_name +"/preprocessed_data_train.feather"
    test_path = "data/preprocessed/"+ db_name +"/preprocessed_data_test.feather"
    config_path = "data/with_normalization/with_stepwise/configurations/" + db_name + "/configuration_classic.txt" 
    save_dir = 'reports/with_normalization/with_stepwise/'+ db_name + '/metrics/'
    
    command = """make run_make_predictions_{0} DB_NAME={1} TRAIN_PATH={2} TEST_PATH={3} DISCRETIZATION_TYPE={4} GRAPH_TYPE={5} CONFIG_PATH={6} SAVE_DIR={7} """.format(
        *[db_name.lower(), db_name.lower(), train_path, test_path, None, None, config_path,  save_dir])
    
    subprocess.run(command, shell=True)
 
def launch_print(db_name):
    
    directory = "reports/with_normalization/without_stepwise/" + db_name + "/metrics/"
    
    commands = ["""make run_print_{0} DB_NAME={1} _DIR={2} DISCRETIZATION_TYPE={3} GRAPH_TYPE={4} """.
                format(*[db_name.lower(), db_name.lower(), directory, None, None])]
 
    for graph in state_of_art_graphs:
            commands.append("""make run_print_{0} DB_NAME={1} _DIR={2}  DISCRETIZATION_TYPE={3} GRAPH_TYPE={4} """.
                            format(*[db_name.lower(), db_name.lower(), directory, None, graph.lower()]))
            
    for graph in proposed_complete_graph:
            commands.append("""make run_print_{0} DB_NAME={1} _DIR={2} DISCRETIZATION_TYPE={3} GRAPH_TYPE={4} """.
                            format(*[db_name.lower(), db_name.lower(), directory, None, graph.lower()]))  
              
    for graph in standard_proposed_graphs:
            for discretization in discretization_types:
                commands.append(
                    """make run_print_{0} DB_NAME={1} _DIR={2}  DISCRETIZATION_TYPE={3} GRAPH_TYPE={4} """.
                    format(*[db_name.lower(), db_name.lower(), directory, discretization, graph.lower()]))


    processes = []
    
    for cmd in commands:
        process = subprocess.Popen(cmd, shell=True)
        processes.append(process)
        
    for p in processes: 
        p.wait()
 
def launch_print_with_stepwise(db_name):
    
    directory = "reports/with_normalization/with_stepwise/" + db_name + "/metrics/"
    
    commands = ["""make run_print_{0} DB_NAME={1} _DIR={2} DISCRETIZATION_TYPE={3} GRAPH_TYPE={4} """.
                format(*[db_name.lower(), db_name.lower(), directory, None, None])]

    for graph in state_of_art_graphs:
            commands.append("""make run_print_{0} DB_NAME={1} _DIR={2}  DISCRETIZATION_TYPE={3} GRAPH_TYPE={4} """.
                            format(*[db_name.lower(), db_name.lower(), directory, None, graph.lower()]))
            
    for graph in proposed_complete_graph:
            commands.append("""make run_print_{0} DB_NAME={1} _DIR={2} DISCRETIZATION_TYPE={3} GRAPH_TYPE={4} """.
                            format(*[db_name.lower(), db_name.lower(), directory, None, graph.lower()]))  
            
    for graph in standard_proposed_graphs:
            for discretization in discretization_types:
                commands.append(
                    """make run_print_{0} DB_NAME={1} _DIR={2}  DISCRETIZATION_TYPE={3} GRAPH_TYPE={4} """.
                    format(*[db_name.lower(), db_name.lower(), directory, discretization, graph.lower()]))


    processes = []
    
    for cmd in commands:
        process = subprocess.Popen(cmd, shell=True)
        processes.append(process)
        
    for p in processes: 
        p.wait()
         
def launch_plot(db_name):
    for model in models:
        for discretization_type in discretization_types:
            train_descriptors_paths = []
            test_descriptors_paths = []
            for graph in ["BIP", "MOD"]:
                best_alphas_path = 'outputs/general_results/results/'+ db_name +'/percent/predictions/' + graph.lower()  + '/best_alpha_values_' + graph.lower() + '_' + discretization_type.lower() + '.txt'
                with open(best_alphas_path, "r") as file:
                    alphas = ast.literal_eval(file.read())
                    
                alpha = alphas[model]
                train_descriptors_paths.append('outputs/' + db_name + '/new_descriptors/' + graph.lower() + '/train/new_descriptors_data' + '_' + discretization_type.lower() + '_' +  graph.lower() + '_' + str(alpha) + '.feather')
                test_descriptors_paths.append('outputs/' + db_name + '/new_descriptors/' + graph.lower() + '/test/new_descriptors_data' + '_' + discretization_type.lower()  + '_' + graph.lower() + '_' + str(alpha) + '.feather')
            
            subprocess.run("""make run_plot_{0} DB_NAME={1} TRAIN_DESCRIPTORS_PATHS={2} TEST_DESCRIPTORS_PATHS={3}  MODEL={4} DISCRETIZATION_TYPE={5} """.format(
                *[db_name.lower(), db_name.lower(), '\"' + str(train_descriptors_paths) + '\"',
                  f'\"{test_descriptors_paths}\"', model, discretization_type]), shell=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    db_name = args[0]
    
    # launch_preprocess(db_name)
    # launch_build_engine_for_unsupervized_discretization(db_name)
    # launch_build_engine_for_supervized_discretization(db_name)
    # launch_unsupervised_discretization(db_name)
    # launch_supervised_discretization(db_name)
    # launch_graph_modeling(db_name)
    # launch_compute_descriptors(db_name)
    # launch_config_without_stepwise(db_name)
    # launch_predict_classic(db_name)
    # launch_predict(db_name)
    launch_print(db_name)
    
    # launch_stepwise_selection(db_name)
    # launch_config_with_stepwise(db_name)
    # launch_predict_classic_with_stepwise(db_name)
    # launch_predict_with_stepwise(db_name)
    # launch_print_with_stepwise(db_name)
    
    # launch_plot(db_name)
    
    
    
    
    
    
# def launch_create_edges_for_complete_graph(db_name):
    
#     _dir = 'graph/with_normalization/'
    
#     base_path = f"{_dir}{db_name}'/subsets"

#     for folder in ["train", "test"]:
#         shutil.rmtree(f"{base_path}/{folder}/", ignore_errors=True)    
    
#     train_path = "data/preprocessed/"+ db_name +"/preprocessed_data_train.feather"
#     test_path = "data/preprocessed/"+ db_name +"/preprocessed_data_test.feather"
    
#     # "data/preprocessed/"+ db_name +"/preprocessed_data_train.feather",
    
#     trainset = pd.read_feather(train_path, keep_default_na=False, na_values=[""])
    
    
#     cpu_count = os.cpu_count()
#     size = trainset.shape[0]
#     step = int(size / cpu_count)
#     end = step
#     start = 0
    
#     commands = []    
    
#     while start < size - 1:
#         if step + start >= size - 1 :
#             end = size
         
#         commands.append(""" make run_create_edges_of_train_{0} DB_NAME={1} START={2} END={3} _PATH={4}  _DIR={5} """.
#                             format(*[db_name.lower(), db_name.lower(), start, end, train_path, _dir]))
       
#         end += step
#         start += step
      
   
#     processes = []
    
#     for cmd in commands:
#         process = subprocess.Popen(cmd, shell=True)
#         processes.append(process)
        
#     for p in processes:
#         p.wait() 
         
  
#     testset = pd.read_feather(test_path, keep_default_na=False, na_values=[""])  
   
#     commands = []    
    
#     size = testset.shape[0]
#     step = int(size / cpu_count)
#     end = step
#     start = 0
    
#     if(cpu_count > step):
#         print("start =>", 0, ", end =>", size)
#         commands.append(""" make run_create_edges_of_test_{0} DB_NAME={1}  START={2} END={3} _PATH={4} _DIR={5} """.
#                             format(*[db_name.lower(), db_name.lower(), 0, size, test_path, _dir]))
#     else:
        
#         while start < size - 1:
            
#             if step + start >= size - 1 :
#                 end = size
                
#             print("start =>", start, ", end =>", end)    
#             commands.append(""" make run_create_edges_of_test_{0} DB_NAME={1}  START={2} END={3} _PATH={4} _DIR={5} """.
#                             format(*[db_name.lower(), db_name.lower(), start, end, test_path, _dir]))
            
        
#             end += step
#             start += step

#     processes = []
    
#     for cmd in commands:
#         process = subprocess.Popen(cmd, shell=True)
#         processes.append(process)
        
#     for p in processes:
#         p.wait()   
  
# def launch_relate_edges_for_complete_graph(db_name):
    
#     # CONNECTIONS BETWEEN TRAINING EDGES
#     directory = 'graph/with_normalization/'+ db_name + '/subsets/train/' 
#     train_path = "data/preprocessed/"+ db_name +"/preprocessed_data_train.feather"
#     _dir = 'graph/with_normalization/'
    
   
#     base_path = f"{_dir}{db_name}/related"

#     for folder in ["train", "test", "both"]:
#         shutil.rmtree(f"{base_path}/{folder}/", ignore_errors=True)
      
#     commands = []  
#     subset_train_data = {}
#     for item in os.listdir(directory):
#         path = os.path.join(directory, item)
#         with open(path, 'r') as file:
#             data = ast.literal_eval(file.read())
#         subset_train_data[path] = (data['start'], data['end'])
  
#     count = 0
        
#     for (_, (start1, end1)), (_, (start2, end2)) in combinations(subset_train_data.items(), 2):
#         commands.append(""" make run_relate_edges_of_train_{0} DB_NAME={1} START1={2} END1={3} START2={4} END2={5} _PATH={6} _DIR={7}""".
#         format(db_name.lower(), db_name.lower(), start1, end1, start2, end2, train_path, _dir)) 
        
        
#         count += 1
        
#         if (count % 8 == 0):
            
#             processes = []    
                
#             for cmd in commands:
#                 process = subprocess.Popen(cmd, shell=True)
#                 processes.append(process)
                
#             for p in processes:
#                 p.wait()     
#                 print(f"Command completed.")  
            
#             commands = []
        
   
    
#     processes = []    
                
#     for cmd in commands:
#         process = subprocess.Popen(cmd, shell=True)
#         processes.append(process)
        
#     for p in processes:
#         p.wait()     
#         print(f"modeling train-train command completed.")          
       
        
#     # CONNECTIONS BETWEEN TEST EDGES    
#     commands = []  
#     subset_test_data = {}
    
#     directory = 'graph/with_normalization/'+ db_name + '/subsets/test/'
#     test_path = "data/preprocessed/"+ db_name +"/preprocessed_data_test.feather"
    
#     for item in os.listdir(directory):
#         path = os.path.join(directory, item)
#         with open(path, 'r') as file:
#             data = ast.literal_eval(file.read())
#         subset_test_data[path] = (data['start'], data['end'])

#     if (len(subset_test_data) == 1):
#         path, (start, end) = list(subset_test_data.items())[0]
#         commands.append(""" make run_relate_edges_of_test_{0} DB_NAME={1} START1={2} END1={3} START2={4} END2={5} _PATH={6} _DIR={7} """.
#          format(db_name.lower(), db_name.lower(), start, end, start, end, test_path, _dir))
        
#     else:     
#         count = 0
        
#         for (_, (start1, end1)), (_, (start2, end2)) in combinations(subset_test_data.items(), 2):
#              commands.append(""" make run_relate_edges_of_test_{0} DB_NAME={1} START1={2} END1={3} START2={4} END2={5} _PATH={6} _DIR={7} """.
#              format(db_name.lower(), db_name.lower(), start1, end1, start2, end2, test_path, _dir))   
             
#              count += 1  
             
#              if (count % 8 == 0):
#                 processes = []    
                    
#                 for cmd in commands:
#                     process = subprocess.Popen(cmd, shell=True)
#                     processes.append(process)
                    
#                 for p in processes:
#                     p.wait()     
#                     print(f"Command completed.")  
                
#                 commands = []
   
#     processes = []    
                
#     for cmd in commands:
#         process = subprocess.Popen(cmd, shell=True)
#         processes.append(process)
        
#     for p in processes:
#         p.wait()     
#         print(f"modeling test-test command  completed.")         
    
   
#    # CONNECTIONS BETWEEN TRAINING EDGES AND TEST EDGES 
#     count = 0
#     commands = []
#     for ((_,(start1, end1)), (_,(start2, end2))) in product(subset_train_data.items(), subset_test_data.items()):
        
#          commands.append(""" make run_relate_edges_of_both_{0} DB_NAME={1} START1={2} END1={3} START2={4} END2={5} TRAIN_PATH={6} TEST_PATH={7} _DIR={8} """.
#          format(db_name.lower(), db_name.lower(), start1, end1, start2, end2, train_path, test_path, _dir))
         
#          count += 1  
             
#          if (count % 8 == 0):
#                 processes = []     
                    
#                 for cmd in commands:
#                     process = subprocess.Popen(cmd, shell=True)
#                     processes.append(process)
                    
#                 for p in processes:
#                     p.wait()     
#                     print(f"Command completed.")  
                
#                 commands = []
   
#     processes = [] 
           
#     for cmd in commands:
#         process = subprocess.Popen(cmd, shell=True)
#         processes.append(process)
        
#     for p in processes:
#         p.wait()     
#         print(f"modeling train-test command  completed.")   
 
# def launch_split(db_name):
#     commands = []
#     commands.append(""" make run_splitting_{0} DB_NAME={1}""".format(*[db_name.lower(), db_name.lower()]))

#     processes = []   
#     for cmd in commands:
#         process = subprocess.Popen(cmd, shell=True)
#         processes.append(process)
        
#     for p in processes:
#         p.wait()

# def launch_build_engine_for_preprocessing(db_name):
#     commands = []
#     commands.append(""" make run_engine_building_pre_{0} DB_NAME={1}""".format(*[db_name.lower(), db_name.lower()]))

#     processes = []   
#     for cmd in commands:
#         process = subprocess.Popen(cmd, shell=True)
#         processes.append(process)
        
#     for p in processes:
#         p.wait()
        
# def launch_preprocess_train(db_name):
#     commands = []
#     commands.append(""" make run_preprocess_train_{0} DB_NAME={1}""".format(*[db_name.lower(), db_name.lower()]))

#     processes = []   
#     for cmd in commands:
#         process = subprocess.Popen(cmd, shell=True)
#         processes.append(process)
        
#     for p in processes:
#         p.wait()
             
# def launch_preprocess_test(db_name):
#     commands = []
#     commands.append(""" make run_preprocess_test_{0} DB_NAME={1}""".format(*[db_name.lower(), db_name.lower()]))

#     processes = []   
#     for cmd in commands:
#         process = subprocess.Popen(cmd, shell=True)
#         processes.append(process)
        
#     for p in processes:
#         p.wait()        

# def launch_sampling(db_name):
#     commands = [""" make run_sampling_{0} DB_NAME={1}""".format(*[db_name.lower(), db_name.lower()])]
#     processes = []
#     for cmd in commands:
#         process = subprocess.Popen(cmd, shell=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
#         processes.append(process)

#     for p in processes:
#         p.wait()
       
    
# commands.append("""make run_graph_modeling_complete_{0} DB_NAME={1}  _DIR={2} """.
#                 format(*[db_name.lower(), db_name.lower(),  _dir]))
 
# launch_create_edges_for_complete_graph(db_name)
# launch_relate_edges_for_complete_graph(db_name)
# launch_split(db_name)
# launch_build_engine_for_preprocessing(db_name)
# launch_preprocess_train(db_name)
# launch_preprocess_test(db_name)