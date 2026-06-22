import subprocess, threading
import sys, ast, os
from dotenv import load_dotenv

discretization_types =  ["SUP"] 
alphas = [0.1, 0.15, 0.5, 0.85]
state_of_art_graphs = ["LIU_V2", "GUI", "LIU_V1"] 
standard_proposed_graphs = ["BIP", "MOD"]   
proposed_complete_graph = ["LOAN"]

MAX_WORKERS = 2
semaphore = threading.Semaphore(MAX_WORKERS)


load_dotenv()

def run_command(cmd, db_name):
        with semaphore:        # for cmd in commands:
            process = subprocess.Popen(cmd, shell=True)
            process.wait()
            print(f"##################### {db_name} processing completed. ############################")
    

def launch_preprocess(db_name):
    command = """ make run_preprocess_{0} DB_NAME={1} """.format(*[db_name.lower(), db_name.lower()])

    subprocess.run(command, shell=True)

def launch_build_engine_for_supervized_discretization(db_name, sub):
    
    _path = 'data/preprocessed/'+ db_name + '/partial_preprocessed_data_train_' + sub + '.feather'
    _dir = "engine/with_normalization/discretization/"
    
    command = """ make run_engine_building_supervised_discretization_{0} DB_NAME={1} _PATH={2} _DIR={3} SUB={4} """.format(
        *[db_name.lower(), db_name.lower(), _path, _dir, sub ] )

    subprocess.run(command, shell=True)

def launch_supervised_discretization(db_name, sub):
    commands = []
    for label in ["train", "test"]:
        _path = 'data/preprocessed/'+ db_name +  '/partial_preprocessed_data_' + label + '_' + sub + '.feather'
        
        commands.append("""make run_supervised_discretization_{0} DB_NAME={1} _PATH={2} NORMALIZATION_LABEL={3}  DATA_LABEL={4} SUB={5}""".
                        format(*[db_name.lower(), db_name.lower(), _path, "with_normalization", label, sub]))
   
    processes = []
    
    for cmd in commands:
        process = subprocess.Popen(cmd, shell=True)
        processes.append(process)
        
    for p in processes:
        p.wait()     
        print(f"Supervized discretization launch  completed.")

def launch_graph_modeling(db_name, sub):
    commands = []
    _dir = 'graph/with_normalization/' 
    
    for graph_type in standard_proposed_graphs:
        for discretization_type in discretization_types:
            train_path = "data/with_normalization/discretized/" + db_name.lower() + "/discretized_train_data_" + discretization_type.lower() + '_' + sub + ".feather"
            test_path = "data/with_normalization/discretized/" + db_name.lower() + "/discretized_test_data_" + discretization_type.lower() + '_' + sub + ".feather"
           
            
            commands.append("""make run_graph_modeling_""" + graph_type.lower() + """_{0} DB_NAME={1} GRAPH_TYPE={2} DISCRETIZATION_TYPE={3}  TRAIN_PATH={4} TEST_PATH={5} _DIR={6} SUB={7} """.
                        format(*[db_name.lower(), db_name.lower(), graph_type.lower(), discretization_type.lower(), train_path, test_path, _dir, sub]))
    
    for graph_type in proposed_complete_graph:
         train_path = "data/preprocessed/"+ db_name +"/partial_preprocessed_data_train_" + sub + ".feather"
         test_path = "data/preprocessed/"+ db_name +"/partial_preprocessed_data_test_" + sub + ".feather"
         
         commands.append("""make run_graph_modeling_""" + graph_type.lower() + """_{0} DB_NAME={1} GRAPH_TYPE={2} TRAIN_PATH={3} TEST_PATH={4} _DIR={5} SUB={6} """.
                         format(*[db_name.lower(), db_name.lower(), graph_type.lower(), train_path, test_path, _dir, sub]))
 

    for graph_type in ["LIU", "GUI"]: 
            commands.append("""make run_graph_modeling_""" + graph_type.lower() + """_{0} DB_NAME={1} GRAPH_TYPE={2} DISCRETIZATION_TYPE={3}  TRAIN_PATH={4} TEST_PATH={5} _DIR={6} SUB={7}""".
                        format(*[db_name.lower(), db_name.lower(), graph_type.lower(), None, None, None,  _dir, sub]))
       
           
    threads = []
    for cmd in commands:
        t = threading.Thread(target=run_command,  args=(cmd, db_name))
        t.start()
        threads.append(t)
        
    for t in threads:
        t.join()
        
def launch_compute_descriptors(db_name, sub):
    
    _dir = 'data/with_normalization/graph_features/'
    _graph_dir = 'graph/with_normalization/'
    
    commands = []
    
    for graph_type in standard_proposed_graphs:
        for discretization_type in discretization_types:
            train_path = "data/with_normalization/discretized/" + db_name.lower() + "/discretized_train_data_" + discretization_type.lower() + "_" + sub + ".feather"
            test_path = "data/with_normalization/discretized/" + db_name.lower() + "/discretized_test_data_" + discretization_type.lower() + "_" + sub + ".feather"
           
            for alpha in alphas:
                commands.append(""" make run_compute_descriptors_""" + graph_type.lower() + """_{0}  BD_NAME={1} GRAPH_TYPE={2} ALPHA={3} DISCRETIZATION_TYPE={4} TRAIN_PATH={5} TEST_PATH={6} _DIR={7} GRAPH_DIR={8} SUB={9}""".
                                format(*[db_name.lower(), db_name.lower(), graph_type.lower(), alpha, discretization_type.lower(), train_path, test_path, _dir, _graph_dir, sub]))
     
     
    train_path = "data/preprocessed/"+ db_name +"/preprocessed_data_train_" + sub + ".feather"
    test_path = "data/preprocessed/"+ db_name +"/preprocessed_data_test_" + sub + ".feather"
    for graph_type in proposed_complete_graph:
         for alpha in alphas:
            commands.append(""" make run_compute_descriptors_loan_{0}  BD_NAME={1} GRAPH_TYPE={2} ALPHA={3}  TRAIN_PATH={4} TEST_PATH={5}  GRAPH_DIR={6} _DIR={7} SUB={8}""".
                        format(*[db_name.lower(), db_name.lower(), graph_type.lower(), alpha, train_path, test_path, _graph_dir, _dir, sub]))
        
                     
    for graph_type in state_of_art_graphs:  
        commands.append(""" make run_compute_descriptors_""" + graph_type.lower() + """_{0}  BD_NAME={1} GRAPH_TYPE={2} TRAIN_PATH={3} TEST_PATH={4} _DIR={5} GRAPH_DIR={6} SUB={7} """.
                        format(*[db_name.lower(), db_name.lower(), graph_type.lower(), train_path, test_path, _dir, _graph_dir, sub]))
        
    threads = []
    for cmd in commands:
        t = threading.Thread(target=run_command,  args=(cmd, db_name))
        t.start()
        threads.append(t)
        
    for t in threads:
        t.join()
        
def launch_config_without_stepwise(db_name, sub):
    
    classic_train_path = "data/preprocessed/"+ db_name +"/preprocessed_data_train_" + sub + ".feather"
    save_dir = 'data/with_normalization/without_stepwise/configurations'
      
    commands = []

    commands.append("""make run_make_configurations_{0}  SAVE_DIR={1} DISCRETIZATION_TYPE={2} GRAPH_TYPE={3} CLASSIC_TRAIN_PATH={4} NEW_DESCRIPTOR_TRAIN_PATH={5} """.
                    format(*[db_name.lower(), save_dir, None, None, classic_train_path, classic_train_path]))
  
    for graph_type in state_of_art_graphs:
        
        new_descriptor_train_path = "data/with_normalization/graph_features/" + db_name.lower() + "/" + sub + "/" + graph_type.lower() + "/new_features_train.feather"
        
        commands.append("""make run_make_configurations_{0}  SAVE_DIR={1} DISCRETIZATION_TYPE={2} GRAPH_TYPE={3} CLASSIC_TRAIN_PATH={4} NEW_DESCRIPTOR_TRAIN_PATH={5}""".
                        format(*[db_name.lower(), save_dir, None, graph_type, classic_train_path, new_descriptor_train_path]))
         

    for graph_type in standard_proposed_graphs:    
        for discretization_type in discretization_types:
            new_descriptor_train_path = "data/with_normalization/graph_features/" + db_name.lower() + "/" + sub + "/" + graph_type.lower()  +"/"+ discretization_type.lower()  + '/train/' + "new_features_0.1.feather"
            
            commands.append("""make run_make_configurations_{0}  SAVE_DIR={1} DISCRETIZATION_TYPE={2} GRAPH_TYPE={3} CLASSIC_TRAIN_PATH={4} NEW_DESCRIPTOR_TRAIN_PATH={5}""".
                            format(*[db_name.lower(), save_dir, discretization_type, graph_type, classic_train_path, new_descriptor_train_path]))
            
    for graph_type in proposed_complete_graph:
        new_descriptor_train_path = "data/with_normalization/graph_features/" + db_name.lower() + "/" + sub + "/" + graph_type.lower() + "/train/new_features_0.1.feather"
        
        commands.append("""make run_make_configurations_{0}  SAVE_DIR={1} DISCRETIZATION_TYPE={2} GRAPH_TYPE={3} CLASSIC_TRAIN_PATH={4} NEW_DESCRIPTOR_TRAIN_PATH={5}""".
                        format(*[db_name.lower(), save_dir, None, graph_type, classic_train_path, new_descriptor_train_path]))
                
            
    threads = []
    for cmd in commands:
        t = threading.Thread(target=run_command,  args=(cmd, db_name))
        t.start()
        threads.append(t)
        
    for t in threads:
        t.join()
        
def launch_config_with_stepwise(db_name, sub):
    
    classic_train_path = "data/preprocessed/"+ db_name + "/" + sub + "/preprocessed_data_train.feather"
    save_dir = 'data/with_normalization/with_stepwise/configurations'
      
    commands = []

    commands.append("""make run_make_configurations_with_stepwise_{0}  SAVE_DIR={1} DISCRETIZATION_TYPE={2} GRAPH_TYPE={3} CLASSIC_TRAIN_PATH={4} NEW_DESCRIPTOR_TRAIN_PATH={5} """.
                    format(*[db_name.lower(), save_dir, None, None, classic_train_path, classic_train_path]))
  
    for graph_type in state_of_art_graphs:
        
        new_descriptor_train_path = "data/with_normalization/graph_features/" + db_name.lower() + "/" + sub + "/" + graph_type.lower() + "/new_features_train.feather"
        
        commands.append("""make run_make_configurations_with_stepwise_{0}  SAVE_DIR={1} DISCRETIZATION_TYPE={2} GRAPH_TYPE={3} CLASSIC_TRAIN_PATH={4} NEW_DESCRIPTOR_TRAIN_PATH={5}""".
                        format(*[db_name.lower(), save_dir, None, graph_type, classic_train_path, new_descriptor_train_path]))
         

    for graph_type in standard_proposed_graphs:    
        for discretization_type in discretization_types:
            new_descriptor_train_path = "data/with_normalization/graph_features/" + db_name.lower() + "/" + sub + "/" + graph_type.lower()  +"/"+ discretization_type.lower()  + '/train/' + "new_features_0.1.feather"
            
            commands.append("""make run_make_configurations_with_stepwise_{0}  SAVE_DIR={1} DISCRETIZATION_TYPE={2} GRAPH_TYPE={3} CLASSIC_TRAIN_PATH={4} NEW_DESCRIPTOR_TRAIN_PATH={5}""".
                            format(*[db_name.lower(), save_dir, discretization_type, graph_type, classic_train_path, new_descriptor_train_path]))
            
    for graph_type in proposed_complete_graph:
        new_descriptor_train_path = "data/with_normalization/graph_features/" + db_name.lower() + "/" + sub + "/" + graph_type.lower() + "/train/new_features_0.1.feather"
        
        commands.append("""make run_make_configurations_with_stepwise_{0}  SAVE_DIR={1} DISCRETIZATION_TYPE={2} GRAPH_TYPE={3} CLASSIC_TRAIN_PATH={4} NEW_DESCRIPTOR_TRAIN_PATH={5}""".
                        format(*[db_name.lower(), save_dir, None, graph_type, classic_train_path, new_descriptor_train_path]))
                
    threads = []
    for cmd in commands:
        t = threading.Thread(target=run_command,  args=(cmd, db_name))
        t.start()
        threads.append(t)
        
    for t in threads:
        t.join()        

def launch_stepwise_selection(db_name, sub):
    train_path = "data/preprocessed/"+ db_name + "/" + sub + "/preprocessed_data_train.feather"
    save_dir = 'data/with_normalization/with_stepwise/configurations/'+ db_name.lower() 
    
    command = """make run_select_features_{0} DB_NAME={1} TRAIN_PATH={2} SAVE_DIR={3} SUB={4} """.format(
        *[db_name.lower(), db_name.lower(), train_path, save_dir, sub])
    
    subprocess.run(command, shell=True)
   
def launch_predict(db_name, sub):
    save_dir = 'reports/with_normalization/without_stepwise/'+ db_name + "/" + sub + "/metrics"
    classic_train_path = "data/preprocessed/"+ db_name + "/" + sub + "/preprocessed_data_train.feather"
    classic_test_path = "data/preprocessed/"+ db_name + "/" + sub + "/preprocessed_data_test.feather"
    classic_config_path = "data/with_normalization/without_stepwise/configurations/"+ db_name + "/" + sub + "/configuration_classic.txt" 
    
    commands = []
    
    for graph_type in state_of_art_graphs:
            train_path = 'data/with_normalization/graph_features/' + db_name.lower() +  "/" + sub + "/" + graph_type.lower() + '/new_features_train.feather'
            test_path = 'data/with_normalization/graph_features/' + db_name.lower() + "/" + sub + '/' + graph_type.lower() + '/new_features_test.feather'
            config_path = "data/with_normalization/without_stepwise/configurations/" + db_name.lower()+"/configuration_" + graph_type.lower() + ".txt"

            commands.append("""make run_make_predictions_{0} DB_NAME={1} TRAIN_PATH={2} TEST_PATH={3}  DISCRETIZATION_TYPE={4} GRAPH_TYPE={5} CONFIG_PATH={6} SAVE_DIR={7}  CLASSIC_TRAIN_PATH={8} CLASSIC_TEST_PATH={9} ALPHA={10}  CLASSIC_CONFIG_PATH={11} SUB={12} """.
                            format(*[db_name.lower(), db_name.lower(), train_path, test_path, None, graph_type, config_path, save_dir,
                                     classic_train_path, classic_test_path, None, classic_config_path, sub]))
        
    for graph_type in proposed_complete_graph:
        train_directory = 'data/with_normalization/graph_features/' + db_name.lower()  +  "/" + sub + "/" + graph_type .lower() + '/train'
        test_directory = 'data/with_normalization/graph_features/' + db_name.lower() +  "/" + sub + "/" + graph_type .lower() + '/test'
        
        config_path = "data/with_normalization/without_stepwise/configurations/" + db_name.lower() + "/configuration_" + graph_type.lower() + ".txt"

        for alpha in alphas: 
            train_path = train_directory + '/new_features_' + str(alpha) + '.feather'
            test_path = test_directory + '/new_features_' + str(alpha) + '.feather'
            commands.append("""make run_make_predictions_{0} DB_NAME={1} TRAIN_PATH={2} TEST_PATH={3}  DISCRETIZATION_TYPE={4} GRAPH_TYPE={5} CONFIG_PATH={6} SAVE_DIR={7}  CLASSIC_TRAIN_PATH={8} CLASSIC_TEST_PATH={9} ALPHA={10} CLASSIC_CONFIG_PATH={11} SUB={12} """.
                            format(*[db_name.lower(), db_name.lower(), train_path, test_path, None, graph_type, config_path, save_dir, classic_train_path, classic_test_path,
                                     alpha, classic_config_path, sub]))

                
    for graph_type in standard_proposed_graphs:
        for disc_type in discretization_types:
            train_directory = 'data/with_normalization/graph_features/' + db_name.lower()  +  "/" + sub + '/' + graph_type .lower() + "/" +  disc_type.lower() + '/train'
            test_directory = 'data/with_normalization/graph_features/' + db_name.lower()  +  "/" + sub + '/' + graph_type .lower() + "/" + disc_type.lower() + '/test' 
            
            config_path = "data/with_normalization/without_stepwise/configurations/" + db_name.lower() + "/" + sub + "/configuration_" + graph_type.lower() + "_" + disc_type.lower() + ".txt"

            for alpha in alphas: 
                train_path = train_directory + '/new_features_' + str(alpha) + '.feather'
                test_path = test_directory + '/new_features_' + str(alpha) + '.feather'
                commands.append("""make run_make_predictions_{0} DB_NAME={1} TRAIN_PATH={2} TEST_PATH={3}  DISCRETIZATION_TYPE={4} GRAPH_TYPE={5} CONFIG_PATH={6} SAVE_DIR={7}  CLASSIC_TRAIN_PATH={8} CLASSIC_TEST_PATH={9} ALPHA={10} CLASSIC_CONFIG_PATH={11} SUB={12} """.
                                format(*[db_name.lower(), db_name.lower(), train_path, test_path, disc_type, graph_type, config_path, save_dir, classic_train_path, classic_test_path,
                                         alpha, classic_config_path], sub))
                

    threads = []
    for cmd in commands:
        t = threading.Thread(target=run_command,  args=(cmd, db_name))
        t.start()
        threads.append(t)
        
    for t in threads:
        t.join()

def launch_predict_classic(db_name, sub):
    train_path = "data/preprocessed/"+ db_name   +  "/" + sub +"/preprocessed_data_train.feather"
    test_path = "data/preprocessed/"+ db_name   +  "/" + sub +"/preprocessed_data_test.feather"
    config_path = "data/with_normalization/without_stepwise/configurations/" + db_name.lower()  +  "/" + sub +  "/configuration_classic.txt" 
    save_dir = 'reports/with_normalization/without_stepwise/' + db_name.lower()  +  "/" + sub + '/metrics/'
    
    command = """make run_make_predictions_{0} DB_NAME={1} TRAIN_PATH={2} TEST_PATH={3} DISCRETIZATION_TYPE={4} GRAPH_TYPE={5} CONFIG_PATH={6} SAVE_DIR={7} SUB={8} """.format(
        *[db_name.lower(), db_name.lower(), train_path, test_path, None, None, config_path,  save_dir, sub])
    
    subprocess.run(command, shell=True)

def launch_predict_with_stepwise(db_name, sub):
    save_dir = 'reports/with_normalization/with_stepwise/'+ db_name  +  "/" + sub + '/metrics'
    classic_train_path = "data/preprocessed/"+ db_name  +  "/" + sub +"/preprocessed_data_train.feather"
    classic_test_path = "data/preprocessed/"+ db_name  +  "/" + sub + "/preprocessed_data_test.feather"
    classic_config_path = "data/with_normalization/with_stepwise/configurations/"+ db_name  +  "/" + sub +"/configuration_classic.txt" 
    
    commands = []
    
    for graph_type in state_of_art_graphs:
            train_path = 'data/with_normalization/graph_features/' + db_name.lower()  +  "/" + sub + '/' + graph_type.lower() + '/new_features_train.feather'
            test_path = 'data/with_normalization/graph_features/' + db_name.lower()  +  "/" + sub + '/' + graph_type.lower() + '/new_features_test.feather'
            config_path = "data/with_normalization/with_stepwise/configurations/"  + db_name.lower()  +  "/" + sub+ "/configuration_" + graph_type.lower() + ".txt"

            commands.append("""make run_make_predictions_{0} DB_NAME={1} TRAIN_PATH={2} TEST_PATH={3}  DISCRETIZATION_TYPE={4} GRAPH_TYPE={5} CONFIG_PATH={6} SAVE_DIR={7} CLASSIC_TRAIN_PATH={8} CLASSIC_TEST_PATH={9} CLASSIC_CONFIG_PATH={10} ALPHA={11} SUB={12} """.
                            format(*[db_name.lower(), db_name.lower(), train_path, test_path, None, graph_type, config_path, save_dir, 
                                     classic_train_path, classic_test_path, classic_config_path, None, sub]))
        
    for graph_type in proposed_complete_graph:
        train_directory = 'data/with_normalization/graph_features/' + db_name.lower() +  "/" + sub + "/" + graph_type .lower() + '/train'
        test_directory = 'data/with_normalization/graph_features/' + db_name.lower()  +  "/" + sub +"/" + graph_type .lower() + '/test'
        
        config_path = "data/with_normalization/with_stepwise/configurations/" + db_name.lower()  +  "/" + sub + "/configuration_" + graph_type.lower() + ".txt"

        for alpha in alphas: 
            train_path = train_directory + '/new_features_' + str(alpha) + '.feather'
            test_path = test_directory + '/new_features_' + str(alpha) + '.feather'
            commands.append("""make run_make_predictions_{0} DB_NAME={1} TRAIN_PATH={2} TEST_PATH={3}  DISCRETIZATION_TYPE={4} GRAPH_TYPE={5} CONFIG_PATH={6} SAVE_DIR={7} CLASSIC_TRAIN_PATH={8} CLASSIC_TEST_PATH={9} ALPHA={10} CLASSIC_CONFIG_PATH={11} SUB={12}""".
                            format(*[db_name.lower(), db_name.lower(), train_path, test_path, None, graph_type, config_path, save_dir, 
                                     classic_train_path, classic_test_path, alpha, classic_config_path, sub]))

                
    for graph_type in standard_proposed_graphs:
        for disc_type in discretization_types:
            train_directory = 'data/with_normalization/graph_features/' + db_name.lower()  +  "/" + sub + '/' + graph_type .lower() + "/" +  disc_type.lower() + '/train'
            test_directory = 'data/with_normalization/graph_features/' + db_name.lower()  +  "/" + sub + '/' + graph_type .lower() + "/" + disc_type.lower() + '/test' 
            
            config_path = "data/with_normalization/with_stepwise/configurations/" + db_name.lower()  +  "/" + sub + "/configuration_" + graph_type.lower() + "_" + disc_type.lower() + ".txt"

            for alpha in alphas: 
                train_path = train_directory + '/new_features_' + str(alpha) + '.feather'
                test_path = test_directory + '/new_features_' + str(alpha) + '.feather'
                commands.append("""make run_make_predictions_{0} DB_NAME={1} TRAIN_PATH={2} TEST_PATH={3}  DISCRETIZATION_TYPE={4} GRAPH_TYPE={5} CONFIG_PATH={6} SAVE_DIR={7} CLASSIC_TRAIN_PATH={8} CLASSIC_TEST_PATH={9} ALPHA={10} CLASSIC_CONFIG_PATH={11} SUB={12}""".
                                format(*[db_name.lower(), db_name.lower(), train_path, test_path, disc_type, graph_type, config_path, save_dir, 
                                         classic_train_path, classic_test_path, alpha, classic_config_path, sub]))

    threads = []
    for cmd in commands:
        t = threading.Thread(target=run_command,  args=(cmd, db_name))
        t.start()
        threads.append(t)
        
    for t in threads:
        t.join()

def launch_predict_classic_with_stepwise(db_name, sub):
    train_path = "data/preprocessed/"+ db_name  +  "/" + sub+"/preprocessed_data_train.feather"
    test_path = "data/preprocessed/"+ db_name  +  "/" + sub +"/preprocessed_data_test.feather"
    config_path = "data/with_normalization/with_stepwise/configurations/" + db_name  +  "/" + sub + "/configuration_classic.txt" 
    save_dir = 'reports/with_normalization/with_stepwise/'+ db_name  +  "/" + sub + '/metrics/'
    
    command = """make run_make_predictions_{0} DB_NAME={1} TRAIN_PATH={2} TEST_PATH={3} DISCRETIZATION_TYPE={4} GRAPH_TYPE={5} CONFIG_PATH={6} SAVE_DIR={7} SUB={8} """.format(
        *[db_name.lower(), db_name.lower(), train_path, test_path, None, None, config_path,  save_dir, sub])
    
    subprocess.run(command, shell=True)
 
def launch_print(db_name, sub):
    
    directory = "reports/with_normalization/without_stepwise/" + db_name  +  "/" + sub + "/metrics/"
    
    commands = ["""make run_print_{0} DB_NAME={1} _DIR={2} DISCRETIZATION_TYPE={3} GRAPH_TYPE={4} SUB={5} """.
                format(*[db_name.lower(), db_name.lower(), directory, None, None, sub])]
 
    for graph in state_of_art_graphs:
            commands.append("""make run_print_{0} DB_NAME={1} _DIR={2}  DISCRETIZATION_TYPE={3} GRAPH_TYPE={4} SUB={5}""".
                            format(*[db_name.lower(), db_name.lower(), directory, None, graph.lower(), sub]))
            
    for graph in proposed_complete_graph:
            commands.append("""make run_print_{0} DB_NAME={1} _DIR={2} DISCRETIZATION_TYPE={3} GRAPH_TYPE={4} SUB={5}""".
                            format(*[db_name.lower(), db_name.lower(), directory, None, graph.lower(), sub]))  
              
    for graph in standard_proposed_graphs:
            for discretization in discretization_types:
                commands.append(
                    """make run_print_{0} DB_NAME={1} _DIR={2}  DISCRETIZATION_TYPE={3} GRAPH_TYPE={4} SUB={5} """.
                    format(*[db_name.lower(), db_name.lower(), directory, discretization, graph.lower(), sub]))


    threads = []
    for cmd in commands:
        t = threading.Thread(target=run_command,  args=(cmd, db_name))
        t.start()
        threads.append(t)
        
    for t in threads:
        t.join()
 
def launch_print_with_stepwise(db_name, sub):
    
    directory = "reports/with_normalization/with_stepwise/" + db_name  +  "/" + sub + "/metrics/"
    
    commands = ["""make run_print_{0} DB_NAME={1} _DIR={2} DISCRETIZATION_TYPE={3} GRAPH_TYPE={4} SUB={5} """.
                format(*[db_name.lower(), db_name.lower(), directory, None, None, sub])]

    for graph in state_of_art_graphs:
            commands.append("""make run_print_{0} DB_NAME={1} _DIR={2}  DISCRETIZATION_TYPE={3} GRAPH_TYPE={4} SUB={5} """.
                            format(*[db_name.lower(), db_name.lower(), directory, None, graph.lower(), sub]))
            
    for graph in proposed_complete_graph:
            commands.append("""make run_print_{0} DB_NAME={1} _DIR={2} DISCRETIZATION_TYPE={3} GRAPH_TYPE={4} SUB={5} """.
                            format(*[db_name.lower(), db_name.lower(), directory, None, graph.lower(), sub]))  
            
    for graph in standard_proposed_graphs:
            for discretization in discretization_types:
                commands.append(
                    """make run_print_{0} DB_NAME={1} _DIR={2}  DISCRETIZATION_TYPE={3} GRAPH_TYPE={4} SUB={5}""".
                    format(*[db_name.lower(), db_name.lower(), directory, discretization, graph.lower(), sub]))


    threads = []
    for cmd in commands:
        t = threading.Thread(target=run_command,  args=(cmd, db_name))
        t.start()
        threads.append(t)
        
    for t in threads:
        t.join()
         
def launch_plot(db_name, sub):
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
    sub = args[1]
    
    launch_preprocess(db_name, sub)
    launch_build_engine_for_supervized_discretization(db_name, sub)
    launch_supervised_discretization(db_name, sub)
    launch_graph_modeling(db_name, sub)
    launch_compute_descriptors(db_name, sub)
    launch_config_without_stepwise(db_name, sub)
    launch_predict_classic(db_name, sub)
    launch_predict(db_name, sub)
    launch_print(db_name, sub)
    
    # ---------------------STEPWISE SECTION----------------------#
    
    launch_stepwise_selection(db_name, sub)
    launch_config_with_stepwise(db_name, sub)
    launch_predict_classic_with_stepwise(db_name, sub)
    launch_predict_with_stepwise(db_name, sub)
    launch_print_with_stepwise(db_name, sub)
    
    
    