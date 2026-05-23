import subprocess
import sys 

discretization_types =  ["SUP"]#,"SUP"]
alphas = [0.1, 0.3, 0.5, 0.7, 0.85, 0.9] 
normalization = ["norm", "unnorm"]

def launch_graph_modeling(db_name):
    
    commands = []
       
    for k in normalization:
        for discretization_type in discretization_types:
            train_path = "data/discretized/" + k + "/" + db_name.lower() + "/discretized_train_data_" + discretization_type + ".csv"
            test_path = "data/discretized/" + k + "/" + db_name.lower() + "/discretized_test_data_" + discretization_type + ".csv"

        commands.append("""make run_graph_modeling_mod_{0} DB_NAME={1} GRAPH_TYPE={2} DISCRETIZATION_TYPE={3} TRAIN_PATH={4} TEST_PATH={5} """.
                        format(*[db_name.lower(), db_name.lower(), "mod", discretization_type, train_path, test_path]))
    
    processes = []
    
    for cmd in commands:
        process = subprocess.Popen(cmd, shell=True)
        processes.append(process)
        
    for p in processes:
        p.wait()

def launch_silm(db_name):
    commands = []
    
    for k in normalization:
        for discretization_type in discretization_types:
            train_path = "data/discretized/" + k + "/" + db_name.lower() + "/discretized_train_data_" + discretization_type + ".csv"
            test_path = "data/discretized/" + k + "/" + db_name.lower() + "/discretized_test_data_" + discretization_type + ".csv"

        for alpha in alphas:
            commands.append(""" make run_compute_descriptors_mod_{0}  BD_NAME={1} GRAPH_TYPE={2} ALPHA={3}  DISCRETIZATION_TYPE={4} TRAIN_PATH={5} TEST_PATH={6} """.
                            format(*[db_name.lower(), db_name.lower(), "mod", alpha, discretization_type, train_path, test_path]))
    processes = []
    
    for cmd in commands:
        process = subprocess.Popen(cmd, shell=True)
        processes.append(process)
        
    for p in processes:
        p.wait()


if __name__ == "__main__":
    args = sys.argv[1:]
    db_name = args[0]
    
    # launch_graph_modeling(db_name)
    launch_silm(db_name)
   
