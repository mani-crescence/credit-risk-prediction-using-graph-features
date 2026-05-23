import subprocess
import sys 

alphas = [0.1, 0.3, 0.5, 0.7, 0.85, 0.9] 
normalization = ["norm", "unnorm"]

def launch_graph_modeling(db_name):
    
    command = """make run_graph_modeling_loan_{0} DB_NAME={1} GRAPH_TYPE={2} """.format(*[db_name.lower(), db_name.lower(), "loan"])
    
    process = subprocess.Popen(command, shell=True)
    process.wait()

def launch_silm(db_name):
    commands = []
     
    for k in normalization:
        train_path = "data/preprocessed/"+ db_name +"/preprocessed_data_train.csv"
        test_path = "data/preprocessed/"+ db_name +"/preprocessed_data_test.csv"
        
        for alpha in alphas:
            commands.append(""" make run_compute_descriptors_loan_{0}  BD_NAME={1} GRAPH_TYPE={2} ALPHA={3}  TRAIN_PATH={4} TEST_PATH={5} """.
                        format(*[db_name.lower(), db_name.lower(), "loan", alpha, train_path, test_path]))
    processes = []
    
    for cmd in commands:
        process = subprocess.Popen(cmd, shell=True)
        processes.append(process)
        
    for p in processes:
        p.wait()


if __name__ == "__main__":
    args = sys.argv[1:]
    db_name = args[0]
    
    launch_graph_modeling(db_name)
    launch_silm(db_name)
   
