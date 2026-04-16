import subprocess
import sys 

discretization_types =  ["UNS", "SUP"]
alphas = [0.1] 

def launch_graph_modeling(db_name):
    
    commands = []
       
    for discretization_type in discretization_types:
        commands.append("""make run_graph_modeling_bip_{0} DB_NAME={1} GRAPH_TYPE={2} DISCRETIZATION_TYPE={3} """.format(*[db_name.lower(), db_name.lower(), "bip", discretization_type]))
    
    processes = []
    
    for cmd in commands:
        
        process = subprocess.Popen(cmd, shell=True)
        processes.append(process)
        
    for p in processes:
        p.wait()

def launch_silm(db_name):
    commands = []
    
    for label in ["train", "test"]:
        for discretization_type in discretization_types:
            for alpha in alphas:
                commands.append(""" make run_compute_descriptors_bip_{0}  BD_NAME={1} GRAPH_TYPE={2} ALPHA={3}  DISCRETIZATION_TYPE={4} LABEL={5} """.format(*[db_name.lower(), db_name.lower(), "bip", alpha, discretization_type, label]))
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
   
