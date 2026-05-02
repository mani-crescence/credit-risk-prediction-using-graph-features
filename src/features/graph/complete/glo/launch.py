import subprocess
import sys

alphas = [0.3]#, 0.3, 0.5, 0.7, 0.85, 0.9] 

def launch_silm(db_name):
    commands = []

    for alpha in alphas:
        commands.append(""" make run_compute_descriptors_glo_{0}  BD_NAME={1} GRAPH_TYPE={2} ALPHA={3}  """
                        .format(*[db_name.lower(), db_name.lower(), "glo", alpha]))
    processes = []
    
    for cmd in commands:
        process = subprocess.Popen(cmd, shell=True)
        processes.append(process)
        
    for p in processes:
        p.wait()
        
        
if __name__ == "__main__":
    args = sys.argv[1:]
    db_name = args[0]   
    
    launch_silm(db_name)     