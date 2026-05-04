import subprocess, sys


def launch_silm(db_name):
    command = """ make run_compute_descriptors_liu_{0}  BD_NAME={1} GRAPH_TYPE={2}  """.format(*[db_name.lower(), db_name.lower(), "liu"])
    
    process = subprocess.Popen(command, shell=True)

    process.wait()

if __name__ == "__main__":
    args = sys.argv[1:]
    db_name = args[0]
    
    launch_silm(db_name)