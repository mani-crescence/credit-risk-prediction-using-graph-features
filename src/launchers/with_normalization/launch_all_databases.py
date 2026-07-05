import subprocess, threading
 

all_db_names = ["SME", "PROSPER", "LENDING_CLUB", "BONDORA"]


def execution(db_names):
    
    dbs = []
    
    commands = []

    for db_name in db_names:
        commands.append(
            f"make run_with_normalization_{db_name.lower()} DB_NAME={db_name.lower()}"
        )
        dbs.append(db_name)

    processes = []

    for cmd in commands:
        p = subprocess.Popen(cmd, shell=True)
        processes.append(p)

    for p in processes:
        p.wait()

                                           
if __name__ == "__main__":
    execution(all_db_names)
   
     