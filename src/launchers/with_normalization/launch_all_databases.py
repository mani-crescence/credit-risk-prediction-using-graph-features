import subprocess, threading
 

all_db_names = ["LENDING_CLUB", "SME", "PROSPER", "BONDORA"] 
MAX_WORKERS = 2
semaphore = threading.Semaphore(MAX_WORKERS)


def execution(db_names):
    commands = [] 
    
    for db_name in db_names:
        for i in range(1,5):
            commands.append(""" make run_with_normalization_{0} DB_NAME={1} SUB={2} """.format(*[db_name.lower(), db_name.lower(), i]))
        
    
    def run_command(cmd, db_name):
        with semaphore:
        # for cmd in commands:
            process = subprocess.Popen(cmd, shell=True)
        #     processes.append(process)
        # for p, db_name in zip(processes, db_names):
            process.wait()
            print(f"##################### {db_name} processing completed. ############################")
    
    threads = []
    for cmd, db_name in zip(commands, db_names):
        t = threading.Thread(target=run_command,  args=(cmd, db_name))
        t.start()
        threads.append(t)
        
    for t in threads:
        t.join()    

                                                
if __name__ == "__main__":
    execution(all_db_names)
   
     