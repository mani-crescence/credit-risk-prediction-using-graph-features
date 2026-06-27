import subprocess, threading
 

all_db_names = ["SME", "PROSPER", "LENDING_CLUB", "BONDORA"] #, "SME", "PROSPER", , "LENDING_CLUB", "BONDORA"] 
MAX_WORKERS = 4
semaphore = threading.Semaphore(MAX_WORKERS)


def execution(db_names):
    commands = [] 
    dbs = []
    
    for db_name in db_names:
        for i in range(1, 6):
            commands.append(""" make run_with_normalization_{0} DB_NAME={1} SUB={2} """.format(*[db_name.lower(), db_name.lower(), i]))
            dbs.append(db_name)
      
    
    def run_command(cmd, db_name):
        with semaphore:
            process = subprocess.Popen(cmd, shell=True)
            process.wait()
            print(f"##################### {db_name} processing completed. ############################")
    
    threads = []
    for cmd, db_name in zip(commands, dbs):
        t = threading.Thread(target=run_command,  args=(cmd, db_name))
        t.start()
        threads.append(t)
   
    for t in threads:
        t.join()    

                                                
if __name__ == "__main__":
    execution(all_db_names)
   
     