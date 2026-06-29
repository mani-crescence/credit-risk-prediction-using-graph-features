import subprocess, threading
 

all_db_names = ["SME", "BONDORA", "PROSPER"] #, "PROSPER", "LENDING_CLUB", "BONDORA"] #, "SME", "PROSPER", , "LENDING_CLUB", "BONDORA"] 
# MAX_WORKERS = 5
# semaphore = threading.Semaphore(MAX_WORKERS)
BATCH_SIZE = 4


def execution(db_names):
    
    dbs = []
    
    for i in range(1, 6):
        commands = []

        for db_name in db_names:
            commands.append(
                f"make run_with_normalization_{db_name.lower()} DB_NAME={db_name.lower()} SUB={i}"
            )
            dbs.append(db_name)

        # Launch processes in batches of 2
        # for i in range(0, len(commands), BATCH_SIZE):
        #     batch = commands[i:i + BATCH_SIZE]
        processes = []

            # Start the batch
        for cmd in commands:
            p = subprocess.Popen(cmd, shell=True)
            processes.append(p)

        # Wait for the batch to finish
        for p in processes:
            p.wait()

    
    # for db_name in db_names:
    #     commands = [] 
    #     for i in range(1, 6):
    #         commands.append(""" make run_with_normalization_{0} DB_NAME={1} SUB={2} """.format(*[db_name.lower(), db_name.lower(), i]))
    #         dbs.append(db_name)
    
    #     processes = []
        
    #     for cmd in commands:
    #         process = subprocess.Popen(cmd, shell=True)
    #         processes.append(process)
            
    #     for p in processes:
    #         p.wait()     
    #         print(f"Su")  
    
    # def run_command(cmd, db_name):
    #     with semaphore:
    #         process = subprocess.Popen(cmd, shell=True)
    #         process.wait()
    #         print(f"##################### {db_name} processing completed. ############################")
    
    # threads = []
    # for cmd, db_name in zip(commands, dbs):
    #     t = threading.Thread(target=run_command,  args=(cmd, db_name))
    #     t.start()
    #     threads.append(t)
   
    # for t in threads:
    #     t.join()    

                                                
if __name__ == "__main__":
    execution(all_db_names)
   
     