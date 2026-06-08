import subprocess
 

all_db_names = ["BONDORA"]#"JAPANESE"] #, "", "HMEQ"]#, "KAGGLE_CREDIT_RISK", "AER", "AUSTRALIAN", "JAPANESE", "HMEQ"]


def execution(db_names):
    commands = [] 
    
    for i in db_names:
        commands.append(""" make run_with_normalization_{0} DB_NAME={1} """.format(*[i.lower(), i.lower()]))

    processes = []
    for cmd in commands:
        process = subprocess.Popen(cmd, shell=True)
        processes.append(process)
    for p, db_name in zip(processes, db_names):
        p.wait()
        print(f"##################### {db_name} processing completed. ############################")


                                                
if __name__ == "__main__":
    execution(all_db_names)
    # for db_names in all_db_names:
    #     execution(db_names)
   
     