import subprocess
 

all_db_names =["LENDING_CLUB"] #["SME", "PROSPER", "BONDORA"] #"JAPANESE"] #, "", "HMEQ"]#, "KAGGLE_CREDIT_RISK", "AER", "AUSTRALIAN", "JAPANESE", "HMEQ"] "PROSPER" "SME"


def execution(db_name):
    # commands = [] 
    
    # for i in db_names:
    command = """ make run_with_normalization_{0} DB_NAME={1} """.format(*[db_name.lower(), db_name.lower()])
    subprocess.run(command, shell=True)

    # processes = []
    # for cmd in commands:
    #     process = subprocess.Popen(cmd, shell=True)
    #     processes.append(process)
    # for p, db_name in zip(processes, db_names):
    #     p.wait()
    #     print(f"##################### {db_name} processing completed. ############################")


                                                
if __name__ == "__main__":
    for i in all_db_names:
        execution(i)
    # for db_names in all_db_names:
    #     execution(db_names)
   
     