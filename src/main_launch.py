import subprocess, ast

all_db_names =  [["AUSTRALIAN", "JAPANESE"], ["GERMAN", "THOMAS"], ["AER", "LGD"], ["HMEQ", "MORTGAGE"], ["KAGGLE_CREDIT_RISK", "LC"]]

def execution(db_names):
    commands = []
    for i in db_names:
        commands.append(""" make run_{0} DB_NAME={1}""".format(*[i.lower(), i.lower()]))

    processes = []
    for cmd in commands:
        process = subprocess.Popen(cmd, shell=True)
        processes.append(process)

    for p, db_name in zip(processes, db_names):
        p.wait()
        print(f"##################### {db_name} processing completed. ############################")
        
    print("------------> Done submitting jobs !!!")   
                                                
if __name__ == "__main__":
    for db_names in all_db_names:
        execution(db_names)
   
     