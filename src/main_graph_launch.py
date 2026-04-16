import subprocess


all_db_names = ["JAPANESE"]#"JAPANESE"] #, "AUSTRALIAN", "HMEQ"]#, "KAGGLE_CREDIT_RISK", "AER", "AUSTRALIAN", "JAPANESE", "HMEQ"]
graphs = ["com"]#, "mod", "com", "gui"]


def execution(db_names):
    commands = [] 
    for i in db_names:
        for j in graphs:
            commands.append(""" make run_graph_{0}_{1} DB_NAME={1}""".format(*[j.lower(), i.lower()]))

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
   
      