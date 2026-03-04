import subprocess
import sys, ast, os
from dotenv import load_dotenv
import numpy as np
from tools.execute import save_global_result


discretization_types = ["UNS", "SUP"]
discretization_for_attributes_types = ["UNS_", "SUP_"]
alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9]
process_type_prediction = ["UNS", "SUP", "SUP_", "UNS_"]
plot_type = ["UNS","SUP"]
pagerank_type = ["PER", "GLO"]
graph_types = ["BIP", "MOD"] #"LOAN"
graph_type_for_prediction = ["MOD", "BIP"]
graphs = ["bip", "bip", "mod", "mod", None, None]
discretizations = ["uns", "sup", "uns", "sup", "na", None]
models = ["log", "svm", "rf", "dtree", "lda", "xgb", "mlp"]
metrics = ["acc", "f1"]
db_names = ["german", "kaggle_credit_risk",  "lgd", "aer", "thomas", "australian",  "mortgage", "japanese", "hmeq"]

load_dotenv()

def launch_preprocess(db_name):
    commands = []
    commands.append(""" make run_preprocess_{0} DB_NAME={1}""".format(*[db_name.lower(), db_name.lower()]))

    processes = []   
    for cmd in commands:
        process = subprocess.Popen(cmd, shell=True)   
        processes.append(process)
        
    for p in processes:
        p.wait()       
        
    print("------------> Done submitting jobs !!!")

def launch_sampling(db_name):
    commands = [""" make run_sampling_{0} DB_NAME={1}""".format(*[db_name.lower(), db_name.lower()])]
    processes = []
    for cmd in commands:
        process = subprocess.Popen(cmd, shell=True)
        processes.append(process)

    for p in processes:
        p.wait()

    print("------------> Done submitting jobs !!!")

def launch_disc(db_name):
    commands = []
    for type in discretization_types:
            commands.append("""make run_discretization_{0} DB_NAME={1} T={2} """.format(*[db_name.lower(), db_name.lower(), type]))
   
    processes = []
    
    for cmd in commands:
        process = subprocess.Popen(cmd, shell=True)   
        processes.append(process)
        
    for p in processes:
        p.wait()     
        print(f"Discretization command '{p}' completed.")
        
def launch_graph_modeling(db_name):
    commands = []
    for graph_type in graph_types:
        if graph_type == "LOAN":
            commands.append("""make run_graph_modeling_{0} DB_NAME={1} DISCRETIZATION_TYPE={2} GRAPH_TYPE={3} """.format(*[db_name.lower(), db_name.lower(), None, graph_type]))
        else:    
            for discretization_type in discretization_types:
                commands.append("""make run_graph_modeling_{0} DB_NAME={1} DISCRETIZATION_TYPE={2} GRAPH_TYPE={3} """.format(*[db_name.lower(), db_name.lower(), discretization_type, graph_type]))
    
    processes = []
    
    for cmd in commands:
        process = subprocess.Popen(cmd, shell=True)  
        processes.append(process)
        
    for p in processes:
        p.wait()   
                        
def launch_silm(db_name):
    commands = []
    for graph_type in graph_types:
        if graph_type == 'LOAN':
            for alpha in alphas:
                commands.append("""make run_compute_descriptors_{0}  BD_NAME={1} ALPHA={2} GRAPH_TYPE={3}""".format(*[db_name.lower(), db_name.lower(), alpha, graph_type]))
        else:
            for discretization_type in discretization_types:
                for alpha in alphas:
                    commands.append("""make run_compute_descriptors_{0}  BD_NAME={1} ALPHA={3} GRAPH_TYPE={4} DISCRETIZATION_TYPE={2} """.format(*[db_name.lower(), db_name.lower(), discretization_type, alpha, graph_type]))
    processes = []
    
    for cmd in commands:
        process = subprocess.Popen(cmd, shell=True)  
        processes.append(process)
        
    for p in processes:
        p.wait()

def launch_conf(db_name):
    commands = []

    commands.append("""make run_make_configurations_{0}  DISCRETIZATION_TYPE={1} GRAPH_TYPE={2}""".format(*[db_name.lower(), None, None]))

    for graph_type in graph_types:
        if graph_type == "LOAN":
             commands.append("""make run_make_configurations_{0}  DISCRETIZATION_TYPE={1} GRAPH_TYPE={2}""".format(*[db_name.lower(), None, graph_type]))
        else:
            for type_id in discretization_types:
                commands.append("""make run_make_configurations_{0}  DISCRETIZATION_TYPE={1} GRAPH_TYPE={2}""".format(*[db_name.lower(), type_id, graph_type]))

    for disc_type in discretization_types:
        commands.append("""make run_make_configurations_{0}  DISCRETIZATION_TYPE={1} GRAPH_TYPE={2}""".format(*[db_name.lower(), disc_type, None]))

    processes = []
    for cmd in commands:
        process = subprocess.Popen(cmd, shell=True)
        processes.append(process)

    for p in  processes:
        p.wait()

def launch_predict(db_name):
    commands = []

    for graph_type in graph_types:
        if graph_type == "LOAN":
            train_directory = 'outputs/'+db_name.lower()+'/new_descriptors/'+graph_type+'/train'
            test_directory = 'outputs/'+db_name.lower()+'/new_descriptors/'+graph_type+'/test'
            config_path = "outputs/"+db_name+"/configurations/configuration_" +graph_type + ".txt"

            for alpha in alphas:
                train_path = train_directory+'/new_descriptors_data_' + graph_type+'_'+str(alpha)+'.csv'
                test_path = test_directory+'/new_descriptors_data_' + graph_type+'_'+str(alpha)+'.csv'
                commands.append("""make run_make_predictions_{0} DB_NAME={1} TRAIN_PATH={2} TEST_PATH={3}  DISCRETIZATION_TYPE={4} GRAPH_TYPE={5} CONFIG_PATH={6} ALPHA={7}""".format(*[db_name.lower(), db_name.lower(), train_path, test_path, None, graph_type, config_path, alpha]))
        else:
            for disc_type in discretization_types:
                train_directory = 'outputs/'+db_name.lower()+'/new_descriptors/'+graph_type+'/train'
                test_directory = 'outputs/'+db_name.lower()+'/new_descriptors/'+graph_type+'/test'
                config_path = "outputs/"+db_name+"/configurations/configuration_" +graph_type+"_"+disc_type+ ".txt"

                for alpha in alphas:
                    train_path = train_directory+'/new_descriptors_data_'+ disc_type+ '_'+graph_type+'_'+str(alpha)+'.csv'
                    test_path = test_directory+'/new_descriptors_data_'+ disc_type+ '_'+ graph_type+'_'+str(alpha)+'.csv'
                    commands.append("""make run_make_predictions_{0} DB_NAME={1} TRAIN_PATH={2} TEST_PATH={3}  DISCRETIZATION_TYPE={4} GRAPH_TYPE={5} CONFIG_PATH={6} ALPHA={7}""".format(*[db_name.lower(), db_name.lower(), train_path, test_path, disc_type, graph_type, config_path, alpha]))

    for disc_type in discretization_types :
        train_path = 'outputs/'+db_name.lower()+'/data/discretized/new_discretized_train_'+disc_type+'.csv'
        test_path = 'outputs/'+db_name.lower()+'/data/discretized/new_discretized_test_'+disc_type+'.csv'
        config_path = "outputs/"+db_name+"/configurations/configuration_" + disc_type + ".txt"
        commands.append("""make run_make_predictions_{0}  DB_NAME={1} TRAIN_PATH={2} TEST_PATH={3} DISCRETIZATION_TYPE={4} CONFIG_PATH={5} GRAPH_TYPE={6} """.format(*[db_name.lower(), db_name.lower(), train_path, test_path, disc_type, config_path,  None]))


    processes = []

    for cmd in commands:
        process = subprocess.Popen(cmd, shell=True)
        processes.append(process)

    for p in  processes:
        p.wait()

def launch_predict_classic(db_name):
    train_path = 'outputs/'+db_name.lower()+'/data/classic/trainset.csv'
    test_path = 'outputs/'+db_name.lower()+'/data/classic/testset.csv'
    config_path = "outputs/"+db_name+"/configurations/configuration_CLASSIC" +".txt"
    command = """make run_make_predictions_{0} DB_NAME={1} TRAIN_PATH={2} TEST_PATH={3} DISCRETIZATION_TYPE={4} GRAPH_TYPE={5} CONFIG_PATH={6} """.format(*[db_name.lower(), db_name.lower(), train_path, test_path, None, None, config_path, "CLASSIC"])
    subprocess.run(command, shell=True)        

def launch_print(db_name):
    commands = []

    commands.append("""make run_print_{0} DB_NAME={1}  DISCRETIZATION_TYPE={2} GRAPH_TYPE={3} """.format(
        *[db_name.lower(), db_name.lower(), None, None]))

    commands.append("""make run_print_{0} DB_NAME={1}  DISCRETIZATION_TYPE={2} GRAPH_TYPE={3} """.format(
        *[db_name.lower(), db_name.lower(), 'na', None]))

    for graph in graph_types:
        if graph == "LOAN":
            commands.append("""make run_print_{0} DB_NAME={1}  DISCRETIZATION_TYPE={2} GRAPH_TYPE={3} """.format(
                *[db_name.lower(), db_name.lower(), None, graph]))
        else:
            for discretization in discretization_types:
                commands.append(
                    """make run_print_{0} DB_NAME={1}  DISCRETIZATION_TYPE={2} GRAPH_TYPE={3} """.format(
                        *[db_name.lower(), db_name.lower(), discretization, graph]))


    processes = []
    
    for cmd in commands:
        process = subprocess.Popen(cmd, shell=True)  
        processes.append(process)
        
    for p in processes: 
        p.wait()        
        
def launch_plot(db_name):
    commands = []

    # for model in models:
    #     train_descriptors_paths = []
    #     test_descriptors_paths = []
    #     best_alphas_path = 'outputs/'+ db_name +'/results/predictions/loan/best_alpha_values.txt'
    #     with open(best_alphas_path, "r") as file:
    #         alphas = ast.literal_eval(file.read())
    #     alpha = alphas[model]
    #     train_descriptors_paths.append('\'outputs/' + db_name + '/new_descriptors/LOAN/train/new_descriptors_data_LOAN_' + str(alpha) + '.csv\'')
    #     test_descriptors_paths.append('\'outputs/' + db_name + '/new_descriptors/LOAN/test/new_descriptors_data_LOAN_' + str(alpha) + '.csv\'')
    #     commands.append("""make run_plot_{0} DB_NAME={1} TRAIN_DESCRIPTORS_PATHS={2} TEST_DESCRIPTORS_PATHS={3} MODEL={4} DISCRETIZATION_TYPE={5} """.format(
    #             *[db_name.lower(), db_name.lower(), train_descriptors_paths, test_descriptors_paths, model, None]))

    for model in models:
        for discretization_type in discretization_types:
            train_descriptors_paths = []
            test_descriptors_paths = []
            for graph in ["BIP", "MOD"]:
                best_alphas_path = 'outputs/'+ db_name +'/results/predictions/' + graph.lower()  + '/best_alpha_values_' + graph.lower() + '_' + discretization_type.lower() + '.txt'
                with open(best_alphas_path, "r") as file:
                    alphas = ast.literal_eval(file.read())
                alpha = alphas[model]
                train_descriptors_paths.append('outputs/' + db_name + '/new_descriptors/' + graph + '/train/new_descriptors_data' + '_' + discretization_type + '_' +  graph + '_' + str(alpha) + '.csv')
                test_descriptors_paths.append('outputs/' + db_name + '/new_descriptors/' + graph  + '/test/new_descriptors_data' + '_' + discretization_type  + '_' + graph + '_' + str(alpha) + '.csv')
            commands.append("""make run_plot_{0} DB_NAME={1} TRAIN_DESCRIPTORS_PATHS={2} TEST_DESCRIPTORS_PATHS={3}  MODEL={4} DISCRETIZATION_TYPE={5} """.format(
                *[db_name.lower(), db_name.lower(),'\"' + str(train_descriptors_paths) + '\"', f'\"{test_descriptors_paths}\"', model, discretization_type]))

    processes = []
    
    for cmd in commands:
        process = subprocess.Popen(cmd, shell=True)
        processes.append(process)
        
    for p in processes: 
        p.wait()                   

def generate_result():
    global_results = {}

    for db in db_names:
        global_results[db] = {}
        for graph, discretization in zip(graphs, discretizations):
            if discretization is None and graph is not None:
                with open('outputs/' + db + '/results/predictions/' + graph + '/main_results.txt') as file:
                    result = ast.literal_eval(file.read())
            elif discretization is not None and graph is None:
                with open('outputs/'+ db +'/results/predictions/na/main_results.txt') as file:
                    result = ast.literal_eval(file.read())
            elif discretization is None and graph is None:
                with open('outputs/'+ db +'/results/predictions/classic/main_results.txt') as file:
                    result = ast.literal_eval(file.read())
            else:
                with open('outputs/'+ db +'/results/predictions/'+ graph + '/main_results_' + graph + '_'+ discretization +'.txt') as file:
                    result = ast.literal_eval(file.read())
            global_results[db].update(result)

    data = {}
    for model in models:
        for metric in metrics:
            for db in db_names:
                for conf, values in global_results[db].items():
                    data[conf] = {}
                break

            for db in db_names:
                for conf, values in global_results[db].items():
                    data[conf][db] = values[model][metric]

            directory = "outputs/general_results/"
            os.makedirs(directory, exist_ok=True)
            with open(directory + 'general_result_'+ model + '_'+ metric + '.txt', 'w') as file:
                file.write(str(data))

            save_global_result(data, db_names, model, metric)


if __name__ == "__main__":
    args = sys.argv[1:]
    db_name = args[0]
    # launch_sampling(db_name)
    launch_preprocess(db_name)
    launch_disc(db_name)
    launch_graph_modeling(db_name)
    launch_silm(db_name)
    launch_conf(db_name)
    launch_predict_classic(db_name)
    launch_predict(db_name)
    launch_print(db_name)
    launch_plot(db_name)
    # generate_result()

