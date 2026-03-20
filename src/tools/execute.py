import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from ..tools.training import *
from ..tools.preprocessing import *
from ..tools.graph import *
from ..tools.cleaning import *
import pickle, re, math, ast, os


def build_graph_attributes(data, graph, descriptors, target, bd_name, alpha,  graph_type, label, discretization_type ):
    
    print(f'############## processing {discretization_type} with  alpha ==>{alpha} ######################')
    
    graph_descriptors = pd.DataFrame()
    
    if label == "train":
        for row in data.itertuples():
            
            graph_copy = graph.copy()
            
            nodes_for_personalization = []
            dict_row = row._asdict()
            del dict_row['Index']
            
            if graph_type == 'bip':
                
                graph_copy.remove_edge('tr_u' + str(row.Index), target + '_' + str(dict_row[target]) + '_' + discretization_type + '_' +graph_type)
                
                pagerank_attributes = pagerank_personalized(graph_copy, alpha, ['tr_u' + str(row.Index)], None, descriptors)
                
            elif graph_type == 'mod':
                
                for k, w in dict_row.items():
                    nodes_for_personalization.append(str(k) + '_' + str(w) + '_' + discretization_type + '_' + graph_type)
                
                pagerank_attributes = pagerank_personalized(graph_copy, alpha, nodes_for_personalization, 'weight', descriptors)
                
            # else:
            #     pagerank_attributes = pagerank_personalized(graph_copy, alpha, ['l'+str(row.Index)], 'weight', descriptors)
            
            graph_descriptors.loc[row.Index, list(pagerank_attributes.keys())] = list(pagerank_attributes.values())
            
        
        graph_descriptors = graph_descriptors.astype(float)
        
        directory='data/graph_features/'+bd_name+'/'+ discretization_type + '/' + graph_type +'/'+ label
        os.makedirs(directory, exist_ok=True)
        graph_descriptors.to_csv(directory + '/new_features_' +  str(alpha)+'.csv')
        
        # if discretization_type is None:
            
        # else:
        #     graph_descriptors.to_csv(directory + '/new_features_' + discretization_type + '_' + graph_type +'_'+str(alpha)+'.csv') 
    else:
        
        
        for row in data.itertuples():
            
            nodes_for_personalization = []
            dict_row = row._asdict()
            
            graph_copy = graph.copy()
    
            if graph_type == "bip":
                
                augmented_graph = graph_bipartite_modality(graph_copy, None, dict_row, discretization_type)
                pagerank_attributes  = pagerank_personalized(augmented_graph, alpha, ['ts_'+str(row.Index)], None, descriptors)
                
            elif graph_type == "mod":
                
                del dict_row[target]
                del dict_row['Index']
                
                for k, w in dict_row.items():
                   nodes_for_personalization.append(str(k) + '_' + str(w) + '_' + discretization_type + '_' + graph_type.lower())
                   
                augmented_graph = graph_modality(graph_copy, None, dict_row, discretization_type)
                pagerank_attributes = pagerank_personalized(augmented_graph, alpha, nodes_for_personalization, "weight", descriptors)
                
                      
            graph_descriptors.loc[row.Index, list(pagerank_attributes.keys())] = list(pagerank_attributes.values())
           
       

        graph_descriptors = graph_descriptors[descriptors]
        graph_descriptors = graph_descriptors.astype(float)
        
        directory='data/graph_features/'+bd_name+'/'+ discretization_type + '/' + graph_type +'/'+ label
        os.makedirs(directory, exist_ok=True)
        graph_descriptors.to_csv(directory + '/new_features_' +  str(alpha)+'.csv')
        
    print(f"finish processed ===> {discretization_type} with alpha {alpha} ")

    
def build_predictions(models, trainset, testset, configurations, target, classic_result = None):
    results_real = {}
    results_with_real = {}

    tr = trainset.copy()
    ts = testset.copy()

    for config_name, config_att in configurations.items():
        print(config_name)
        if len(config_att) > 1:
            tr_set = tr[config_att]
            ts_set = ts[config_att]
            results_real[config_name], results_with_real[config_name]  = train(models, tr_set, ts_set, target, classic_result)
    return results_real, results_with_real

def  arrange_result(dir, ptype, nb_metrics):
    inter ={}
    results = {}

    t = {'log': 0, 'svm': 0, 'dtree': 0, 'rf': 0, 'xgb': 0, 'lda':0}
    t2 = {'log': 0, 'svm': 0, 'dtree': 0, 'rf': 0, 'xgb': 0, 'lda':0}
    t3 = {'log': 0, 'svm': 0, 'dtree': 0, 'rf': 0, 'xgb': 0, 'lda':0}
    directory = dir + "predictions/"+ ptype.lower() +"/"
    paths = []
    for item in os.listdir(directory):
        dir = directory
        paths.append(os.path.join(dir, item))

    if nb_metrics == 3:
        for path in paths:
            with open(path,"rb" ) as f:
                conf = pickle.load(f)
            alpha = re.findall(r'\d+\.\d+', path)

            for k, l in conf.items():
                for i, v in l.items():
                        t[i] = v['acc']
                        t2[i] = v['f1']
                        t3[i] = v['cost']
                t_ = t.copy()
                t2_ = t2.copy()
                t3_ = t3.copy()
                inter[k] = {}
                inter[k]['acc'] = t_
                inter[k]['f1'] = t2_
                inter[k]['cost'] = t3_
            results[alpha[0]] = inter
            inter = {}

        return results
    else:
        for path in paths:
            with open(path,"rb" ) as f:
                conf = pickle.load(f)
            alpha = re.findall(r'\d+\.\d+', path)
            for k, l in conf.items():
                for i, v in l.items():
                        t[i] = v['acc']
                        t2[i] = v['f1']
                t_ = t.copy()
                t2_ = t2.copy()
                inter[k] = {}
                inter[k]['acc'] = t_
                inter[k]['f1'] = t2_
            results[alpha[0]] = inter
            inter = {}

        return results

def  arrange_one_result(dir, ptype, nb_metrics):
    inter ={}
    t = {'log': 0, 'svm': 0, 'dtree': 0, 'rf': 0, 'xgb': 0, 'lda':0}
    t2 = {'log': 0, 'svm': 0, 'dtree': 0, 'rf': 0, 'xgb': 0, 'lda':0}
    t3 = {'log': 0, 'svm': 0, 'dtree': 0, 'rf': 0, 'xgb': 0, 'lda':0}
    directory = dir + "predictions/"+ ptype.lower() +"/"
    paths = []

    for item in os.listdir(directory):
        dir = directory
        paths.append(os.path.join(dir, item))

    if nb_metrics == 3:
        for path in paths:
            with open(path,"rb" ) as f:
                conf = pickle.load(f)

            for k, l in conf.items():
                for i, v in l.items():
                        t[i] = v['acc']
                        t2[i] = v['f1']
                        t3[i] = v['cost']
                t_ = t.copy()
                t2_ = t2.copy()
                t3_ = t3.copy()
                inter[k] = {}
                inter[k]['acc'] = t_
                inter[k]['f1'] = t2_
                inter[k]['cost'] = t3_

        return inter
    else:
         for path in paths:
            with open(path,"rb" ) as f:
                conf = pickle.load(f)

            for k, l in conf.items():
                for i, v in l.items():
                        t[i] = v['acc']
                        t2[i] = v['f1']
                t_ = t.copy()
                t2_ = t2.copy()
                inter[k] = {}
                inter[k]['acc'] = t_
                inter[k]['f1'] = t2_
         return inter

def select_best_result(results, db_name, ptype):
    best_params = {}
    result = {}

    for k, j in results.items():
        for ka, ja in j.items():
            result[ka] = {}
            best_params[ka] = {}
            for kb, jb in ja.items():
                result[ka][kb] = {}
                for kc, _ in jb.items():
                    best_params[ka][kc] = 0
                    result[ka][kb][kc] = -math.inf
        break

    for k, j in results.items():
        for ka, ja in j.items():
            for kb, jb in ja.items():
                for kc, _ in jb.items():
                    if float(results[k][ka][kb][kc]) > result[ka][kb][kc]:
                        result[ka][kb][kc] = float(results[k][ka][kb][kc])
                        best_params[ka][kc] = k

    directory = 'outputs/'+ db_name +'/results/'
    os.makedirs(directory, exist_ok=True)
    with open(directory +'best_alpha_values_'+ptype.lower() , 'wb') as file:
        pickle.dump(best_params, file)

    return result

def print_result_html(results, models, p_type, dir, nb_metrics):
    html = "<table > \n"
    html += "<tr > <th> Config</th> \n "
    html += "<th> Metrics</th> \n "
    for m in models:
        html += "<th> {}</th>".format(m)
    html += "</tr> \n"
    html += "<tr>"

    for key, value in results.items():
        html += "<th rowspan='{0}'>{1}</th>".format(*[nb_metrics, key])
        t = 0
        for k, col in value.items():
            t += 1
            html +="<td>{}</td>".format(k)
            for k_, v in col.items():
                html+= "<td>{}</td>".format(v)
            html += "</tr>"

            if t < len(value):
                html += "<tr>"

    html += "</table>"

    html_with_styles = """
    <style>
    table {
        width: 60%;
        border: 1px solid black;
        border-collapse: collapse;
    }

    th, td{
        padding: 5px;
        border: 2px solid black;
    }
    </style>
    """ + html
    directory = dir + "print/"
    os.makedirs(directory, exist_ok=True)
    with open(directory + "/result_" +p_type.lower()+".html", "w", encoding="utf-8") as file:
        file.write(html_with_styles)

def print_result_latex(results, models, p_type, dir, nb_metrics, db_name):
    latex_code = r"\begin{table}[H]" + "\n"
    latex_code += r"\centering" + "\n"
    latex_code += r"\begin{tabular}{c|c|c" + "c" * len(models) + "}" + "\n"
    latex_code += r"\hline" + "\n"

    latex_code += "Données & Config & Métrique "
    for m in models:
        latex_code += " & {}".format(m)
    latex_code += r" \\" + "\n"
    latex_code += r"\hline" + "\n"
    l = len(results) * nb_metrics

    i = 0
    latex_code += r"\multirow{" + str(l) + r"}{*}{" + db_name + "}"

    for key, value in results.items():
        latex_code += r" & \multirow{" + str(nb_metrics) + r"}{*}{" + key + "}"
        t = 0
        for k, col in value.items():

            latex_code += r" & "

            latex_code += "{}".format(k)

            for _, v in col.items():
                latex_code += " & {}".format(v)

            t += 1
            i += 1

            if t < len(value):
                latex_code += r" \\  " + "\cline{3-9}"+ " \n"
                latex_code += r" & "

        if i < l:
            latex_code += r" \\  " + "\cline{2-9}"+ " \n"
        else:
            latex_code += r" \\  " + "\n"+ "\hline" + " \n"

    latex_code += r"\end{tabular}" + "\n"
    latex_code += r"\caption{Your Caption Here}" + "\n"
    latex_code += r"\label{tab:your_label}" + "\n"
    latex_code += r"\end{table}" + "\n"


    directory = dir + "print/"
    os.makedirs(directory, exist_ok=True)
    with open(directory + "/result_o_" +p_type.lower()+".latex", "w", encoding="utf-8") as file:
        file.write(latex_code)

def save_result_latex(models, metrics, results, dir):
    latex_code = r"\begin{table}[H]" + "\n"
    latex_code += r"\centering" + "\n"
    latex_code += r"\scalebox{0.65}{" + "\n"
    latex_code += r"\begin{tabular}{"

    nb_columns = len(models) * len(metrics)
    latex_code += r"|c"
    for _ in range(nb_columns):
        latex_code += r"|c"
    latex_code += r"} "   + "\n" + "  \hline" + "\n"
    latex_code += r"\multicolumn{" + "{}".format(nb_columns + 1) + r"}{|c|}{Modèles} \\  \hline" + "\n"

    latex_code +=r"\multirow{2}{*}{Config}"

    for model in models:
        latex_code+= "& "+ r"\multicolumn{"+ "{}".format(len(metrics)) +r"}{c}{" +"{}".format(model)+ r"}"
    latex_code += r"\\" + "  " + r"\cline{2-" +"{}".format(nb_columns + 1) + r"}" + "\n"

    for model in models:
        for metric in metrics:
            latex_code += "& {}".format(metric)
    latex_code +=  r"\\"  + "  \hline" + "\n"

    for config_name, metric_values in results.items():
        latex_code += "{}".format(config_name)
        for value in metric_values:
            temp = float(value)
            if  temp < 0:
               latex_code +=  " &  \cellcolor{red!25} " + " {}".format(value)
            elif temp == 0:
                latex_code +=  " & " + "{}".format(value)
            elif temp  > 0:
                latex_code +=  " & \cellcolor{green!25} " + "{}".format(value)


        latex_code+= r"\\"  + " \hline" + "\n"

    latex_code += r"\end{tabular}" + "\n"
    latex_code += r"}" + "\n"
    latex_code += r"\caption{Your Caption Here}" + "\n"
    latex_code += r"\label{tab:your_label}" + "\n"
    latex_code += r"\end{table}" + "\n"

    directory = dir + "print/"
    os.makedirs(directory, exist_ok=True)
    with open(directory + "/results.tex", "w", encoding="utf-8") as file:
        file.write(latex_code)

def result(directory, discretization_type, graph_type):
    print(discretization_type, graph_type, "\n")
    
    max_result = {}
    paths = []
    best_alpha_values = {}

    if discretization_type != "None" and graph_type != "None":
        files_dir = directory + "real/" + graph_type.lower() + "/" + discretization_type.lower()

        for item in os.listdir(files_dir):
            paths.append(os.path.join(files_dir, item))

        for path in paths:
            with open(path, "r") as f:
                file = ast.literal_eval(f.read())

            for config_name, models in file.items():
                 max_result[config_name] = {}
                 for model, metrics in models.items():
                     max_result[config_name][model] = {}
                     for metric, _ in metrics.items():
                         max_result[config_name][model][metric] = -math.inf
            break

        for path in paths:
            with open(path, "r") as f:
                file = ast.literal_eval(f.read())
            alpha = re.findall(r'\d+\.\d+', path)
            for config_name, models in file.items():
                for model, metrics in models.items():
                     for metric, value in metrics.items():
                         if float(max_result[config_name][model][metric]) < float(value):
                             max_result[config_name][model][metric] = float(value)
                             best_alpha_values[model] = alpha[0]

        with open(directory + "real/" + graph_type.lower()  + '/main_results_' + graph_type.lower() + '_'+ discretization_type.lower() +'_r.txt', 'w') as file:
            file.write(str(max_result))

        with open(directory + "real/" + graph_type.lower()  + '/best_alpha_values_' + graph_type.lower() + '_'+ discretization_type.lower() +'_r.txt', 'w') as file:
            file.write(str(best_alpha_values))


    elif  discretization_type == "None" and graph_type != "None":
        path = directory + "real/" + graph_type.lower() + "/metrics_results.txt"
        # for item in os.listdir(files_dir):
        #     paths.append(os.path.join(files_dir, item))

        # for path in paths:
        #     with open(path, "r") as f:
        #         file = ast.literal_eval(f.read())

        #     for config_name, models in file.items():
        #         max_result[config_name] = {}
        #         for model, metrics in models.items():
        #             max_result[config_name][model] = {}
        #             for metric, _ in metrics.items():
        #                 max_result[config_name][model][metric] = -math.inf
        #     break

        # for path in paths:
        with open(path, "r") as f:
            file = ast.literal_eval(f.read())
            
        for config_name, models in file.items():
            max_result[config_name] = {}
            for model, metrics in models.items():
                max_result[config_name][model] = {}
                for metric, value in metrics.items():
                    max_result[config_name][model][metric] = float(value)

        os.makedirs(directory + "real/" + graph_type.lower() , exist_ok=True)
        with open(directory + "real/" + graph_type.lower() + '/main_results_r.txt', 'w') as file:
            file.write(str(max_result))

    #     with open(directory + "real/predictions/" + graph_type.lower()  + '/best_alpha_values_r.txt', 'w') as file:
    #         file.write(str(best_alpha_values))

    # elif discretization_type != "None" and graph_type == "None":
    #     files_dir = directory + "real/predictions/na/"
    #     max_result = load_result(files_dir)
    #     os.makedirs(files_dir, exist_ok=True)
    #     with open(files_dir + '/main_results_r.txt', 'w') as file:
    #         file.write(str(max_result))
    else:
        files_dir = directory + "/classic/"
        max_result = load_result(files_dir)
        os.makedirs(files_dir, exist_ok=True)
        with open(files_dir + '/main_results_r.txt', 'w') as file:
            file.write(str(max_result))

def load_result(directory):
    paths = []
    max_result = {}

    for item in os.listdir(directory):
        paths.append(os.path.join(directory, item))

    for path in paths:
        with open(path) as f:
            file = ast.literal_eval(f.read())
            # print(file, path)

        for config_name, models in file.items():
            max_result[config_name] = {}
            for model, metrics in models.items():
                max_result[config_name][model] = {}
                for metric, value in metrics.items():
                    max_result[config_name][model][metric] = float(value)
    return max_result

def save_global_result(data, db_names, metrics, model):
    code = r"\begin{table}[H]" + "\n"
    code += r"\centering" + "\n"
    code += r"\scalebox{0.7}{" + "\n"
    code += r"\begin{tabular}{"
    nb_db = len(db_names) * len(metrics)
    code += r"|l|"
    for _ in range(nb_db):
        code += r"c|"
    code += r"} \hline" + "\n"

    code += r"\multirow{2}{*}{Configurations}"

    for db in db_names:
        code += r"& \multicolumn{" + r"2}{" + r"c|}{"+ r"{}".format(db) + r"}"
    code += r" \\ \cline{2-13}" + "\n"

    for _ in db_names:
        for metric in metrics:
            code += "& {}".format(metric)
    code +=  r"\\"  + "  \hline" + "\n"


    for conf, dbs in data.items():
        code += r"{}".format(conf)
        for db, metrics in dbs.items():
            for metric, value in metrics.items():
                code += r" & {}".format(value)
        code += r" \\ \hline " + "\n"


    code += r"\end{tabular}" + "\n"
    code += r"}" + "\n"
    code += r"\caption{" +"Résulats généraux suivant le modèle " + r"{}".format(model)  + "} \n"
    code += r"\label{global-results}" +"\n"
    code += r"\end{table}" + "\n"

    directory = "outputs/general_results"
    os.makedirs(directory, exist_ok=True)
    with open(directory + "/general_results_"+ model + "_.tex", "w", encoding="utf-8") as file:
        file.write(code)

def save_global_result_(data, models, metrics, db):
    # exit(data)
    code = r"\begin{table}[H]" + "\n"
    code += r"\centering" + "\n"
    code += r"\scalebox{0.7}{" + "\n"
    code += r"\begin{tabular}{"
    nb_db = len(models) * len(metrics)
    code += r"|l|"
    for _ in range(nb_db):
        code += r"c|"
    code += r"} \hline" + "\n"

    code += r"\multirow{2}{*}{Configurations}"

    for model in models:
        code += r"& \multicolumn{" + r"2}{" + r"c|}{"+ r"{}".format(model) + r"}"
    code += r" \\ \cline{2-" +r"{}".format(nb_db-1) + "} \n"

    for _ in models:
        for metric in metrics:
            code += "& {}".format(metric)
    code +=  r"\\"  + "  \hline" + "\n"

    for conf_name, conf_values in data.items():
        code += r"{}".format(conf_name)
        # exit(conf_values)
        for metric_name, metric_values in conf_values.items():
            for name, value in metric_values.items():
                    code += "& {}".format(value)

        code +=  r"\\"  + "  \hline" + "\n"


    # for conf, dbs in data.items():
    #     code += r"{}".format(conf)
    #     for db, metrics in dbs.items():
    #         for metric, value in metrics.items():
    #             code += r" & {}".format(value)
    #     code += r" \\ \hline " + "\n"


    code += r"\end{tabular}" + "\n"
    code += r"}" + "\n"
    code += r"\caption{" +"Résulats généraux suivant le modèle " + r"{}".format(db)  + "} \n"
    code += r"\label{global-results}" +"\n"
    code += r"\end{table}" + "\n"

    directory = "reports/summary"
    os.makedirs(directory, exist_ok=True)
    with open(directory + "/general_results_"+ db + "_r.tex", "w", encoding="utf-8") as file:
        file.write(code)

def compute_degree_centralities(graph, trainset, testset,  db_name, graph_type, target):
    
    directory='data/graph_features/' + db_name +'/' + graph_type +'/'
        
    os.makedirs(directory, exist_ok=True)
    
    training_centralities_df = pd.DataFrame(columns=['deg0', 'deg1'])
    test_centralities_df = pd.DataFrame(columns=['deg0', 'deg1'])
    
    
    ### COMPUTATION OF DEGREE CENTRALITIES IN TRAINING SET    
    
    for i, _ in  trainset.iterrows():
        
        print('tr_u' + str(i), '\n')
        
        neighbors = list(nx.all_neighbors(graph, 'tr_u' + str(i)) ) 
        
        all_nodes = graph.nodes(data=True)
        
        training_nodes = [node  for node, att in all_nodes if att.get("type") == 'train']
        
        training_neighbors = [node for node in training_nodes if node in neighbors]
       
        indexes = []
        
        for node in training_neighbors:
            number = (re.findall(r'\d+',  node))[0]
            indexes.append(int(number)) 
         
        subset = pd.DataFrame(trainset, index=indexes) 
        
        subset[target] = subset[target].astype(int)
        
        targets = {1: 0, 0: 0}
         
        target_counts = subset[target].value_counts().to_dict() 
        
        try:
            if target_counts[1] is not None:
                targets[1]= target_counts[1]
                
            if target_counts[0] is not None:
                targets[0]= target_counts[0]
        except:  
            print("something went wrong")     
          
        training_centralities_df.loc[i] = [targets[0] , targets[1]]
      
        
    training_centralities_df = training_centralities_df.astype(float)
    
    training_centralities_df.to_csv(directory + '/new_features_train.csv')
        
        
        
    ### COMPUTATION OF DEGREE CENTRALITIES IN TESTING SET    
    
    for i, _ in  testset.iterrows():
        neighbors = list(nx.all_neighbors(graph, 'ts_u' + str(i)) ) 
        
        all_nodes = graph.nodes(data=True)
        
        training_nodes = [node  for node, att in all_nodes if att.get("type") == 'train']
        
        training_neighbors = [node for node in training_nodes if node in neighbors]
       
        indexes = []
        
        for node in training_neighbors:
            number = (re.findall(r'\d+',  node))[0]
            indexes.append(int(number)) 
         
        subset = pd.DataFrame(trainset, index=indexes) 
        subset[target] = subset[target].astype(int)
        
        targets = {1: 0, 0: 0}
         
        target_counts = subset[target].value_counts().to_dict() 
        
        try:
            if target_counts[1] is not None:
                targets[1]= target_counts[1]
                
            if target_counts[0] is not None:
                targets[0]= target_counts[0]
        except:  
            print("something went wrong")     
            
        test_centralities_df.loc[i] = [targets[0] , targets[1]]
        
    test_centralities_df = test_centralities_df.astype(float)
        
    test_centralities_df.to_csv(directory + '/new_features_test.csv')
    
    # print(training_centralities_df.shape, test_centralities_df.shape)
            
        
        
        
        
        
        
        
        
        
        
        
         
            
            
        
            
            
                     
        
        
    
    
    


                    
       
    

  