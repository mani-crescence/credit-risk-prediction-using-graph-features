import subprocess
import sys, ast, os
from dotenv import load_dotenv
import numpy as np
from .tools.execute import save_global_result, save_global_result_
import pandas as pd
from itertools import islice
import math

db_names = ["australian"]#]#["german", "hmeq", "australian", "japanese"]#, "hmeq"] #"kaggle_credit_risk",
discretization_types =  ["SUP", "UNS"]
graphs = [None, "bip", "bip", "mod", "mod", "com"]
discretizations = [None, "uns", "sup", "uns", "sup", None]
# models = ["LR", "SVM", "DT", "RF", "XGB", "LDA", "MLP"]
models = ["log", "svm", "dtree", "rf", "xgb", "lda"]
metrics = ["acc", "f1"]

def launch_attributes_importance():
    commands = []
    for discretization_type in discretization_types:
        for db_name in db_names:
            commands.append("""make run_summarize_{0} DB_NAME={1}  DISCRETIZATION_TYPE={2}""".format(*[db_name.lower(), db_name.lower(), discretization_type.lower()]))

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
                with open('reports/' + db + '/metrics/real/' + graph + '/main_results_r.txt') as file:
                    result = ast.literal_eval(file.read())
            # elif discretization is not None and graph is None:
            #     with open('outputs/general_results/results/'+ db +'/real/predictions/na/main_results_r.txt') as file:
            #         result = ast.literal_eval(file.read())
            elif discretization is None and graph is None:
                with open('reports/'+ db +'/metrics/classic/main_results_r.txt') as file:
                    result = ast.literal_eval(file.read())
            else:
                with open('reports/'+ db +'/metrics/real/'+ graph + '/main_results_' + graph + '_'+ discretization +'_r.txt') as file:
                    result = ast.literal_eval(file.read())
            global_results[db].update(result)

        save_global_result_(global_results[db], models, metrics, db)

    # exit(global_results)
    # data = {}
    # for model in models:
    #     for metric in metrics:
    #         for db in db_names:
    #             for conf, values in global_results[db].items():
    #                 data[conf] = {}
    #             break
    #
    #         for db in db_names:
    #             for conf, values in global_results[db].items():
    #                 data[conf][db] = values[model][metric]
    #
    #
    #
    #         directory = "outputs/general_results/"
    #         os.makedirs(directory, exist_ok=True)
    #         with open(directory + 'general_result_'+ model + '_'+ metric + '.txt', 'w') as file:
    #             file.write(str(data))
    #
    #         save_global_result(data, db_names, model, metric)

    # for db in db_names:
    #     for conf, values in global_results[db].items():
    #         data[conf] = {}
    #     break

    # for model in models:
    #     for db in db_names:
    #         for conf, values in global_results[db].items():
    #             data[conf][db] = {}
    #             for metric in metrics:
    #                 data[conf][db][metric] = values[model][metric]

    #     directory = "outputs/general_results/for_best_combination/"
    #     os.makedirs(directory, exist_ok=True)
    #     with open(directory + 'general_result_' + model + '.txt', 'w') as file:
    #         file.write(str(data))
    #
    #         save_global_result(data, db_names, metrics, model)

def attributes_classification():

    for discretization_type in discretization_types:
        datas = {}
        for db_name in db_names:
            datas[db_name] = {}

            for type in ['ORD', 'BIP', 'MOD']:
                datas[db_name][type]={}

            for model in models:
                data = pd.read_csv('outputs/general_results/results/' + db_name + '/shap/' + discretization_type.lower() + '/shap_data_bip_mod_' + model + '.csv')
                data.set_index('Unnamed: 0', inplace=True)
                data.index.name = None

                N = data.shape[0]
                importance = {}

                for col in data.columns:
                    sum = 0
                    for i in data[col]:
                        sum += abs(i)
                    importance[col] = sum / N
                sorted_importance_attributes = dict(sorted(importance.items(), key=lambda item: item[1], reverse=True))
                first_importance_attributes = dict(islice(sorted_importance_attributes.items(), 10))
                first_attributes = first_importance_attributes.keys()
                T = len(first_attributes)

                bip_attributes = []
                mod_attributes = []
                columns = []
                for col in first_attributes:
                    columns.append(col)

                ord_attributes = columns.copy()
                for col in columns:
                    if col.find('_bip') != -1:
                        bip_attributes.append(col)
                        ord_attributes.remove(col)
                    elif col.find('_mod') != -1:
                        mod_attributes.append(col)
                        ord_attributes.remove(col)

                bip = len(bip_attributes)
                mod = len(mod_attributes)
                ord = len(ord_attributes)

                datas[db_name]['BIP'][model] = round((bip / T) * 100, 2)
                datas[db_name]['MOD'][model] = round((mod / T) * 100, 2)
                datas[db_name]['ORD'][model] = round((ord / T) * 100, 2)

        M = len(models) + 1
        code = r"\begin{table}[H]" + "\n"
        code += r"\centering" + "\n"
        code += r"\scalebox{1}{" + "\n"
        code += r"\begin{tabular}{"
        code += r"|lc|"
        for _ in range(M):
            code += r"c|"
        code += r"} \hline" + "\n"
        code+= r"&"
        for model in models:
            code += r"&{}".format(model)
        code += r"&Total \\ \hline" + "\n"

        for db, types in datas.items():
            code += r"\multirow{3}{*}{" + r"{}".format(db) + r"}"
            i = 0
            for type, models_values in types.items():
                code += r"&{} (\%)".format(type)
                sum = 0
                for _, value in models_values.items():
                    code += r"&{}".format(value)
                    sum += value
                i += 1
                code += r"&{}".format(round(sum/(M-1), 2))
                if i < 3:
                    code+=r"\\" + "\n"
            code += r"\\ \hline" + "\n"

        code += r"\end{tabular}" + "\n"
        code += r"}" + "\n"
        code += r"\caption{" + "Pourcentages des contributions des nouveaux descripteurs} \n"
        code += r"\label{global-results}" + "\n"
        code += r"\end{table}" + "\n"

        directory = 'outputs/general_results/classification/'
        os.makedirs(directory, exist_ok=True)
        with open(directory + 'general_contributions_'+ discretization_type.lower()+'.tex', 'w') as file:
            file.write(code)

def generate_best_combination():
    for metric in ['acc', 'f1']:
        results = {}
        for model in models:
            results[model] = {}
            results[model]['clas'] = {}
            results[model]['pa'] = {}
            results[model]['comb'] = {}
            results[model]['va'] = {}

            for db in db_names:
                with open('outputs/general_results/results/' + db + '/predictions/classic/metrics_results.txt', 'r') as file:
                    classic = ast.literal_eval(file.read())
                classic = classic['CLASSIC']

                max_value = - math.inf
                best_conf = ''
                with open('outputs/general_results/for_best_combination/general_result_' + model + '.txt', 'r') as file:
                    data = ast.literal_eval(file.read())

                for conf, values in data.items():
                    if data[conf][db][metric] > max_value:
                        best_conf = conf
                        max_value = data[conf][db][metric]

                results[model]['clas'][db] = classic[model][metric]
                results[model]['comb'][db] = best_conf
                results[model]['pa'][db] = max_value
                results[model]['va'][db] = round(
                    (max_value * float(classic[model][metric])) / 100 + float(classic[model][metric]), 2)

        M = len(db_names)
        code = r"\begin{table}[H]" + "\n"
        code += r"\centering" + "\n"
        code += r"\scalebox{1}{" + "\n"
        code += r"\begin{tabular}{"
        code += r"|l|"

        for _ in range(M):
            code += r"c|"

        code += r"} \hline" + "\n"
        # code += r"&"
        for db in db_names:
            code += r"&{}".format(db)
        code += r"\\ \hline" + "\n"


        for model, values in results.items():
            # code += r"\multirow{3}{*}{" + r"{}".format(model) + r"}"

            # for i in values['clas'].values():
            #     code += r"& {}".format(i)
            # code += r"\\ " + "\n"

            for i, j in zip(values['comb'].values(), values['pa'].values()):
                code += r"{}&{}".format(model,i) + r"(+" + r"{}".format(j) + r"\%)"
            # code += r"\\" + "\n"

            # for i in values['comb'].values():
            #     code += r"& \footnotesize {}".format(i)
            code += r"\\ \hline" + "\n"

        code += r"\end{tabular}" + "\n"
        code += r"}" + "\n"
        code += r"\caption{" + "Meilleurs combinaisons suivant " + r"{}".format(metric) + "} \n"
        code += r"\label{global-results}" + "\n"
        code += r"\end{table}" + "\n"

        with open('outputs/general_results/best_combinations_summary_' + metric + '_reverse.tex', 'w') as file:
            file.write(code)

if __name__ == "__main__":
    # a=0
    generate_result()
    # launch_attributes_importance()
    # attributes_classification()
    # generate_best_combination()

    # for model, values in results.items():
    #     code += r"\multirow{3}{*}{" + r"{}".format(model) + r"}"
    #
    #     for i in values['clas'].values():
    #         code += r"& {}".format(i)
    #     code += r"\\ " + "\n"
    #
    #     for i, j in zip(values['va'].values(), values['pa'].values()):
    #         code += r"&{}".format(i) + r"(" + r"{}".format(j) + r"\%)"
    #     code += r"\\" + "\n"
    #
    #     for i in values['comb'].values():
    #         code += r"& \footnotesize {}".format(i)
    #     code += r"\\ \hline" + "\n"
    #
