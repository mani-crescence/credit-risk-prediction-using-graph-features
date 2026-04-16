import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from ..tools.execute import *
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
import sys, os
import pickle
import ast

def processing(configurations, original_directory, path, alpha = None, discretization_type = None):
    # exit(configurations)
    classic_result_path = "reports/" + db_name + "/metrics/classic/metrics_results.txt"
    trainset = pd.read_csv("data/preprocessed/" + db_name + "/preprocessed_data_train.csv", keep_default_na=False, na_values=[""])
    testset = pd.read_csv("data/preprocessed/" + db_name + "/preprocessed_data_test.csv", keep_default_na=False, na_values=[""])
    trainset.drop(columns=['Unnamed: 0'], inplace=True)
    testset.drop(columns=['Unnamed: 0'], inplace=True)

    with open(classic_result_path, "r") as file:
        classic_result = file.read()
    classic_result = ast.literal_eval(classic_result)

    train_new_descriptors = pd.read_csv(train_data_path, keep_default_na=False, na_values=[""])
    train_new_descriptors.drop(columns=['Unnamed: 0'], inplace=True)
    test_new_descriptors = pd.read_csv(test_data_path, keep_default_na=False, na_values=[""])
    test_new_descriptors.drop(columns=['Unnamed: 0'], inplace=True)
    
    final_trainset = pd.concat([trainset, train_new_descriptors], axis=1)
    final_testset = pd.concat([testset, test_new_descriptors], axis=1)

    results_real, results_with_percent = build_predictions(models, final_trainset, final_testset, configurations, target, classic_result)

    if alpha:
        os.makedirs(original_directory+ '/real/' + path, exist_ok=True)
        with open(original_directory+ '/real/' + path + '/metrics_results_' +  args[7] + ".txt",
                  'w') as file:
            file.write(str(results_real))

        os.makedirs(original_directory + '/percent/' + path, exist_ok=True)
        with open(original_directory+ '/percent/' + path + '/metrics_results_' +  args[7] + ".txt",
                  'w') as file:
            file.write(str(results_with_percent))

    else: 
        os.makedirs(original_directory + '/real/' + path, exist_ok=True)
        with open(original_directory + '/real/' + path + '/metrics_results.txt',
                  'w') as file:
            file.write(str(results_real))

        os.makedirs(original_directory + '/percent/' + path, exist_ok=True)
        with open(original_directory + '/percent/' + path + '/metrics_results.txt',
                  'w') as file:
            file.write(str(results_with_percent))

if __name__ == "__main__":
    args = sys.argv[1:]
    target = args[0]
    db_name = args[1]
    train_data_path = args[2]
    test_data_path = args[3]
    discretization_type = args[4]
    graph_type = args[5]
    config_path = args[6]
    
    cost_data = None
    cost_attributes = None
    
    directory = 'reports/'+ db_name + '/metrics/'
    os.makedirs(directory, exist_ok=True )
    
    ############ MODELS #############
    models = {
                'log': LogisticRegression(random_state=16, max_iter=1000), 
                'svm': svm.SVC(kernel='linear'),
                'dtree': DecisionTreeClassifier(), 
                'rf': RandomForestClassifier(), 
                'xgb': XGBClassifier(objective='binary:logistic', max_depth=4, learning_rate=0.1, n_estimators=100, alpha=10),
                'lda': LinearDiscriminantAnalysis(n_components=1),
             # 'mlp': MLPClassifier(random_state=1, max_iter=300)
              }
    
    # exit(models['xgb'])

    with open(config_path, "r") as file:
        configurations = file.read()
    configurations = ast.literal_eval(configurations)

    if  discretization_type == 'None' and graph_type == 'None':
        with open(directory + '/models', 'wb') as file:
            pickle.dump(models, file)

        classic_result = None
        final_trainset = pd.read_csv(train_data_path, keep_default_na=False, na_values=[""])
        final_trainset.drop(columns=['Unnamed: 0'], inplace=True)
        final_testset = pd.read_csv(test_data_path, keep_default_na=False, na_values=[""])
        final_testset.drop(columns=['Unnamed: 0'], inplace=True)

        real_results, _ = build_predictions(models, final_trainset, final_testset, configurations, target,
                                    classic_result)
        directory_ = directory + '/classic'
        os.makedirs(directory_, exist_ok=True)
        with open(directory_ + '/metrics_results.txt', 'w') as file:
            file.write(str(real_results))

    elif discretization_type != 'None' and graph_type != 'None':
        alpha = args[7]
        directory_ = '/' + graph_type.lower() +'/'+ discretization_type.lower()

        processing(configurations, directory, directory_, alpha)

    elif discretization_type == 'None' and graph_type != 'None':
        directory_ = '/' + graph_type.lower()

        processing(configurations, directory, directory_, None)

    # else:
    #     directory_ = '/predictions/na/'

    #     processing(configurations, directory, directory_, None, discretization_type)
































































# cost_attributes = ast.literal_eval(cost_attributes)
    
        
    # if len(cost_attributes) > 0:
    #     with open("outputs/"+db_name+"/preprocessed_sets/cost_data", 'rb') as file:
    #        cost_data = pickle.load(file)  
    # else:
  # elif disc_type == "BASIC":
    #     classic_result_path = "outputs/" + db_name + "/results/predictions/classic/metrics_results"
    #     with open(classic_result_path, "rb") as file:
    #         classic_result = pickle.load(file)

    #     with open(train_data_path, "rb") as f:
    #         final_trainset = pickle.load(f)

    #     with open(test_data_path, "rb") as f:
    #         final_testset = pickle.load(f)

    #     test_index = final_testset.index
    #     if cost_attributes is not None:
    #         cost_data = cost_data.loc[test_index]
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    










