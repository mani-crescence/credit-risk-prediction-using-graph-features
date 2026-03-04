import ast, json
import pickle
import pandas as pd
from tools.training import *
import sys

def main_plot(db_name, train_descriptors_paths, test_descriptors_paths, target, model_name, discretization_type = None):
    ordinary_path_train = "outputs/"+db_name+"/data/classic/trainset.csv"
    ordinary_path_test = "outputs/" + db_name + "/data/classic/testset.csv"
    ordinary_train = pd.read_csv(ordinary_path_train)
    ordinary_train.set_index('Unnamed: 0', inplace=True)
    ordinary_train.index.name = None
    ordinary_test= pd.read_csv(ordinary_path_test)
    ordinary_test.set_index('Unnamed: 0', inplace=True)
    ordinary_test.index.name = None

    with open('outputs/general_results/results/' + db_name + '/models', 'rb') as file:
        models = pickle.load(file)
    model = models[model_name]

    descriptors = [ordinary_train]
    for path in train_descriptors_paths:
        train_new_descriptors = pd.read_csv(path)
        train_new_descriptors.set_index('Unnamed: 0', inplace=True)
        train_new_descriptors.index.name = None
        descriptors.append(train_new_descriptors)
    trainset= pd.concat(descriptors, axis = 1)

    descriptors = [ordinary_test]
    for path in test_descriptors_paths:
        test_new_descriptors = pd.read_csv(path)
        test_new_descriptors.set_index('Unnamed: 0', inplace=True)
        test_new_descriptors.index.name = None
        descriptors.append(test_new_descriptors)
    testset = pd.concat(descriptors, axis=1)
    trainset_for_shap = trainset.drop(target, axis=1)
    testset_for_shap = testset.drop(target, axis=1)

    model.fit(trainset_for_shap, trainset[target])

    directory = 'outputs/general_results/results/' + db_name + '/shap/'
    os.makedirs(directory, exist_ok=True)
    build_shap_plot(model, model_name, trainset_for_shap, testset_for_shap, directory, discretization_type)

if __name__ == "__main__":
    # print("i'm here")
    args = sys.argv[1:]
    target = args[0]
    db_name = args[1]
    train_descriptors_paths = ast.literal_eval(args[2])
    test_descriptors_paths = ast.literal_eval(args[3])
    model_name = args[4]
    discretization_type = args[5]
    # print(model_name)
    main_plot(db_name, train_descriptors_paths, test_descriptors_paths, target, model_name, discretization_type)