import pandas as pd
import sys, ast
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from preprocessing import preprocess_main
#
def sampling(file, db_name, data_size, target, unuseful_attributes, attributes_for_manual_encoding = None, values_for_manual_encoding = None):
    data = pd.read_csv(file, low_memory=False)
    data[target] = data[target].astype(int)
    data[target] = data[target].astype(object)
    sampling_rate = data_size / len(data)
    sample_1 = data[data[target] == 1]
    sample_0 = data[data[target] == 0]

    opt_sample = ""
    opt_f1 = 1

    for i in range(30):
        sub_sample_0 = sample_0.sample(round(sampling_rate * len(sample_0)))
        sub_sample_1 = sample_1.sample(round(sampling_rate * len(sample_1)))
        sample = pd.concat([sub_sample_1, sub_sample_0], axis=0)
        preprocessed_data = preprocess_main(sample, unuseful_attributes, target, db_name, attributes_for_manual_encoding, values_for_manual_encoding, None)
        trainset, testset = train_test_split(preprocessed_data, test_size=0.2, random_state=42, shuffle=True)
        # models = {'log': LogisticRegression(random_state=16, max_iter=1000), 'svm': svm.SVC(kernel='linear'),
        #           'dtree': DecisionTreeClassifier(), 'rf': RandomForestClassifier(), 'xgb': XGBClassifier(),
        #           'lda': LinearDiscriminantAnalysis(n_components=1, solver='lsqr')}
        model = DecisionTreeClassifier()

        # for key, model in models.items():
        X_train = trainset.drop(target, axis=1)
        y_train = trainset[target]
        X_test = testset.drop(target, axis=1)
        y_test = testset[target]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        if f1 < opt_f1 :
            opt_f1 = f1
            opt_sample = sample
            print(f"accuracy ==> {accuracy}, f1 ===> {f1}")

    opt_sample.to_csv(db_name+"_sample.csv", index=False)

if __name__ == "__main__":
    args = sys.argv[1:]
    db_name = args[0]
    path = args[1]
    target = args[2]
    unuseful_attributes = args[3]
    # print(args)
    # exit()
    # cost_attributes = args[4]
    unuseful_attributes = ast.literal_eval(unuseful_attributes)
    # cost_attributes = ast.literal_eval(cost_attributes)

    if len(args) > 5:
        attributes_for_manual_encoding = args[4]
        values_for_manual_encoding = args[5]
        attributes_for_manual_encoding = ast.literal_eval(attributes_for_manual_encoding)
        values_for_manual_encoding = ast.literal_eval(values_for_manual_encoding)
        sampling(path, db_name, 10000, target, unuseful_attributes, attributes_for_manual_encoding, values_for_manual_encoding )
    else:
        sampling(path, db_name, 10000, target, unuseful_attributes, None,  None)




