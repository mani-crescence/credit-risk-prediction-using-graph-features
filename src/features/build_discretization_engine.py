import pandas as pd
from ..tools.preprocessing import kmeans_discretization_engine
import sys, os
import pickle

from sklearn.tree import DecisionTreeClassifier 
from feature_engine.discretisation import DecisionTreeDiscretiser


if __name__ == "__main__":
    args = sys.argv[1:]
    target = args[2]
    db_name = args[0]
    discretization_type = args[1]
    
    directory = "engine/discretization/" + db_name+ "/"
    os.makedirs(directory, exist_ok=True)
    
    partial_preprocessed_data  = pd.read_csv("data/preprocessed/" + db_name + "/partial_preprocessed_data_train.csv", keep_default_na=False)
    partial_preprocessed_data.drop(columns=['Unnamed: 0'], inplace = True)
    numeric_data = partial_preprocessed_data.select_dtypes('float')
    partial_preprocessed_data[target] = partial_preprocessed_data[target].astype("object")
    
    # exit(numeric_data.columns)
    
    if discretization_type == "UNS":
        for col in numeric_data.columns:
            att = numeric_data[col].values
            att = att.reshape(-1, 1)
            discretization_engine = kmeans_discretization_engine(att)
            
            with open(directory + 'kmeans_engine_'+ col , 'wb') as file:
                pickle.dump(discretization_engine, file)
    else:      
        disc = DecisionTreeDiscretiser(
                    cv=3, scoring='neg_mean_squared_error',
                    variables=numeric_data.columns.tolist(),
                    param_grid={'max_depth': [1, 2, 3]},
                    bin_output='bin_number')  
            
        # disc = DecisionTreeClassifier(
        #     max_leaf_nodes=3,
        #     random_state=42
        # )    
        
        disc.fit(numeric_data, partial_preprocessed_data[target])
        with open(directory + 'tree_discretization_engine' , 'wb') as file:
            pickle.dump(disc, file)
        

    
    
    
    