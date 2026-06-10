import pandas as pd
import sys, os
import pickle
from feature_engine.discretisation import DecisionTreeDiscretiser


if __name__ == "__main__":
    args = sys.argv[1:]
    db_name = args[0]
    target = args[1]
    path = args[2]
    dir_= args[3]
    
    directory = dir_ + db_name+ "/"
    os.makedirs(directory, exist_ok=True)
    
    partial_preprocessed_data  = pd.read_feather(path)
    
    # try:
    #     partial_preprocessed_data.drop(columns=['Unnamed: 0'], inplace = True)
    # except:
    #     print("Column 'Unnamed: 0' not existed!") 
        
    numeric_data = partial_preprocessed_data.select_dtypes('float')
    partial_preprocessed_data[target] = partial_preprocessed_data[target].astype("object")
    
    
          
    disc = DecisionTreeDiscretiser(
                cv=3, scoring='neg_mean_squared_error',
                variables=numeric_data.columns.tolist(),
                param_grid={'max_depth': [1, 2, 3]},
                bin_output='bin_number') 
    
    disc.fit(numeric_data, partial_preprocessed_data[target])
    with open(directory + 'tree_discretization_engine' , 'wb') as file:
        pickle.dump(disc, file)
        

    
    
    
    