import pandas as pd
import sys, os
import pickle
# from feature_engine.discretisation import DecisionTreeDiscretiser
from optbinning import MDLP


if __name__ == "__main__":
    args = sys.argv[1:]
    db_name = args[0]
    target = args[1]
    path = args[2]
    dir_= args[3]
    
    directory = dir_ + db_name+ "/"
    os.makedirs(directory, exist_ok=True)
    
    partial_preprocessed_data  = pd.read_feather(path)
    
    numeric_data = partial_preprocessed_data.select_dtypes('float')
    partial_preprocessed_data[target] = partial_preprocessed_data[target].astype("int")
    
    for col in numeric_data.columns:
        mdlp = MDLP(min_samples_split=2, min_samples_leaf=2) 
        mdlp.fit(numeric_data[col], 
                partial_preprocessed_data[target])
    
        with open(directory + 'mdlp_engine_' + col, 'wb') as file:
            pickle.dump(mdlp, file)
    

    
    
    
    