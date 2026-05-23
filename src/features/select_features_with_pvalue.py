import statsmodels.api as sm
import numpy as np
import pandas as pd, sys, os

def backwardElimination(x, Y, sl, columns):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(Y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
                    columns = np.delete(columns, j)
                    
    regressor_OLS.summary()
    return x, columns

def stepwise_selection(X, y, initial_list=[], threshold_in=0.01,
                       threshold_out = 0.05, verbose=True
                       ):
    included = list(initial_list)
    while True: 
        
        changed = False
        
        ### FORMWARD STEP 
        excluded = list(set(X.columns) - set(included))
        new_pval = pd.Series(index=excluded)
        
        ### compute the p-value of each feature seperately
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included + [new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
          
        best_pval = new_pval.min()
        
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed = True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))  
                # pass
                
        
        ### BACKWARD STEP
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        
        ### for each feature test if it  can be removed
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max()
        if worst_pval > threshold_out:
            changed = True
            worst_feature = pvalues.argmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
                # pass
            
        if not changed:
            break
        
        
    return included                     



if __name__ == "__main__" :
    args = sys.argv[1:]
    target = args[0]
    db_name = args[1]
    train_path = args[2]
    save_dir = args[3]
    
    final_trainset = pd.read_csv(train_path, keep_default_na=False, na_values=[""])
    final_trainset.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')

    selected_attributes = stepwise_selection(final_trainset.drop(columns=[target]), final_trainset[target])
    
    configurations = {}
    configurations['CLASSIC_WITH_STEPWISE'] = selected_attributes + [target]
    os.makedirs(save_dir, exist_ok=True)
    with open(save_dir + '/selected_features.txt', 'w') as f:
            f.write(str(configurations))
    
    
