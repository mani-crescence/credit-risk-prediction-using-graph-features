import sys, os, pickle, ast
import pandas as pd, numpy as np
import networkx as nx
  
          
def compute_personalized_degree(df_sample, X_train, dense_matrix, target):

    sample_numbers = len(df_sample)
    
    print('sample_numbers', sample_numbers)

    degree_pos = np.zeros(len(df_sample), dtype=int)
    degree_neg = np.zeros(len(df_sample), dtype=int)

    for i in range(dense_matrix.shape[0]):  
        all_indices = set(range(sample_numbers))
        exclude_i = all_indices - {i}
        train_set = set(X_train.index)

        # pos
        pos_mask = df_sample[target].values == 1
        pos_indices = set(np.where(pos_mask)[0])
        col_indices_pos = list(train_set & exclude_i & pos_indices)
        current_row_pos = dense_matrix[i, col_indices_pos]
        degree_pos[i] = int(np.sum(~np.isinf(current_row_pos)))

        # neg
        neg_mask = df_sample[target].values == 0
        neg_indices = set(np.where(neg_mask)[0])
        col_indices_neg = list(train_set & exclude_i & neg_indices)
        current_row_neg = dense_matrix[i, col_indices_neg]
        degree_neg[i] = int(np.sum(~np.isinf(current_row_neg)))

        if (i + 1) % 1000 == 0:
            print(i + 1)
            
    return  degree_pos, degree_neg      
  
  
def main(G, train_index, test_index, trainset, df, target, db_name):
    A = nx.adjacency_matrix(G)
    dense_matrix = A.toarray()

    degree_pos, degree_neg = compute_personalized_degree(df, trainset, dense_matrix, target)

    degree_df = pd.DataFrame([degree_pos, degree_neg]).T
    degree_df.columns = ['degree_pos', 'degree_neg']
    
    train = degree_df.loc[train_index]
    test = degree_df.loc[test_index]
    
    directory = _dir + db_name + '/' + graph_type
    os.makedirs(directory, exist_ok=True)
    train.to_feather(directory + '/new_features_train.feather')

    directory = _dir + db_name + '/' + graph_type
    os.makedirs(directory, exist_ok=True)
    test.to_feather(directory + '/new_features_test.feather')
    
   

if __name__ == "__main__":
    args = sys.argv[1:]
    target = args[0]
    db_name = args[1]
    graph_type = args[2].lower()
    train_path = args[3]
    test_path = args[4]
    _dir = args[5]
    _graph_dir = args[6]
    
    trainset = pd.read_feather('data/preprocessed/' + db_name + '/preprocessed_data_train.feather')
    
    testset  = pd.read_feather('data/preprocessed/' + db_name + '/preprocessed_data_test.feather')
    testset.drop(columns=target, inplace=True)
    
    with open(_graph_dir + db_name + "/graph_liu","rb" ) as f:
        graph = pickle.load(f)
    df = pd.concat([trainset, testset])

    main(graph, trainset.index, testset.index, trainset, df, target, db_name)
    
    
    