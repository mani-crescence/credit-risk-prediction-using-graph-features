import sys, os, pickle
import networkx as nx
from ....tools.graph import gower_distance
import pandas as pd


def main(train_data, test_data, target, _dir):
    
    max_col = {}
    min_col = {}
    
    for col in train_data.columns:
        max_col[col] = train_data[col].max()
        min_col[col] = train_data[col].min()
    
    graph = nx.Graph()
    
    ### TRAINSET PROCESSING
    train_loans = {}
    for i, row in train_data.iterrows():
        train_loans['tr_u'+str(i)] = {}
        for j, w in row.items():
            train_loans['tr_u'+ str(i)][j] = w
            if j == target:
                w = int(w)
                graph.add_edge('tr_u'+ str(i), target + '_loan_' + str(w), weight=1)
        
    processed_loans = []
    for current_train_loan_index, current_train_loan in train_loans.items():
        processed_loans.append(current_train_loan_index)
        neighbor_loans= dict([(key, val) for key, val in train_loans.items() if key not in processed_loans])

        for train_loan_neighbor_index, train_loan_neighbor in neighbor_loans.items():
            weight = gower_distance(current_train_loan, train_loan_neighbor, train_data.columns, max_col, min_col)
            graph.add_edge(current_train_loan_index, train_loan_neighbor_index, weight=weight)
    
    
    ### TESTSET PROCESSING        
    test_loans = {}        
    for i, row in test_data.iterrows():
        test_loans['ts_u'+str(i)] = {}
        for j, w in row.items():
            test_loans['ts_u'+str(i)][j] = w    
    
    processed_loans = []
    for current_test_loan_index, current_test_loan in test_loans.items():
        processed_loans.append(current_test_loan_index)
        neighbor_loans = dict([(key, val) for key, val in test_loans.items() if key not in processed_loans]) 
        
        for test_loan_neighbor_index, test_loan_neighbor in neighbor_loans.items():
            weight = gower_distance(current_test_loan, test_loan_neighbor, test_data.columns, max_col, min_col)
            graph.add_edge(current_test_loan_index, test_loan_neighbor_index, weight = weight)        
    
    ### TRAINSET AND TESTSET PROCESSING        
    train_loans_for_test = {}
    train_data.drop(columns=[target], inplace=True)   
    
    for i, row in train_data.iterrows():
        train_loans_for_test['tr_u'+str(i)] = {}
        for j, w in row.items():
            train_loans_for_test['tr_u'+ str(i)][j] = w 
                 
    for i, row in test_data.iterrows():
        dict_row = row
        current_loan = {}
        
        for j, w in  dict_row.items():
            current_loan[j] = w
             
        for train_index, train_neighbor in train_loans_for_test.items():
            weight = gower_distance(current_loan, train_neighbor, train_data.columns, max_col, min_col)
            graph.add_edge('ts_u' + str(i), train_index, weight = weight) 
      
           
    graph_data = {"graph": graph, "descriptors" : [target + '_loan_0', target + '_loan_1']}        
    directory = _dir + db_name + '/'  
    os.makedirs(directory, exist_ok=True)
    with open(directory + 'graph_' + graph_type.lower(), 'wb') as file:
        pickle.dump(graph_data, file)    


if __name__ == "__main__":
    args =  sys.argv[1:]   
    db_name = args[0]
    target = args[1]
    graph_type = args[2] 
    train_path = args[3]
    test_path = args[4]
    _dir = args[5]
    
    trainset  = pd.read_csv(train_path)
    testset  = pd.read_csv(test_path)
    testset.drop(columns=[target], inplace=True) 
    
    main(trainset, testset, target, _dir)