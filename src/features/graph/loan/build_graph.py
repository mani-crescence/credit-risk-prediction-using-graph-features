import sys
import networkx as nx
from ....tools.graph import gower_distance
import pandas as pd


def main(graph, data, target, new_loan = None):
    loans = {}
    max_col = {}
    min_col = {}
    
    numeric_attributes = list(data.select_dtypes(float).columns) + list(data.select_dtypes(int).columns)
    
    for col in numeric_attributes:
        max_col[col] = data[col].max()
        min_col[col] = data[col].min()
        
    if graph is None:
        graph = nx.Graph()
        for i, row in data.iterrows():
            loans['l'+str(i)] = {}
            for j, w in row.items():
                loans['l'+str(i)][j] = w
                if j == target:
                    w = int(w)
                    graph.add_edge('l'+ str(i), target+ '_loan_' + str(w), weight=1)
            
        processed_loans = []
        for loan, value_loan in loans.items():
            processed_loans.append(loan)
            loans_neighbor = dict([(key, val) for key, val in loans.items() if key not in processed_loans])
    
            for loan_neighbor, value_loan_neighbor in loans_neighbor.items():
                weight = gower_distance(value_loan, value_loan_neighbor, data.columns, max_col, min_col)
                graph.add_edge(loan, loan_neighbor, weight=weight)
        
    else :
        
        new_loan_copy = new_loan.copy()
        index = new_loan['Index']
        del new_loan['Index']
        new_loan_values = {}
        for j, w in new_loan.items():
            new_loan_values[j] = w
            if j == target:
                graph.add_edge('l'+ str(index), target+ '_loan_' + str(w), weight=1)
        for i, row in data.iterrows():
            loans['l'+str(i)] = {}
            for j, w in row.items():
                loans['l'+str(i)][j] = w

        for loan, value in loans.items():
            weight = gower_distance(value, new_loan_values, data.columns, max_col, min_col)
            graph.add_edge(loan, 'l'+str(new_loan_copy['Index']), weight = weight)


    return graph

if __name__ == "__main__":
    args =  sys.argv[1:]   
    db_name = args[0]
    target = args[1]
    graph_type = args[2] 
    discretization_type = args[3].lower()
    
         
    data_file ='data/discretized/'+ db_name +'/discretized_train_data_'+ discretization_type + '.csv'
    trainset  = pd.read_csv(data_file, dtype='object', keep_default_na=False, na_values=[""])
    
    trainset.drop(columns=['Unnamed: 0'], inplace=True)
    
    main(None, trainset, None, discretization_type)