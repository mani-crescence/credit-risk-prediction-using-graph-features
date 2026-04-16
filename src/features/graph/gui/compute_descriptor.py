from tools.execute import *
import sys
import pickle

def main( db_name, graph_type):
    
    with open('graph/'+ db_name + '/complete_graph' ,"rb" ) as f:
                graph_data = pickle.load(f)
                
    trainset = pd.read_csv("data/preprocessed/"+ db_name +"/preprocessed_data_train.csv", keep_default_na=False, na_values=[""])
    testset = pd.read_csv("data/preprocessed/"+ db_name +"/preprocessed_data_test.csv", keep_default_na=False, na_values=[""])
    trainset.drop(columns=['Unnamed: 0'], inplace=True)
    testset.drop(columns=['Unnamed: 0'], inplace=True)  

    build_global_pagerank(graph_data["graph"], trainset, testset, db_name, graph_type.lower()) 
    

if __name__ == "__main__":
    args = sys.argv[1:]
    target = args[0]
    db_name = args[1]
    graph_type = args[2].lower() 
    
    main(db_name, graph_type)   