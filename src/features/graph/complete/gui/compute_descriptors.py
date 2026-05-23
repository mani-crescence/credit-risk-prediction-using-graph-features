from .....tools.execute import *
import sys
import pickle

def main( db_name, graph_type, train_path, test_path, _dir, _graph_dir):
    
    with open(_graph_dir + db_name + '/complete_graph' ,"rb" ) as f:
                graph_data = pickle.load(f)
                
    trainset = pd.read_csv(train_path, keep_default_na=False, na_values=[""])
    testset = pd.read_csv(test_path, keep_default_na=False, na_values=[""])
    trainset.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')
    testset.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')  

    build_global_pagerank(graph_data["graph"], trainset, testset, db_name, graph_type.lower(), _dir) 
    

if __name__ == "__main__":
    args = sys.argv[1:]
    target = args[0]
    db_name = args[1]
    graph_type = args[2].lower()
    train_path = args[3]
    test_path = args[4]
    _dir = args[5]
    _graph_dir = args[6] 
    
    main(db_name, graph_type, train_path, test_path, _dir, _graph_dir)   