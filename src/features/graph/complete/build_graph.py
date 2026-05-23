import os, ast, pickle, sys
import networkx as nx

def main(db_name, _dir):
    graph = nx.Graph()
          
    directories = [
    f'{_dir}{db_name}/related/train/',
    f'{_dir}{db_name}/related/test/',
    f'{_dir}{db_name}/related/both/',
    f'{_dir}{db_name}/subsets/train/',
    f'{_dir}{db_name}/subsets/test/'
]

    def get_node_type(node):
        if 'tr' in node:
            return 'train'
        if 'ts' in node:
            return 'test'
        return None

    for directory in directories:
        for item in os.listdir(directory):
            path = os.path.join(directory, item)

            with open(path, "r") as file:
                edges = ast.literal_eval(file.read())

            for n1, n2, weight in edges['edges']:
                # Add nodes with type
                for node in (n1, n2):
                    node_type = get_node_type(node)
                    if node_type:
                        graph.add_node(node, type=node_type)

                # Add edge
                graph.add_edge(n1, n2, weight=weight)        
            
    print( "length nodes =>", len(graph.nodes), "edges length =>", len(graph.edges))  
   
    mst = nx.minimum_spanning_tree(graph, algorithm="prim")
    
    print("minimun spanning tree computation terminated ")    
    
    descriptors_attributes = ["deg0", "deg1"]     
    graph_data = {"graph": graph, "mst": mst, "descriptors": descriptors_attributes}  
    
    directory= _dir +  db_name + '/'  
    os.makedirs(directory, exist_ok=True)
    with open(directory + 'complete_graph', 'wb') as file:
            pickle.dump(graph_data, file)
    

if __name__ == "__main__":
    args = sys.argv[1:]
    db_name = args[0]
    directory_ = args[1]
    
    # print(directory_)
    # exit()

    main(db_name, directory_)    