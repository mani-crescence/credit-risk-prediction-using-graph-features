import os, ast, pickle, sys
import networkx as nx

def main(db_name, graph_type):
    graph = nx.Graph()
          
    directories = [
    f'graph/{db_name}/related/train/',
    f'graph/{db_name}/related/test/',
    f'graph/{db_name}/related/mix/',
    f'graph/{db_name}/subsets/train/',
    f'graph/{db_name}/subsets/test/'
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
    graph_data = {"graph": mst, "descriptors": descriptors_attributes}  
    
    directory='graph/'+ db_name + '/'  
    os.makedirs(directory, exist_ok=True)
    with open(directory + 'complete_graph_'+ graph_type, 'wb') as file:
            pickle.dump(graph_data, file)
    
    # return mst    
    # print(graph.edges(data=True))
    # print( "length nodes =>", len(graph.nodes), "edges length =>", len(graph.edges))
    # exit()
    # return graph    

if __name__ == "__main__":
    args = sys.argv[1:]
    db_name = args[0]
    graph_type = args[1]

    main(db_name, graph_type)    