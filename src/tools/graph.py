from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
import math
from bokeh.io import output_file, show
from bokeh.plotting import figure, from_networkx
from bokeh.models import (BoxZoomTool, Circle, HoverTool,
                          MultiLine, Plot, Range1d, ResetTool)
from bokeh.palettes import Spectral4



def graph_bipartite_modality(graph, data, new_loan = None, discretization_type = None):
    if graph is None:
        graph = nx.Graph()

        for i , row in data.iterrows():
            graph.add_node('tr_u'+ str(i), type='loan', bipartite=0)
            
            for j, w in row.items():
                if not graph.has_node(str(j) + '_' + str(w) + '_' + discretization_type + '_bip'):
                    graph.add_node(str(j)+ '_'+ str(w)+'_'+discretization_type+'_bip', type='attribute', bipartite=1)
                    
                graph.add_edge('tr_u'+ str(i), str(j)+ '_'+ str(w)+'_'+discretization_type+'_bip')

        # bipartite_0_nodes = [node for node, attr in graph.nodes(data=True) if attr.get('bipartite') == 0]
        # node_colors = ['red' if node in bipartite_0_nodes else 'skyblue' for node in graph.nodes]
        # top = nx.bipartite.sets(graph)[0]
        # pos = nx.bipartite_layout(graph, top)
        
        # plt.figure(figsize=(12, 10))
        # nx.draw(graph, pos=pos, with_labels=True, node_color = node_colors, node_size=200, font_size=5, font_weight='bold', width=2)
        # plt.savefig("bip.png")
        # plt.close()
        # print(graph.edges)
        return graph
        
    else:
        edges = []
        nodes = [] 
        new_node = 'ts_'+str(new_loan['Index'])
        
        graph.add_node(new_node, type = 'loan')
        nodes.append(new_node)
        
        del new_loan['Index']
        
        for j, w in new_loan.items():
            if graph.has_node(str(j)+ '_' + str(w) + '_' + discretization_type + '_bip') is False:
                graph.add_node(str(j)+ '_'+ str(w)+'_'+discretization_type+'_bip', type = 'attribute')
            graph.add_edge(new_node, str(j)+ '_'+ str(w)+'_'+discretization_type+'_bip')
            # print(('nl', str(j)+ '_'+ str(w)+'_'+discretization_type+'_bip'))
            edges.append((new_node, str(j)+ '_'+ str(w)+'_'+discretization_type+'_bip'))
        
        return graph     
        
def graph_modality(graph, data, new_loan = None, discretization_type = None):
    if graph is None:
        graph = nx.Graph()
        for i, row in data.iterrows():
            for j , (k,w)  in enumerate(row.items()):
                if not graph.has_node(str(k) + '_' + str(w) + '_' + discretization_type + '_mod'):
                    graph.add_node(str(k)+ '_'+ str(w)+'_'+discretization_type+'_mod', type = 'attribute')
                for (k2 , y) in list(row.items())[j+1:]:
                    if not graph.has_node(str(k2) + '_' + str(y) + '_' + discretization_type + '_mod'):
                        graph.add_node(str(k2)+ '_'+ str(y)+'_'+discretization_type+'_mod', type = 'attribute')
                    if not graph.has_edge(str(k) + '_' + str(w) + '_' + discretization_type + '_mod',
                                          str(k2) + '_' + str(y) + '_' + discretization_type + '_mod'):
                        graph.add_edge(str(k)+ '_'+ str(w)+'_'+discretization_type+'_mod', str(k2)+ '_'+ str(y)+'_'+discretization_type+'_mod', weight=1)
                    else:
                        graph[str(k) + '_' + str(w) + '_' + discretization_type + '_mod'][str(k2) + '_' + str(y) + '_' + discretization_type + '_mod']['weight'] += 1
                        # print("already existed")

        # # Draw the graph with node and edge attributes
        # print_graph(graph)
        # Print the graph
        # pos = nx.spring_layout(graph)
        # edge_labels = nx.get_edge_attributes(graph, 'weight')
        # nx.draw(graph, pos, with_labels=True, node_size=200,font_size=8, font_weight='bold', width=0.5)
        # nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
        # plt.savefig("mod.png")
        # plt.close()
        
        # plot = figure(title="Graph of Modalities",  sizing_mode='stretch_both', x_range=(-2, 2), y_range=(-2, 2), tools="", toolbar_location=None)
        # G = from_networkx(graph, nx.spring_layout, scale=2, center=(0, 0))
        # G.node_renderer.glyph = Circle(radius=7.5, fill_color=Spectral4[0])
        # plot.renderers.append(G)
        # show(plot)
        
        # output_file("modality_graph.html")
        
        # exit(graph.edges)  
        # weights = nx.get_edge_attributes(graph, "weight")
        # for edge, w in weights.items():
        #     print(f"Arête: {edge}  | Poids: {w}")
      
        return graph
    
    else:
        
        for j , (k,w)  in enumerate(new_loan.items()):
            if graph.has_node(str(k) + '_' + str(w) + '_' + discretization_type + '_mod') is False:
               graph.add_node(str(k)+ '_'+ str(w)+'_'+discretization_type+'_mod', type = 'attribute')
            
            for (k2 , y) in list(new_loan.items())[j+1:]:
                if graph.has_node(str(k2) + '_' + str(y) + '_' + discretization_type + '_mod') is False:
                    graph.add_node(str(k2)+ '_'+ str(y)+'_'+discretization_type+'_mod', type = 'attribute')
                    
                if graph.has_edge(str(k) + '_' + str(w) + '_' + discretization_type + '_mod', str(k2) + '_' + str(y) + '_' + discretization_type + '_mod') is False :
                    graph.add_edge(str(k)+ '_'+ str(w)+'_'+discretization_type+'_mod', str(k2)+ '_'+ str(y)+'_'+discretization_type+'_mod', weight=1)
                
                else:
                    graph[str(k) + '_' + str(w) + '_' + discretization_type + '_mod'][str(k2) + '_' + str(y) + '_' + discretization_type + '_mod']['weight'] += 1
        return graph

def graph_loans(graph, data, target, new_loan = None):
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
                # graph.add_edge(loan, loan_neighbor,
                #                weight=round(np.linalg.norm(np.array(value) - np.array(value_neighbor)), 2))
        
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

def delete_element(graph, borrower):
    
    for j , (k,w)  in enumerate(borrower.items()):
                
        for (k2 , y) in list(borrower.items())[j+1:]:
            
            if graph.has_node(str(k) + '_' + str(w)) is False:
                graph.remove_node(str(k)+ '_'+ str(w))
            graph.remove_node(str(k2)+ '_'+ str(y))
                
def pagerank_global(graph):
    return nx.pagerank(graph, alpha = 0.85)

def pagerank_personalized(graph, alpha, personalized_nodes, weight = None, nodes = None):
   
    personalized_nodes_length = len(personalized_nodes)
    personalization = {node : 0 for node in graph.nodes}
    
    for i in personalized_nodes:
        personalization[i] =  1 / personalized_nodes_length


    attribute_pagerank = nx.pagerank(graph, alpha = alpha , personalization = personalization, weight = None)
    
    new_descriptors = {key : value for key, value in attribute_pagerank.items() if key in nodes}

    pg_max = max(new_descriptors.values())
    
    for key in new_descriptors:
        if pg_max != 0:
            new_descriptors[key] /= pg_max
     
    return new_descriptors

def print_graph(graph):
    for u, v, data in graph.edges(data=True):
        if len(data) != 0:
            weight = data['weight']
            print(f"Edge: ({u}, {v}), Weight: {weight}")
        else:
            print(f"Edge: ({u}, {v})")
                
def gower_distance(u, v, attributes, max = None, min = None):
    sum = 0
    attributes_length = len(u)
    for i in attributes:
        if type(u[i]) is str:
            
            if u[i] == v[i]:
                sum += 0
            else:
                sum += 1
        else:
            if (max[i] - min[i]) == 0:
                sum += abs(u[i] - v[i])
            else:
                sum += abs(u[i] - v[i]) / (max[i] - min[i])

    return sum / attributes_length

def euclidian_distance(u, v, attributes):
    sum = 0
    for i in attributes:
        sum += (u[i] - v[i])**2

    return  math.sqrt(sum)

def complete_graph(trainset, testset, target): 
    graph = nx.Graph()
    
    col_max = trainset.max()
    col_min = trainset.min()
    cols = trainset.columns
    
    train_dicts = {i: row.to_dict() for i, row in trainset.iterrows()}
    test_dicts  = {j: row.to_dict() for j, row in testset.iterrows()}
    
    for i in  trainset.index:
        graph.add_node('tr_u' + str(i), type='train')
        
    for i, loan1 in train_dicts.items():
        for j, loan2 in train_dicts.items():
            graph.add_edge(
                'tr_u' + str(i), 
                'tr_u' + str(j), 
                weight=gower_distance(loan1, loan2, cols, col_max, col_min))
                    
    
    cols = cols.drop(target)
                
    for i in  testset.index:
        graph.add_node('ts_u' + str(i), type='test')
        
    
    for i, loan1 in train_dicts.items():
        for j, loan2 in test_dicts.items():
            graph.add_edge(
                'tr_u' + str(i), 
                'ts_u' + str(j), 
                weight=gower_distance(loan1, loan2, cols, col_max, col_min))
                
    mst = nx.minimum_spanning_tree(graph, algorithm="prim")            
                            
    return mst           