import ast
import sys
import os
import pandas as pd



def  build_configurations(ordinary_descriptors, target, db_name, new_descriptors = None, graph_type = None, disc_type = None,  target_values = None):
    
    configurations = {}
    targets = []
    
    directory='outputs/'+db_name+'/configurations'
    os.makedirs(directory, exist_ok=True)
    
    if graph_type is not None and disc_type is not None:
        for t in target_values:
            targets.append(str(target)+'_'+str(t)+'_'+str(disc_type).lower()+'_'+graph_type.lower())

        configurations[graph_type + '_GX_'+disc_type+'_ORD'] = list(set(new_descriptors) - set(targets)) + ordinary_descriptors +  [target]
        configurations[graph_type + '_GY_'+disc_type+'_ORD'] =  targets + ordinary_descriptors + [target]
        configurations[graph_type + '_GXY_' + disc_type + '_ORD'] = new_descriptors + ordinary_descriptors + [target]
        
        with open(directory + '/configuration_'+graph_type.lower() + '_'+ disc_type.lower() + '.txt', 'w') as f:
            f.write(str(configurations))
      
    elif  graph_type is None and disc_type is None:
        configurations['CLASSIC'] = ordinary_descriptors + [target]
        with open(directory + '/configuration_classic.txt', 'w') as f:
            f.write(str(configurations))
       
    elif  graph_type is not None and disc_type is None:
        for t in target_values:
            targets.append(str(target)+'_'+str(t))

        # configurations[graph_type + '_GX_ORD'] = list(set(new_descriptors) - set(targets)) + ordinary_descriptors +  [target]
        # configurations[graph_type + '_GY_ORD'] =  targets + ordinary_descriptors + [target]
        configurations[graph_type + '_GXY_ORD'] = new_descriptors + ordinary_descriptors + [target]
        
        with open(directory + '/configuration_' + graph_type.lower() + '.txt', 'w') as f:
            f.write(str(configurations))
        
    else:
        configurations[disc_type] = new_descriptors + [target]
        configurations[disc_type+"_ORD"] = ordinary_descriptors + new_descriptors + [target]

        with open(directory + '/configuration_' + disc_type.lower() + '.txt', 'w') as f:
            f.write(str(configurations))
        
                
if __name__ == '__main__':
    args = sys.argv[1:]
    target = args[0]
    db_name = args[1]
    disc_type = args[2]
    graph_type = args[3]

    preprocessed_data = pd.read_csv("outputs/"+db_name+"/data/classic/trainset.csv", index_col=0)
    train_ = pd.read_csv("outputs/"+db_name+"/preprocessed_sets/partial_preprocessed_data.csv", index_col=0)
    target_values = train_[target].unique()
    
    ordinary_descriptors= preprocessed_data.columns.tolist()
    ordinary_descriptors.remove(target)
    
    if disc_type == 'None':
        disc_type = ast.literal_eval(disc_type)
        
    if graph_type == 'None':
        graph_type = ast.literal_eval(graph_type)
        
    if graph_type is not None and disc_type is not None:
        train_new_descriptors = pd.read_csv("outputs/"+db_name+"/new_descriptors/"+graph_type.lower() +"/train/new_descriptors_data_"+ disc_type.lower() + '_' + graph_type.lower() +"_0.1.csv", index_col=0)
        new_descriptors = list(train_new_descriptors.columns)
        build_configurations(ordinary_descriptors, target, db_name, new_descriptors, graph_type, disc_type, target_values)
        
    elif graph_type is None and disc_type is None:
        build_configurations(ordinary_descriptors, target, db_name)
        
    elif graph_type is not None and disc_type is None:
         train_new_descriptors = pd.read_csv("outputs/"+db_name+"/new_descriptors/"+graph_type.lower() +"/train/new_descriptors_data_"+ graph_type.lower() +"_0.1.csv", index_col=0)
         new_descriptors = list(train_new_descriptors.columns)
         build_configurations(ordinary_descriptors, target, db_name, new_descriptors, graph_type, None, target_values)
    else:
        train_new_descriptors = pd.read_csv("outputs/"+db_name+"/data/discretized/new_discretized_train_"+disc_type.lower()+".csv", index_col=0)
        new_descriptors = list(train_new_descriptors.columns)
        build_configurations(ordinary_descriptors, target, db_name, new_descriptors, None, disc_type)

























# def build_configurations(ordinary_descriptors, ptype, target, db_name, new_descriptors=None, target_values=None):
#     configurations = {}
#     target_MOD = []
#     target_BIP = []
#
#     if ptype == 'UNS_' or ptype == 'SUP_':
#         configurations[ptype + "ORD"] = ordinary_descriptors + new_descriptors + [target]
#         configurations[ptype] = new_descriptors + [target]
#     elif ptype == "CLASSIC":
#         configurations['CLASSIC'] = ordinary_descriptors + [target]
#
#     elif ptype == "LOAN":
#         targets = []
#         for t in target_values:
#             targets.append(str(target) + '_' + str(t))
#
#         configurations[ptype + '_GXY_ORD'] = new_descriptors + ordinary_descriptors + [target]
#         configurations[ptype + '_GX_ORD'] = list(set(new_descriptors) - set(targets)) + ordinary_descriptors + [target]
#         configurations[ptype + '_GY_ORD'] = targets + ordinary_descriptors + [target]
#     else:
#         data_from_bip = pd.read_csv(
#             "outputs/" + db_name + "/new_descriptors/BIP/train/new_descriptors_data_" + ptype + "_BIP_0.1.csv",
#             index_col=0)
#         data_from_mod = pd.read_csv(
#             "outputs/" + db_name + "/new_descriptors/MOD/train/new_descriptors_data_" + ptype + '_' + "MOD_0.1.csv",
#             index_col=0)
#
#         for t in target_values:
#             target_MOD.append(str(target) + '_' + str(t) + '_' + str(ptype) + '_mod')
#             target_BIP.append(str(target) + '_' + str(t) + '_' + str(ptype) + '_bip')
#
#         attributes_from_bipartite_graph = list(data_from_bip.columns)
#         attributes_from_modality_graph = list(data_from_mod.columns)
#
#         configurations[
#             'BAM_GXY_' + ptype + '_ORD'] = attributes_from_bipartite_graph + attributes_from_modality_graph + ordinary_descriptors + [
#             target]
#         configurations['BAM_GX_' + ptype + '_ORD'] = list(
#             set(configurations['BAM_GXY_' + ptype + '_ORD']) - set(target_BIP) - set(target_MOD))
#         configurations['BAM_GY_' + ptype + '_ORD'] = target_BIP + target_MOD + ordinary_descriptors + [target]
#         configurations['BIP_GXY_' + ptype + '_ORD'] = attributes_from_bipartite_graph + ordinary_descriptors + [target]
#         configurations['BIP_GX_' + ptype + '_ORD'] = list(
#             set(attributes_from_bipartite_graph) - set(target_BIP)) + ordinary_descriptors + [target]
#         configurations['BIP_GY_' + ptype + '_ORD'] = target_BIP + ordinary_descriptors + [target]
#         configurations['MOD_GXY_' + ptype + '_ORD'] = attributes_from_modality_graph + ordinary_descriptors + [target]
#         configurations['MOD_GX_' + ptype + '_ORD'] = list(
#             set(attributes_from_modality_graph) - set(target_MOD)) + ordinary_descriptors + [target]
#         configurations['MOD_GY_' + ptype + '_ORD'] = target_MOD + ordinary_descriptors + [target]
#
#     directory = 'outputs/' + db_name + '/configurations'
#
#     os.makedirs(directory, exist_ok=True)
#     with open(directory + '/configuration_' + ptype + '.txt', 'w') as file:
#         file.write(str(configurations))
#
#
# if __name__ == '__main__':
#     args = sys.argv[1:]
#     target = args[0]
#     db_name = args[1]
#     ptype = args[2]
#
#     classic_train_data = pd.read_csv("outputs/" + db_name + "/data/classic/trainset.csv", index_col=0)
#     train_ = pd.read_csv("outputs/" + db_name + "/preprocessed_sets/partial_preprocessed_data.csv", index_col=0,
#                          keep_default_na=False)
#     target_values = train_[target].unique()
#
#     ordinary_descriptors = classic_train_data.columns.tolist()
#     ordinary_descriptors.remove(target)
#
#     if ptype == "UNS_" or ptype == "SUP_":
#         train_new_descriptors = pd.read_csv(
#             "outputs/" + db_name + "/data/discretized/new_discretized_train_" + ptype + ".csv", index_col=0)
#         new_descriptors = list(train_new_descriptors.columns)
#         build_configurations(ordinary_descriptors, ptype, target, db_name, new_descriptors)
#
#     elif ptype == "CLASSIC":
#         build_configurations(ordinary_descriptors, ptype, target, db_name)
#
#     elif ptype == "LOAN":
#         train_new_descriptors = pd.read_csv(
#             "outputs/" + db_name + "/new_descriptors/" + ptype + "/train/new_descriptors_data_" + ptype + "_0.1.csv",
#             index_col=0)
#         new_descriptors = list(train_new_descriptors.columns)
#         build_configurations(ordinary_descriptors, ptype, target, db_name, new_descriptors, target_values)
#
#     else:
#         build_configurations(ordinary_descriptors, ptype, target, db_name, None, target_values)
#


    
     
    