import ast
import sys
import os
import pandas as pd



def  build_configurations(ordinary_descriptors, target, db_name, save_dir, new_descriptors = None, graph_type = None,
                          disc_type = None,  target_values = None):
    
    configurations = {}
    targets = []
    
    directory = save_dir + '/' + db_name + '/'
    os.makedirs(directory, exist_ok=True)
    
    if graph_type is not None and disc_type is not None:
        for t in target_values:
            targets.append(str(target)+'_'+str(t)+'_'+str(disc_type).lower()+'_'+graph_type.lower())

        configurations[graph_type + '_GX_'+disc_type+'_ORD'] = new_descriptors + ordinary_descriptors +  [target]
        configurations[graph_type + '_GX_'+disc_type+'_ORD'].remove("gy") 
        configurations[graph_type + '_GY_'+disc_type+'_ORD'] =  ["gy"] + ordinary_descriptors + [target]
        configurations[graph_type + '_GXY_' + disc_type + '_ORD'] = new_descriptors + ordinary_descriptors + [target]
        
        with open(directory + '/configuration_'+graph_type.lower() + '_'+ disc_type.lower() + '.txt', 'w') as f:
            f.write(str(configurations))
    
    elif  graph_type is not None and disc_type is None:
        configurations[graph_type] = new_descriptors + ordinary_descriptors + [target]  
            
        with open(directory + '/configuration_' + graph_type.lower() + '.txt', 'w') as f:
            f.write(str(configurations))        
            
      
    else:
        configurations['CLASSIC'] = ordinary_descriptors + [target]
        with open(directory + '/configuration_classic.txt', 'w') as f:
            f.write(str(configurations))
       
 
        
                
if __name__ == '__main__':
    args = sys.argv[1:]
    target = args[0]
    db_name = args[1]
    save_dir = args[2]
    disc_type = args[3]
    graph_type = args[4]
    classic_train_path = args[5]
    new_descriptor_train_path = args[6]

    classic_data = pd.read_csv(classic_train_path, index_col = 0, keep_default_na = False, na_values = [""])
    target_values = classic_data[target].unique()
    
    ordinary_descriptors = classic_data.columns.tolist()
    ordinary_descriptors.remove(target)
    
    if disc_type == 'None':
        disc_type = ast.literal_eval(disc_type)
        
    if graph_type == 'None':
        graph_type = ast.literal_eval(graph_type)
        
    train_new_descriptors = pd.read_csv(new_descriptor_train_path, index_col=0, keep_default_na=False, na_values=[""])
    
    new_descriptors = list(train_new_descriptors.columns)
    build_configurations(ordinary_descriptors, target, db_name, save_dir, new_descriptors, graph_type, disc_type, target_values)
    
    
    
    
    
    
    
    