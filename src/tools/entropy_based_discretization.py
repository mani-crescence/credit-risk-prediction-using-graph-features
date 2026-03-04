import pandas as pd
import numpy as np
import math
from sklearn.tree import DecisionTreeRegressor



def compute_binary_entropy(data):
    counts = data.value_counts()
    n = data.count()
    entropy = 0
    
    for i, v in counts.items():
        prob = v/n
        entropy += prob * np.log2(prob)
        
    if entropy == 0:
        return 0
        
    entropy = -1 * entropy   
    
    return entropy


def compute_info_gain(sub_data, att_name, target, splits, min_number_per_bin = 1):
    n = len(sub_data)
    global_entropy = compute_binary_entropy(sub_data[target])

    if global_entropy == 0:
        # print(sub_data['st'].value_counts())
        # print('global entropy equal to zero')
        # # print(splits)
        # print('---------end---------\n')
        return splits

    else:
        m = sub_data[target].value_counts().count()
        first_index = sub_data.index[0]
        last_index = sub_data.index[-1]

        max_split = {
            'info': -1,
            'index': 0,
            'gain': 0
        }
        entropy1 = 0
        entropy2 = 0

        for i in range(first_index+(min_number_per_bin-1), last_index-(min_number_per_bin-1)):
            sub1 = sub_data[target].loc[first_index:i]
            sub2 = sub_data[target].loc[i+1:last_index]
            n1 = len(sub1)
            n2 = len(sub2)

            entropy1 = compute_binary_entropy(sub1)
            entropy2 = compute_binary_entropy(sub2)

            info = ((n1/n)*entropy1 + (n2/n)*entropy2)

            gain =  global_entropy - info

            if gain > max_split["gain"]:
                max_split['info'] = info
                max_split['index'] = i
                max_split['gain'] = gain

        if max_split["index"] == 0:
            # print(sub_data['st'].value_counts())
            # print('index 0 - max split index ===>', max_split["index"])
            # print(splits)
            # print('---------end---------\n')
            return splits
        splits.append(max_split['index'])
        sub1_best = sub_data.loc[first_index: max_split['index']]
        sub2_best = sub_data.loc[max_split['index']+1:last_index]

        # print('start index', first_index, ':', max_split['index'], '\t')
        # print('end index', max_split['index']+1,':', n-1, '\n')
        # print('-----s1', sub1_best, '\n')
        # print('-----s2', sub2_best, '\n')
        # print('-------max splits index',max_split['index'], '\n \n \n' )

        m1 = sub1_best.value_counts().count()
        m2 = sub2_best.value_counts().count()
        d = compute_stopping_criterion(global_entropy, entropy1, entropy2, m, m1, m2, n)

        if max_split['gain'] > d :
            # print(sub_data['st'].value_counts())
            # print('max split', max_split['gain'], d )
            # # print(splits)
            # print('---------end---------\n')
            # print(max_split['gain'], d)
            return splits
        else:
            return list(set(compute_info_gain(sub1_best, att_name, target, splits, min_number_per_bin)) | set(compute_info_gain(sub2_best, att_name, target, splits, min_number_per_bin)))


def compute_stopping_criterion(global_entropy, ent1, ent2, m, m1, m2, n):
    d = (math.log(n-1, 2) + math.log((pow(3, 2) - 2), 2) - (m*global_entropy - m1*ent1 - m2*ent2 )) / n
    return d


def entropy_discretization(data, target):
    disc_df = data[target]

    n = len(data[target])

    for i in data.drop([target], axis=1).columns:
        sub_df = data[[i, target]]
        sub_df_sorted = sub_df.sort_values(by=i)
        print(i, "---------------initial value number",len(sub_df[i].unique()))

        sub_df_sorted_index_reset = sub_df_sorted.reset_index()
        splits = compute_info_gain(sub_df_sorted_index_reset, i,  target, [], 100)

        sub_df_sorted_index_reset = sub_df_sorted_index_reset.drop(target, axis=1)
        bin = 0
        splits.append(0)
        splits.append(n-1)
        splits_sorted = np.sort(splits)
        l = splits_sorted.size

        for j in range(l-1):
            if j == 0:
                for k in range(splits_sorted[j], splits_sorted[j+1]+1):
                    sub_df_sorted_index_reset.loc[k, i] = bin
            else:
                for k in range(splits_sorted[j]+1, splits_sorted[j+1]+1):
                    sub_df_sorted_index_reset.loc[k, i] = bin

            bin+=1

        print("---------------bin", bin, "\n")

        part = sub_df_sorted_index_reset.set_index('index')
        disc_df = pd.concat([disc_df, part], axis=1)
        disc_df[i] = disc_df[i].astype(int)

    return disc_df.drop(target, axis=1)    