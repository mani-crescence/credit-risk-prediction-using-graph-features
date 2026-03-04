from sklearn.preprocessing import LabelEncoder , OneHotEncoder
import warnings
import numpy as np 
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix, classification_report , roc_auc_score, precision_score,accuracy_score, recall_score ,ConfusionMatrixDisplay
from sklearn.model_selection import learning_curve
from sklearn import metrics
from sklearn.model_selection import train_test_split
from tools.cleaning import delete_attribute
from sklearn.preprocessing import KBinsDiscretizer
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import ast

warnings.filterwarnings("ignore", category=UserWarning)

### preprocessing functions ### 
def standardization(df ,attributes, val1, val2):
    for col in attributes:
        j = 0
        q1 = df[col].quantile(val1)
        q3 = df[col].quantile(val2)
        iqr = q3 - q1
         
        inf = q1 - (1.5 * iqr)
        sup = q3 + (1.5 * iqr)
        
        cmax = df[col].max()
        
        if cmax == 0 or iqr == 0:
            # print(col, "cmax ->", cmax, "iqr ->", iqr)
            df = df.drop(col,axis=1)
            
        elif cmax > sup:
            for i, _ in df.iterrows():
                if (df.loc[i,col] < inf):
                    df.loc[i,col]  = inf / sup  
                    j= j +1
                    
                elif (df.loc[i,col] > sup):
                    df.loc[i,col]  = 1
                    j = j+ 1
                
                else: df.loc[i,col] = df.loc[i,col] / sup
        elif cmax <= sup:  
            for i, _ in df.iterrows():
                if (df.loc[i,col] < inf):
                    df.loc[i,col]  = inf / cmax  
                    j = j + 1
                else: df.loc[i,col] = df.loc[i,col] / cmax
                   
    return df 


def plot_float_attribute(df,attributes,first,third,length):
    for col in attributes:
        q1 = df[col].quantile(first)
        q3 = df[col].quantile(third)
        iqr = q3 - q1

        inf = q1 - (1.5 * iqr)
        sup = q3 + (1.5 * iqr)
#         x = [inf,sup]
#         plt.xticks(x)
        sb.displot(df[col], kde=False)
        plt.vlines(x = inf, ymin = 0, ymax = length,
               color = 'green',  
               label = 'borne inferieure',
               linestyles='dotted')
        plt.vlines(x = sup, ymin = 0, ymax = length,
               colors = 'red',
               label = 'borne superieure',
               linestyles='dotted')
#         fig, ax = plt.subplots(figsize=(12,6))
        plt.legend(bbox_to_anchor = (1.0, 1), loc = 'upper right')
        plt.show()
        plt.savefig(str(col)+".jpg", dpi=300)


def label_encoder(df, attributes):
    encoder = LabelEncoder()
    for col in attributes:
        data = encoder.fit_transform(df[col])
        df[col] = np.divide(data,data.max())
            
    return df 

def manual_encoder_f(df,attributes):
    for col in attributes:
        values = df[col].unique()
        code = {value: index + 1 for index, value in enumerate(values)}    
        data = df[col].map(code)
        if data.max() != 0 :
            df[col] = np.divide(data, data.max())
        else: df[col]= data.astype({col:'float'})
           
    return df

def feature_selection_correlation(df,threshold,target):
    col_corr = set()
    
    cor_matrix = df.corr()
    
    for i in cor_matrix.columns:
        attributes = set()
        attributes.add(i)
        for j in cor_matrix.drop(i,axis=1).columns:
            if abs(df[i].corr(df[j])) > threshold:
                attributes.add(j)
                
        value = 0
        elt = ''
        for t in attributes:
            if abs(df[t].corr(df[target])) >= value:
                value = df[t].corr(df[target])
                elt = t 
               
      
        if elt != '':
            attributes.discard(elt) 
            for i in attributes:
                col_corr.add(i)
                
                
    return col_corr
    
def preprocessing(data,trainset, testset, classe, split_size, first_quartile, third_quartile, is_pre_processed = True):
    
    features = feature_selection_correlation(data,0.8, classe)
    
    corr_features = np.array(list(features))
    
    
    # print()
    
    data = data.drop(corr_features, axis=1)

    ########### SPLITTING SET ###############
    if( trainset is None and testset is None):
        trainset, testset = train_test_split(data, test_size = split_size, random_state=42)
        
    print(f" trainset shape: {trainset.shape}")
    print(f"testset shape: {testset.shape}")


    if (is_pre_processed):   
        ########### PREPROCESSING #################
        train_process = standardization(trainset, data.columns, first_quartile, third_quartile)
        test_process = standardization(testset, data.columns, first_quartile, third_quartile)
        
        # print('yes')

        # for col in data.select_dtypes('object'):
        #     train_process = manual_encoder(train_process,train_process.select_dtypes('object'))
        #     test_process = manual_encoder(test_process,test_process.select_dtypes('object'))
            # manual_encoder(test, col)
        # print(train_process)     

        ############## HIGH CORRELATED VARIABLES ##########
        # train_rv = train.drop(['application_type','hardship_flag'],axis=1)
        # print(train)
        # features = feature_selection_correlation(test,0.8, classe)
        # train = delete_attribute(train,features)
        # test = delete_attribute(test,features)
        # print(f"train: {train_f[classe].value_counts()}")
        # print(f"test: {test_f[classe].value_counts()}")

        return train_process, test_process
    else: 
        return trainset, testset
     
#dicretisation of continues attributs
def discretization_engine(data):
    variables = []
    
   
    # for column in data.columns:
        #  variables.append(column)
    
    # disc = EqualWidthDiscretiser(bins=5, variables=variables, return_boundaries=False)
    # disc.fit(data)
    # disc_data = disc.transform(data)
    # disc_test = disc.transform(test)
    
    # n_bins = 
    for col in data.columns:
        
        n_bins = int(np.ceil(3.49 * (1 / np.cbrt(data['LiabilitiesTotal'].shape[0]))))

        print(n_bins)
    
    disc = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='kmeans')
    print(disc)
    disc_data = disc.fit_transform(data)
    
    # print(data)
    disc_data_df = pd.DataFrame(disc_data, index = data.index, columns=data.columns)
    
    # print(disc_data_df)
    return disc_data_df
    
# receive data discretized and transform it into a dictionnary    
def discretization_encoding(data):
    
    attributes = {}
    
    for column in data.columns:
        attributes[column] = []
        attributes[column] = data[column].unique().tolist()  
        
    return attributes  

def process_discretization(data):
   
    # train = trainset.drop(classe, axis=1)
    # test = testset.drop(classe, axis=1)
    # c = data.columns

    # data_cont = data[c]
    # test_cont = data[c]
    

    tr_var, tr_data = discretization(data)
    # attributes_rest = data.drop(c, axis = 1)
    # rest = data.drop(c, axis = 1)
    
    # rest.astype(object)

    # r = discretization_encoding(attributes_rest)
    # ts = discretization_encoding(ts_rest)

    # to concatenate two dictionnaries, you need to use update
    # tr_var.update(r)
    # ts_rest.update(ts)  
    
    # df = pd.concat([tr_data, rest], axis=1)
    
    return tr_var, tr_data
        

def discretization(data):
    tr  = discretization_engine(data)
    tr_discretized = discretization_encoding(tr)
    # ts_discretized = discretization_encoding(ts)     
    return tr_discretized, tr

def fill_nan_attribute(df):
    df_float = df.select_dtypes('float').fillna(0)
    df_int = df.select_dtypes('int').fillna(0)
    df_object = df.select_dtypes('object').infer_objects().fillna('NA')
    df_bool = df.select_dtypes('bool').fillna(False)
    
    attributes_float = df_object.columns
    
    df = pd.concat([df_float, df_int, df_object, df_bool], axis=1)
    for col in attributes_float:
        df[col] = df[col].astype(object)
    
    return df  

def fill_nan_attributes2(data, attributes):
    unnan_data = data[data.columns[(data.isnull().sum() / data.shape[0])*100 < 80]]
    
    df_float = df.select_dtypes('float').fillna(0)
    df_int = df.select_dtypes('int').fillna(0)
    df_object = df.select_dtypes('object').infer_objects().fillna('NA')
    df_bool = df.select_dtypes('bool').fillna(False)
    
    attributes_float = df_object.columns
    
    df = pd.concat([df_float, df_int, df_object, df_bool], axis=1)
    for col in attributes_float:
        df[col] = df[col].astype(object)
    
    return df  
    


def one_hot_encoder(data, attributes):
    
    encoder = OneHotEncoder(sparse_output=False)

    columns_encoded = encoder.fit_transform(data[attributes])

    data_one_hot_encoded = pd.DataFrame(columns_encoded, columns=encoder.get_feature_names_out(attributes), index=data.index)


    return data_one_hot_encoded


def manual_encoder(data, att):
    for i in range(len(att[0])):
        d = {name: eval(value) for name , value in att[1][i].items()}
        data.loc[:,att[0][i]] = data.loc[:,att[0][i]].map(d)
        data[att[0][i]] = data[att[0][i]].astype('float')
    return data    

def bool_encoder(data, eng):
    for col in data.select_dtypes('bool'):
        data[col] = data[col].map(eng).astype(int)
    
    return data    

def convert_int_to_float(data, attributes):
    for col in attributes:
        # inter = data.loc[:, col].astype(float)
        data[col] = data[col].astype('float')
        
    return data    

def KMeans_discretisation(data):
    
    for col in data.select_dtypes('float').columns:

        att = data[col].values
        
        att = att.reshape(-1, 1)
        optimal_discretization = {
            'score' : 0,
            'data' : []
        }
        
        for i in  range(2, 10):
            kmeans = KMeans(n_clusters=i)
            kmeans.fit(att)
            cluster_labels = kmeans.predict(att)
            
            silhouette_avg = silhouette_score(att, cluster_labels)
            
            if silhouette_avg > optimal_discretization['score']:
                optimal_discretization['score'] = silhouette_avg
                optimal_discretization['data'] = cluster_labels
                    
        data.loc[:, col] = optimal_discretization['data']   
    
    return data


def convert_int_to_object(data, attributes):
    for col in attributes:
        print(f'col{col} ==> {data[col].unique()}')
        data.loc[:,col] = data.loc[:,col].astype(str)
        
    return data     