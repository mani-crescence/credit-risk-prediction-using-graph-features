import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from ..tools.training import *
from ..tools.preprocessing import * 
from ..tools.cleaning import *
import sys
import os
from sklearn.preprocessing import StandardScaler

pd.set_option('display.max_columns', None)

def clean_data(df, useless_columns):
    """
    Clean the given DataFrame, drop unnecessary columns, handle missing data, remove biased variables, 
    avoid multicollinearity by checking correlation, and keep only relevant columns.
    """
    
    # Drop these forward-looking biased variables
    df = df.drop(useless_columns, axis=1)

    df = df.dropna(axis=0)

    df = df.reset_index(drop=True)
    
    partial_preprocessed_data = df.copy()

    object_columns = df.select_dtypes("object").columns.to_list()
    

    encoder = OneHotEncoder(sparse_output=False)
    
    columns_encoded = encoder.fit_transform(df[object_columns])
    
    
    data_one_hot_encoded = pd.DataFrame(columns_encoded, columns=encoder.get_feature_names_out(object_columns)).astype('float')
    
    data_preprocessed = pd.concat([df, data_one_hot_encoded], axis=1)
    
    data_preprocessed = data_preprocessed.drop(object_columns, axis=1)

    # Generate a correlation matrix of the DataFrame
    corr_matrix = data_preprocessed.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find columns with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    
    # Drop these columns from the DataFrame
    data_preprocessed = data_preprocessed.drop(data_preprocessed[to_drop], axis=1)
    
    return data_preprocessed, partial_preprocessed_data[sorted(list(set(partial_preprocessed_data.columns) - set(to_drop)))]

    
def remove_highly_correlated_features(df, target, threshold=0.8):
    """
    Remove highly correlated features from a DataFrame based on a given threshold.

    Parameters:
    df (pandas.DataFrame): The input DataFrame.
    threshold (float): The threshold for high correlation. Default is 0.8.

    Returns:
    df_filtered (pandas.DataFrame): The filtered DataFrame with highly correlated features removed.
    """

    features_to_remove = []

    # Exclude the "Default" column from correlation analysis
    df_subset = df.drop([target], axis=1)

    # Calculate the correlation matrix
    correlation_matrix = df_subset.corr()

    # Identify highly correlated features
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            if abs(correlation_matrix.iloc[i, j]) > threshold:
                # Append the feature to the removal list
                features_to_remove.append(correlation_matrix.columns[j])

    # Remove the highly correlated features
    df_filtered = df.drop(features_to_remove, axis=1)

    return df_filtered, correlation_matrix, features_to_remove

def create_balanced_sample(df, df_partial, n, target, replace = False):
    df_sample = pd.concat(
        [
            df[df[target] == 0].sample(n=n, random_state=1, replace=replace),
            df[df[target] == 1].sample(n=n, random_state=1, replace=replace),
        ]
    )
    
    df_sample_partial = df_partial.loc[df_sample.index]
    df_sample.reset_index(drop=True, inplace=True)
    df_sample_partial.reset_index(drop=True, inplace=True)
    
    return df_sample, df_sample_partial

if __name__== "__main__":
    args = sys.argv[1:]
    db_name = args[0]
    path = args[1]
    target = args[2]
    unuseful_attributes = args[3]
    unuseful_attributes = ast.literal_eval(unuseful_attributes)
    number_of_sample = int(args[4])
    
    data = pd.read_csv(path)
    
    df_clean, partial_preprocessed_data = clean_data(data, unuseful_attributes)
   
    # Initialize the scaler
    scaler = StandardScaler()

    X_df = df_clean.drop(target, axis=1)

    # Transform the features
    scaled_features = scaler.fit_transform(X_df)

    # # Create a new DataFrame for the scaled features
    scaled_df = pd.DataFrame(scaled_features, columns=X_df.columns)  # Exclude the 'Default' column name

    # Add the 'Default' column back into the DataFrame
    scaled_df[target] = df_clean[target]
    
    
    df, correlation_matrix, features_to_remove = remove_highly_correlated_features(scaled_df, target, threshold=0.9)
    
    df[target] = df[target].astype('bool')
    
    df_sample, df_sample_partial = create_balanced_sample(df, partial_preprocessed_data, number_of_sample, target)
    
    trainset, testset = train_test_split(df_sample, test_size=0.2, random_state=42) 
    
    # df_sample_partial = df_sample_partial[list(set(partial_preprocessed_data.columns).intersection(set(df.columns)))]
    
    partial_preprocessed_train = df_sample_partial.loc[trainset.index]
    partial_preprocessed_test = df_sample_partial.loc[testset.index]
    
    
    directory='data/preprocessed/'+ db_name +'/'
    os.makedirs(directory, exist_ok=True)
    trainset.to_feather(directory + '/preprocessed_data_train.feather')
    testset.to_feather(directory + '/preprocessed_data_test.feather')
    
    partial_preprocessed_train.to_feather(directory + '/partial_preprocessed_data_train.feather')
    partial_preprocessed_test.to_feather(directory + '/partial_preprocessed_data_test.feather')
    


    
    
    
 