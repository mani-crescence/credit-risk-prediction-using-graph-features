import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from ..tools.training import *
from ..tools.preprocessing import * 
from ..tools.cleaning import *
import sys
import os
from sklearn.preprocessing import StandardScaler

def preprocess_main(data, target, db_name, attributes_for_manual_encoding = None, values_for_manual_encoding = None, label = None):
    data[target]  = data[target].astype('object')
    ############# STANDARDIZATION  ###############
    int_attributes = data.select_dtypes('int').columns
    numerical_data = convert_int_to_float(data, int_attributes)
    numerical_attributes = numerical_data.select_dtypes('float').columns.tolist()
    
    # exit(numerical_attributes)
    
    for col in numerical_attributes:
        try:    
            with open("engine/preprocessing/" + db_name + "/" + col + "_params_stan.txt") as file:
                parameters = ast.literal_eval(file.read())
        except:
            print("Empty file for " + col)
        else: 
            # exit(numerical_data.head())
            numerical_data = standardization(parameters['cmax'], parameters['sup'], parameters['inf'], parameters['iqr'], numerical_data, col)     
         
        
    partial_preprocessed_data = numerical_data.copy()

        ############# ENCODING #################  
    boolean_values = ast.literal_eval(os.getenv('BOOLEAN_VALUES') )    
    numerical_data = bool_encoder(numerical_data, boolean_values)
    numerical_data[target]  = numerical_data[target].astype('float')

     

          ######### ORDINAL ENCODING #######   
    if attributes_for_manual_encoding is not None and values_for_manual_encoding is not None:
         numerical_data = manual_encoder(numerical_data, [attributes_for_manual_encoding, values_for_manual_encoding])
         
          ####### ONE HOT ENCODING #######
        
    columns_for_one_hot_encoding = numerical_data.select_dtypes(include=['object']).columns.tolist()
    
    
    preprocessed_data = numerical_data.copy()
    
    try:
        with open("engine/preprocessing/" + db_name+ "/one_hot_encoder_engine", "rb") as file:
            encoder = pickle.load(file) 
    except:
        print("Not process for categorical data")
    else:
        one_hot_encoded_features = encoder.transform(preprocessed_data[columns_for_one_hot_encoding])  
        one_hot_encoded_data = pd.DataFrame(one_hot_encoded_features, columns=encoder.get_feature_names_out(columns_for_one_hot_encoding))
        preprocessed_data = pd.concat([preprocessed_data, one_hot_encoded_data], axis=1)
        preprocessed_data = preprocessed_data.drop(columns_for_one_hot_encoding, axis=1)  
    

    # partial_preprocessed_data = partial_preprocessed_data.sample(50)
    # index_t = partial_preprocessed_data.index
    # data_preprocessed= data_preprocessed.loc[index_t]
    
    directory='data/preprocessed/'+ db_name +'/'
    os.makedirs(directory, exist_ok=True)
    preprocessed_data.to_csv(directory+'/preprocessed_data_'+ label+'.csv')
    partial_preprocessed_data.to_csv(directory+'/partial_preprocessed_data_'+ label+'.csv')

def clean_data(df):
    """
    Clean the given DataFrame, drop unnecessary columns, handle missing data, remove biased variables, 
    avoid multicollinearity by checking correlation, and keep only relevant columns.
    """
    useless_columns = ["LoanDate", "FirstPaymentDate", "MaturityDate_Original", "MaturityDate_Last",
            "LanguageCode", "Country", "County",  "City", "year", "CreditScoreEsMicroL",
            "Restructured", "Rating", "LastPaymentOn", "MonthlyPaymentDay", "VerificationType", 'PrincipalPaymentsMade',
       'InterestAndPenaltyPaymentsMade', 'PrincipalBalance',  'InterestAndPenaltyBalance'
           ]

    # Drop these forward-looking biased variables
    df = df.drop(useless_columns, axis=1)

    df = df.dropna(axis=0)

    df = df.reset_index(drop=True)

    object_columns = df.select_dtypes("object").columns.to_list()

    encoder = OneHotEncoder(sparse_output=False)
    
    columns_encoded = encoder.fit_transform(df[object_columns])
    
    data_one_hot_encoded = pd.DataFrame(columns_encoded, columns=encoder.get_feature_names_out(object_columns)).astype('float')
    
    data_preprocessed = pd.concat([df, data_one_hot_encoded], axis=1)
    
    data_preprocessed = data_preprocessed.drop(object_columns, axis=1)

    # Generate a correlation matrix of the DataFrame
    corr_matrix = data_preprocessed.corr().abs()

    # print(corr_matrix)
    # print('before', len(data_preprocessed.columns))
    
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find columns with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    
    # Drop these columns from the DataFrame
    data_preprocessed = data_preprocessed.drop(data_preprocessed[to_drop], axis=1)

    # print('after', len(data_preprocessed.columns))
    
    
    df_clean = clean_data(df)

# Initialize the scaler
    scaler = StandardScaler()


    X_df = df_clean.drop('Default', axis=1)

    # Transform the features
    scaled_features = scaler.fit_transform(X_df)

    # # Create a new DataFrame for the scaled features
    scaled_df = pd.DataFrame(scaled_features, columns=X_df.columns)  # Exclude the 'Default' column name

    # # Add the 'Default' column back into the DataFrame
    scaled_df['Default'] = df_clean['Default']
    
    def remove_highly_correlated_features(df, threshold=0.8):
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
        df_subset = df.drop("Default", axis=1)

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
    df, correlation_matrix,features_to_remove = remove_highly_correlated_features(scaled_df, threshold=0.9)
    # df = create_balanced_sample(df, num_default_samples, replace=False)
    df['Default'] = df['Default'].astype('bool')

    return data_preprocessed

if __name__== "__main__":
    args = sys.argv[1:]
    db_name = args[0]
    path = args[1]
    target = args[2]
    unuseful_attributes = args[3]
    label = args[4]
    unuseful_attributes = ast.literal_eval(unuseful_attributes)
    
    data = pd.read_csv(path, low_memory=False, keep_default_na = False, na_values=[""])
    
    if len(args) > 5:
        attributes_for_manual_encoding = args[5]
        values_for_manual_encoding = args[6]

        attributes_for_manual_encoding = ast.literal_eval(attributes_for_manual_encoding)
        values_for_manual_encoding = ast.literal_eval(values_for_manual_encoding)
        preprocess_main(data, target,  db_name, attributes_for_manual_encoding, values_for_manual_encoding, label)
    else:
        preprocess_main(data, target,  db_name, None, None, label)



    
    
    
 