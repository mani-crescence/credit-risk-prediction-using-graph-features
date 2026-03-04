import pandas as pd



### cleaning functions ###
def delete_attribute(df,attributes):
    for i in attributes:
        df = df.drop(i,axis=1)
    return df 


def select_nan_attribute(df, percent):
    # we retain only columns which nan values are < 'percent'%
    return df.columns[df.isna().sum()*100/df.shape[0] >= percent]

def delete_nan_row(df):
    df = df.dropna(axis=0)
    return df


def to_float(df,symbol, attributes):
    
    for col in attributes:
        for i , row in df.iterrows():
            if isinstance(df.loc[i, col], str):
                # df.loc[i,col] = float(re.sub(symbol,"",df.loc[i,col])) * 0.01
                df.loc[i,col] = float(df.loc[i,col].replace('%', '')) * 0.01
        df[col] = df[col].astype(float)       
                
                
    return df


def cleaning(df,useless_attributes, percent, float_attributes, symbol):
    d = df.drop_duplicates()
    
    if useless_attributes:
        d = delete_attribute(d, useless_attributes)
        
    att = select_nan_attribute(d,percent)
    d = delete_attribute(d, att)
    d = delete_nan_row(d)
    
    if float_attributes:
        d = to_float(d, symbol, float_attributes)
    

    return d


def remove_char(data, attributes, char):
    for col in attributes:
        data.loc[:, col] = data.loc[:, col].str.replace(char, '')
        # data.loc[:, col] = data.loc[:, col].astype(float)
        data[col] = pd.to_numeric(data[col], errors='coerce')

        
    return data  
