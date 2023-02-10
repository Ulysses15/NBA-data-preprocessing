import pandas as pd
import os
import requests
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Checking ../Data directory presence
if not os.path.exists('../Data'):
    os.mkdir('../Data')

# Download data if it is unavailable.
if 'nba2k-full.csv' not in os.listdir('../Data'):
    print('Train dataset loading.')
    url = "https://www.dropbox.com/s/wmgqf23ugn9sr3b/nba2k-full.csv?dl=1"
    r = requests.get(url, allow_redirects=True)
    open('../Data/nba2k-full.csv', 'wb').write(r.content)
    print('Loaded.')

data_path = "../Data/nba2k-full.csv"

# write your code here


def clean_str(salary):
    return salary.replace('$', '')


def clean_data(path):
    df = pd.read_csv(path)
    df['draft_year'] = pd.to_datetime(df['draft_year'], format='%Y')
    df['b_day'] = pd.to_datetime(df['b_day'], format='%m/%d/%y')
    df['team'].fillna('No Team', inplace=True)
    df['height'] = df['height'].apply(lambda x: x.split()[2])
    df['weight'] = df['weight'].apply(lambda x: x.split()[3])
    df['salary'] = df['salary'].apply(lambda x: clean_str(x))
    df['height'] = df['height'].astype('float')
    df['weight'] = df['weight'].astype('float')
    df['salary'] = df['salary'].astype('float')
    mask = df.country != 'USA'
    df.loc[mask, 'country'] = 'Not-USA'
    df.loc[df.draft_round == 'Undrafted', 'draft_round'] = '0'
    return df


def feature_data(df_clean):
    k = df_clean
    k['version'] = pd.to_datetime(k['version'], format='NBA2k%y')
    k['age'] = k['version'].dt.to_period('Y').astype(int) - k['b_day'].dt.to_period('Y').astype(int)
    k['experience'] = k['version'].dt.to_period('Y').astype(int) - k['draft_year'].dt.to_period('Y').astype(int)
    k['bmi'] = k['weight'] / k['height'] ** 2
    k.drop(columns=['version', 'b_day', 'draft_year', 'weight', 'height'], inplace=True)
    exclude_columns = list(k.select_dtypes(exclude='object'))
    count = k.nunique()
    drop_col = []
    for row in count.items():
        if row[1] >= 50:
            if row[0] not in exclude_columns:
                drop_col.append(row[0])
    k.drop(columns=drop_col, inplace=True)
    return k


def multicol_data(f_data):
    return f_data.drop(columns='age')


def transform_data(mlt_col_data) -> 'transform data to high quality':
    num_feat_df = mlt_col_data.select_dtypes('number')  # numerical features
    cat_feat_df = mlt_col_data.select_dtypes('object')  # categorical features
    y = num_feat_df['salary']
    num_wo_salary = num_feat_df.drop(columns='salary')
    """scaling and getting first part of the 'x' """
    scaler = StandardScaler()
    scaler.fit(num_wo_salary)
    num_feat_df = scaler.transform(num_wo_salary)
    num_col = list(num_wo_salary.columns)
    num_feat_df = pd.DataFrame(num_feat_df)
    num_feat_df.columns = num_col
    """encoding and getting the second part of the 'x' """
    encode = OneHotEncoder()
    encode.fit(cat_feat_df)
    df_encode = encode.transform(cat_feat_df)
    cat_feat_df = pd.DataFrame.sparse.from_spmatrix(df_encode)
    encode_col = encode.categories_  # a list of arrays
    columns_list = np.concatenate(encode_col).ravel().tolist()  # flatten the list of arrays and convert it to list
    cat_feat_df.columns = columns_list
    """concatenate two columns"""
    x = pd.concat([num_feat_df, cat_feat_df], axis=1)
    return x, y


# a = clean_data(data_path)
# b = feature_data(a)
# c = multicol_data(b)
# print(c)
# d = transform_data(c)
# print(d)
