import numpy as np
def data_cleaning(data,name):
    df=data[data['Combined_Key']==name].reset_index(drop=True)
    df.loc[0, 'New Cases'] = int(df.loc[1:5, 'New Cases'].mean() * 0.8)
    df.loc[0, 'New deaths'] = int(df.loc[1:5, 'New deaths'].mean() * 0.8)
    df.loc[0, 'Recovered'] = int(df.loc[0, 'Confirmed'] * 0.2)

    # fill in new recovered
    ## fill in the second part
    for i in range(14, df.shape[0]):
        df.loc[i, 'New recovered'] = int(df.loc[i - 14, 'New Cases'] * 0.85)
    ## fill in the first half
    mean_value = df.loc[:7, 'New Cases'].mean() * 0.5 * 0.45
    std = df.loc[:7, 'New Cases'].mean() * 0.5 * 0.45 * 0.5
    
    np.random.seed(123)
    df.loc[:13, 'New recovered'] = np.random.normal(mean_value, std, 14).astype('int')

    # fill in recovery data
    for i in range(1, df.shape[0]):
        df.loc[i, 'Recovered'] = df.loc[i - 1, 'Recovered'] + df.loc[i, 'New recovered']

    # rename and format
    df['I'] = df['Confirmed'] - df['Deaths'] - df['Recovered']
    df['R'] = df['Recovered'] + df['Deaths']
    df['Day'] = np.arange(df.shape[0])
    df.to_pickle('data/'+name)
    return df

def real_data(data,name):
    df=data[data['Combined_Key']==name].reset_index(drop=True)
    return df
