import pandas as pd
import numpy as np

data = pd.read_pickle('./data/total_data.pkl')

def county_file(county):
    countyfile = data.query('Combined_Key == {}'.format(county))
    countyfile.reset_index(inplace=True, drop=True)
    countyfile.to_pickle('./data/{}.pkl'.format(county[1:].split(',')[0].lower()))



def fill_recovery(county_data):
    df = pd.read_pickle(county_data)
    df = county_data
    # fill in the first row
    df.loc[0, 'New Cases'] = int(df.loc[1:5, 'New Cases'].mean() * 0.8)
    df.loc[0, 'New deaths'] = int(df.loc[1:5, 'New deaths'].mean() * 0.8)
    df.loc[0, 'Recovered'] = int(df.loc[0, 'Confirmed'] * 0.2)

    # fill in new recovered

    ## fill in the second part
    for i in range(14, df.shape[0]):
        df.loc[i, 'New recovered'] = int(df.loc[i - 14, 'New Cases'] * 0.72)
    ## fill in the first half
    mean_value = df.loc[:7, 'New Cases'].mean() * 0.5 * 0.45
    std = df.loc[:7, 'New Cases'].mean() * 0.5 * 0.45 * 0.5
    df.loc[:13, 'New recovered'] = np.random.normal(mean_value, std, 14).astype('int')

    # fill in recovery data
    for i in range(1, df.shape[0]):
        df.loc[i, 'Recovered'] = df.loc[i - 1, 'Recovered'] + df.loc[i, 'New recovered']

    # rename and format
    df['I'] = df['Confirmed'] - df['Deaths'] - df['Recovered']
    df['R'] = df['Recovered'] + df['Deaths']
    df['Day'] = np.arange(df.shape[0])
    df.to_pickle(county_data)


def clean(county_full_name_with_quotes):
    # load county data
    county_file(county_full_name_with_quotes)

    # prep for the model format
    county_pkl_name = './data/' + county_full_name_with_quotes[1:].split(',')[0].lower() + '.pkl'


    fill_recovery(county_pkl_name)
#county='"New York City, New York, US"'
#clean(county)