import glob
import numpy as np
import pandas as pd

pd.options.display.max_rows = 60
pd.options.display.max_columns = None

from bs4 import BeautifulSoup
import requests
import re


def fetch_data(url):
    r = requests.get(url)
    html_doc = r.text
    soup = BeautifulSoup(html_doc,features="lxml")
    a_tags = soup.find_all('a')
    urls = ['https://raw.githubusercontent.com' + re.sub('/blob', '', link.get('href'))
            for link in a_tags if '.csv' in link.get('href')]
    df_list_names = [url.split('.csv')[0].split('/')[url.count('/')] for url in urls]
    df_list = [pd.DataFrame([None]) for i in range(len(urls))]
    return [pd.read_csv(url, sep=',') for url in urls], df_list_names
    # df_dict = dict(zip(df_list_names, dfs))


def getdata_country(dfs, country):
    for i in range(len(dfs)):
        if 'Country/Region' in dfs[i].columns:
            dfs[i] = dfs[i].loc[dfs[i]['Country/Region'] == country]
        else:
            dfs[i] = dfs[i].loc[dfs[i]['Country_Region'] == country]
    return dfs


def data_cleaning(dfs, all_files):
    [(i[0:5], j.shape[0]) for i, j in zip(all_files, dfs)]
    indices = [i for i, s in enumerate(all_files) if '03-22' in s]
    df = [i.drop(i.columns[[0, 1, 2, 3, 4, 5, 6]], axis=1) for i in dfs[indices[0]:]]

    for i, j in zip(all_files[indices[0]:], df):
        j['Date'] = i[0:5] + '-2020'

    data = pd.concat([i for i in df], ignore_index=True)
    data.Date = pd.to_datetime(data.Date)
    return data


def match_population(fips_url):
    fips = pd.read_csv(fips_url)
    return data.merge(fips[['Population', 'Combined_Key']], how='left', on='Combined_Key')


def total_data(data):
    temp = data.groupby(['Combined_Key', 'Date'])[['Confirmed', 'Deaths', 'Recovered']].sum().diff().reset_index()

    mask = temp['Combined_Key'] != temp['Combined_Key'].shift(1)
    # make the first of a region as np.nan
    temp.loc[mask, 'Confirmed'] = np.nan
    temp.loc[mask, 'Deaths'] = np.nan
    temp.loc[mask, 'Recovered'] = np.nan

    temp.columns = ['Combined_Key', 'Date', 'New Cases', 'New deaths', 'New recovered']
    data = pd.merge(data, temp, on=['Combined_Key', 'Date'])

    data = data.fillna(0)

    cols = ['New Cases', 'New deaths', 'New recovered']
    data[cols] = data[cols].astype('int')
    data['New Cases'] = data['New Cases'].apply(
        lambda x: 0 if x < 0 else x)  # question mark why there is negative values

    data.to_pickle('./data/total_data.pkl', protocol=4)


def county_file(county):
    countyfile = data.query('Combined_Key == {}'.format(county))
    countyfile.reset_index(inplace=True, drop=True)
    countyfile.to_pickle('county.pkl', protocol=4)


def run_all_fetch_data():
    url = 'https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data/csse_covid_19_daily_reports'
    fips_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/UID_ISO_FIPS_LookUp_Table.csv'
    country = 'US'
    dfs, all_files = fetch_data(url)
    dfs = getdata_country(dfs, country)
    data = data_cleaning(dfs, all_files)
    # Match population
    data = match_population(fips_url)

    # Daily data Cleaning
    total_data(data)

url = 'https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data/csse_covid_19_daily_reports'
fips_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/UID_ISO_FIPS_LookUp_Table.csv'
country = 'US'

dfs, all_files = fetch_data(url)
dfs = getdata_country(dfs, country)
data = data_cleaning(dfs, all_files)

data.head()

# Match population
data = match_population(fips_url)

# Daily data Cleaning
total_data(data)

