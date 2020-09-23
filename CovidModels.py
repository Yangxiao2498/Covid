import pathlib
import pandas as pd
from datacleaning import *
from helpers import load_data
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from controls import COUNTIES
from sklearn.metrics import mean_squared_error
from datetime import datetime as dt
from datetime import datetime, timedelta
from collections import defaultdict
from csv import DictWriter
from json import dump
import numpy as np
from scipy import stats
from scipy.integrate import odeint
from statistics import mean
import json
#############################################################################################
###                                    Functions                                          ###
#############################################################################################

#---------------------
##### Name:    deriv       
##### SIR simulation differential equation
#---------------------

#### Name: moving_average
#### moving average on beta and gamma
#---------------------


#### Name: calcvalues
#### clean the whole raw data and pass to prediction
#---------------------

#### Name: prediction
#### Ridge regression training and prediction function
#---------------------

#### Name: update_interventionR0
#### Calculate Rt in intervention date after prediction
#---------------------



data = load_data('total_data.pkl')
dferr = pd.read_csv('errorlist.csv')
error_list = list(dferr['errorlist'])
pp = 14



## SIR simulation differential equation
#Input:
# 1)S,I,R,N(population),beta and gamma at the beginning of sim
# 2)t defines the period of sim

#ouput:S,I,R at the end of sim after t time

def deriv(y, t, N, beta, gamma):
    S, I, R = y

    dSdt = -beta * S * I / N

    dIdt = beta * S * I / N - gamma * I

    dRdt = gamma * I

    return dSdt, dIdt, dRdt

def moving_average(data,window_size):
    data_to_series = pd.Series(data)
    windows = data_to_series.rolling(window_size)
    moving_average = windows.mean()
    return moving_average

##
def calcvalues(selectstate, selectcounty):
    name = selectcounty + ', ' + selectstate.strip() + ', ' + 'US'
    df = data_cleaning(data, name)

    date=df['Date'].dt.date.max()
    currentdf=df[df['Date'].dt.date ==date]

    df['I'] = df['I'] + 1
    shape = df.shape[0]
    last_day = data['Date'].max()

    recent=len(df)
    Date=df['Date'][recent-1]
    dfdate=df[df['Date']==Date]
    confirmed = dfdate.loc[dfdate['Date']== Date,'Confirmed'].iloc[0]
    death_rate=round(((dfdate.loc[dfdate['Date']== Date,'Deaths'].iloc[0]) / confirmed )*100,3)

    return confirmed, (confirmed*death_rate /100), death_rate, df


## Ridge regression training and prediction function
# Input: selectstate(str),selectcounty(str)
# Output:Json file including 
                        #'df': df,County data frame
                        # 'pred_data_date':dates in prediction period
                        # 'pred_data':df,predicted active,recovered and death cases
                        # 'rr':float, Rt at the beginning of prediction
                        # 'predict_period':int
                        # 'R0_pred':list,Rt at the end of prediction
                        # 'beta_pred': float,beta at the end of prediction
                        # 'gamma_pred':float,gamma at the end of prediction
def prediction(selectstate,selectcounty,predict_period = pp):
    print('prediction')
    name = selectcounty + ', ' + selectstate.strip() + ', ' + 'US'
    confirmed, death, death_rate, df = calcvalues(selectstate,selectcounty)

    # Train data is full dataset - last 7 days which is made into the test data
    train_df = df[df['Date'] < df.Date.iloc[-7]]
    test_df = df[(df['Date'] >= df.Date.iloc[-7]) & (df['Date'] <= df.Date.iloc[-1])]
    recent = len(df)
    shape = df.shape[0]

    Date = df['Date'][recent - 1]
    dfdate = df[df['Date'] == Date]


    confirmed = dfdate.loc[dfdate['Date'] == Date, 'Confirmed'].iloc[0]
    deaths = round(((dfdate.loc[dfdate['Date'] == Date, 'Deaths'].iloc[0]) / confirmed) * 100, 3)

    population = df.Population[1] 
    df['R(t+1)'] = df['R'].shift(-1)
    df['I(t+1)'] = df['I'].shift(-1)
    df['gamma'] = (df['R(t+1)'] - df['R'])/df['I']

    df['gamma'] = df['gamma'].apply(lambda x: 0  if x < 0 else x)

    df['beta'] = (df['gamma'] + (df['I(t+1)'] - df['I'])/df['I']).apply(lambda x:0  if x < 0 else x)

    #using z_score to find the outlier and set threshold to be 4
    # replace gamma
    # z_gamma = np.abs(stats.zscore(df['gamma']))
    # loc = np.where(z_gamma >4)
    # num_of_outliers = len(loc[0])
    # g_mean = df['gamma'].mean()
    # for i in range(num_of_outliers):
    #     df['gamma'].iloc[loc[0][i]] = g_mean

    # #replace beta
    # z_beta = np.abs(stats.zscore(df['beta']))
    # loc = np.where(z_beta >4)
    # num_of_outliers = len(loc[0])
    # b_mean = df['beta'].mean()
    # for i in range(num_of_outliers):
    #     df['beta'].iloc[loc[0][i]] = b_mean

    ## End
    #----------------------------------------------------------------------

    # 3_days moving average to smooth the beta and gamma before doing regression
    df['beta'] = moving_average(df['beta'],3)
    df['gamma'] = moving_average(df['gamma'],3)
    #-----------
    df['beta(t-3)'] = df['beta'].shift(3)
    df['beta(t-2)'] = df['beta'].shift(2)
    df['beta(t-1)'] = df['beta'].shift(1)
    ####
    df['beta(t-4)'] = df['beta'].shift(4)
    df['beta(t-5)'] = df['beta'].shift(5)
    df['beta(t-6)'] = df['beta'].shift(6)
    df['beta(t-7)'] = df['beta'].shift(7)
    df['gamma(t-4)'] = df['gamma'].shift(4)
    df['gamma(t-5)'] = df['gamma'].shift(5)
    df['gamma(t-6)'] = df['gamma'].shift(6)
    df['gamma(t-7)'] = df['gamma'].shift(7)
    ###
    df['gamma(t-3)'] = df['gamma'].shift(3)
    df['gamma(t-2)'] = df['gamma'].shift(2)
    df['gamma(t-1)'] = df['gamma'].shift(1)
    ##
    beta_X = df[['beta(t-3)','beta(t-2)','beta(t-1)','beta(t-4)','beta(t-5)','beta(t-6)','beta(t-7)']][(shape-1):]
    gamma_X = df[['gamma(t-3)','gamma(t-2)','gamma(t-1)','gamma(t-4)','gamma(t-5)','gamma(t-6)','gamma(t-7)']][(shape-1):]
    
    # training data set range  df.loc[i:]
    i= 10
    train_beta_y = df['beta'][i:(shape-1)]
    train_beta_X = df[['beta(t-3)','beta(t-2)','beta(t-1)','beta(t-4)','beta(t-5)','beta(t-6)','beta(t-7)']][i:(shape-1)]
    ##--dynamic alpha for regression(elastic)
    ridge =Ridge()
    alpha = np.linspace(0,1,11)
    Parameter_ = {'alpha':alpha.tolist()}
    ##--
    beta_ridge = GridSearchCV(ridge,Parameter_,scoring='neg_mean_squared_error',cv=5)
    #beta_ridge = Ridge(alpha=0.03)
    beta_ridge.fit(train_beta_X,train_beta_y)
    #beta_ridge.fit(train_beta_X,train_beta_y,sample_weight = weight)
    ##--same as here,try to ues past 40 days 
    train_gamma_y = df['gamma'][i:(shape-1)]
    train_gamma_X = df[['gamma(t-3)','gamma(t-2)','gamma(t-1)','gamma(t-4)','gamma(t-5)','gamma(t-6)','gamma(t-7)']][i:(shape-1)]
    ##
    #gamma_ridge=Ridge(alpha=10**(-6))
    gamma_ridge = GridSearchCV(ridge,Parameter_,scoring='neg_mean_squared_error',cv=5)
    gamma_ridge.fit(train_gamma_X,train_gamma_y)
    #gamma_ridge.fit(train_gamma_X,train_gamma_y,sample_weight = weight)
    beta_= pd.DataFrame(beta_X.loc[shape-1]).transpose()
    gamma_ = pd.DataFrame(gamma_X.loc[shape-1]).transpose()
    beta_begin_pred = beta_ridge.predict(beta_)
    gamma_begin_pred = gamma_ridge.predict(gamma_)
    #rr=round(beta_begin_pred[0]/gamma_begin_pred[0],3)

    

    #forecast
    test_beta_y = df['beta'][(shape - 7):]
    test_beta_X = df[['beta(t-3)', 'beta(t-2)', 'beta(t-1)', 'beta(t-4)', 'beta(t-5)', 'beta(t-6)', 'beta(t-7)']][
                  (shape - 7):]
    beta_pred = beta_ridge.predict(test_beta_X)
    test_gamma_y = df['beta'][(shape - 7):]
    test_gamma_X = df[['gamma(t-3)', 'gamma(t-2)', 'gamma(t-1)', 'gamma(t-4)', 'gamma(t-5)', 'gamma(t-6)',
                       'gamma(t-7)']][(shape - 7):]
    gamma_pred = gamma_ridge.predict(test_gamma_X)
    I_pred = []
    I_pred.append(df.loc[shape - 1]['I'])
    R_pred = []
    S_pred = []
    R0_pred = []
    R_pred.append(df.loc[shape - 1]['R'])
    S_pred.append(population - I_pred[-1] - R_pred[-1])
    #time_type = datetime.strptime(pred_date, '%m-%d-%y')
    predict_period = predict_period
    for i in range(0, predict_period):
        beta_ = pd.DataFrame(beta_X.loc[shape - 1 + i]).transpose()
        gamma_ = pd.DataFrame(gamma_X.loc[shape - 1 + i]).transpose()
        # Force beta_pred to be non-negative
        beta_pred = np.maximum(0, beta_ridge.predict(beta_))

        # Force gamma_pred to be non-negative
        gamma_pred = np.maximum(0, gamma_ridge.predict(gamma_))
        beta_X.loc[shape + i] = [beta_X['beta(t-2)'].loc[shape + i - 1], beta_X['beta(t-1)'].loc[shape + i - 1],
                                 beta_pred[0], beta_X['beta(t-3)'].loc[shape + i - 1],
                                 beta_X['beta(t-4)'].loc[shape + i - 1], beta_X['beta(t-5)'].loc[shape + i - 1],
                                 beta_X['beta(t-6)'].loc[shape + i - 1]]
        gamma_X.loc[shape + i] = [gamma_X['gamma(t-2)'].loc[shape + i - 1], gamma_X['gamma(t-1)'].loc[shape + i - 1],
                                  gamma_pred[0], gamma_X['gamma(t-3)'].loc[shape + i - 1],
                                  gamma_X['gamma(t-4)'].loc[shape + i - 1], gamma_X['gamma(t-5)'].loc[shape + i - 1],
                                  gamma_X['gamma(t-6)'].loc[shape + i - 1]]
        data_pred = round((1 + beta_pred[0] - gamma_pred[0]) * I_pred[-1])
        S_to_I = round((beta_pred[0] - gamma_pred[0]) * I_pred[-1])
        I_to_R = round((gamma_pred[0]) * I_pred[-1])
        S_pred.append(S_pred[-1] - S_to_I)
        R_pred.append(R_pred[-1] + I_to_R)
        I_pred.append(data_pred)
        R0_ = beta_pred[0]/gamma_pred[0]
        #limit R0 value
        R0_ = round(R0_,3)
        R0_pred.append(R0_)
    rr =round(R0_pred[0],3)
        
    Death = list(map(lambda x: round((x * deaths) / 100), I_pred))
    pred_data = pd.DataFrame({'Time': list(range(len(I_pred))), 'I': I_pred, 'R': R_pred, 'S': S_pred, 'Death': Death})
    ##Calculate social dictancing factor
    beta_sd = (max(df['beta']) - min(df['beta'])) / mean(df['beta'][:-1])
    current_social = 100 * (beta_pred[0] - min(df['beta'])) / (beta_sd * mean(df['beta'][:-1]))
    recent = len(df)
    Date = df['Date'][recent - 1]
    dfdate = df[df['Date'] == Date]
    N = dfdate.loc[dfdate['Date'] == Date, 'Population'].iloc[0]
    pred_data_date = pred_data['Time'].iloc[0:(predict_period + 1)] + df['Day'].max()
    obj = {
        'df': df.to_json(), # must use pd.read_json() to convert back
        'pred_data_date': pred_data_date.to_json(), # must use pd.read_json() to convert back
        'pred_data': pred_data.to_json(), # must use pd.read_json() to convert back
        'rr': rr, # float
        'predict_period': predict_period, # int
        'R0_pred': R0_pred, # list
        'beta_pred': beta_pred[0], # float
        'gamma_pred': gamma_pred[0], # float
    }
    json_obj = json.dumps(obj)
        
    
    return json_obj


##Calculate Rt in intervention date after prediction
#Input 
        # 1)selectcounty(str), selectstate(str)
        # 2)simdatepicker(time): what time to do intervention
        # 3)stored_prediction(js):the output of the prediction
#Output
        # Rt

def update_interventionR0(selectcounty, selectstate, simdatepicker, stored_prediction):
    name = selectcounty + ', ' + selectstate.strip() + ', ' + 'US'
    if stored_prediction is not None:
        if name not in error_list:
            sim_period = 100
            # df,pred_data_date,pred_data,rr,predict_period,R0_pred,beta_pred,gamma_pred = prediction(selectstate,selectcounty)
            # Load stored JSON and create appropriate variables
            prediction = json.loads(stored_prediction)
            df = pd.read_json(prediction['df'])
            pred_data = pd.read_json(prediction['pred_data'])
            forecast_period = prediction['predict_period']
            
            shape = df.shape[0]
            date=df['Date'].max()
            prediction_end_date = date + timedelta(days=pp)
            try:
                time_type = datetime.strptime(simdatepicker, '%Y-%m-%d')
            except ValueError as err:
                simdatepicker = simdatepicker.split(" ")
                if len(simdatepicker) == 2:
                    time_type = datetime.strptime(simdatepicker[0], '%Y-%m-%d')

            # #--simulation period before preventive
            simu_1 = (time_type - prediction_end_date).days
            # --beta_sd = (max(df['beta']) - min(df['beta'])) / mean(df['beta'][:-1])
            # current_social = 100 * (beta_pred[0] - min(df['beta'])) / (beta_sd * mean(df['beta'][:-1]))
            # --
            N = df.iloc[-1].Population
            gamma = prediction['gamma_pred']

            # Initial conditions vector

            I0_simu = pred_data['I'].iloc[-1]
            R0_simu = pred_data['R'].iloc[-1] + pred_data['Death'].iloc[-1]
            S0_simu = N - I0_simu - R0_simu
        
            pred_data_date = pred_data['Time'].iloc[0:(forecast_period + 1)] + df['Day'].max()
            # --social_after_preventive = current_social * (1 - prevent_impact / 100)
            # beta_preventive = round(((social_after_preventive * beta_sd * mean(df['beta'][:-1]) / 100) + min(df['beta'])), 3)
            # beta = beta_preventive--
            #------------simulation before preventive
            t_1 = np.linspace(0,simu_1,simu_1)
            simu_1_y0 =  S0_simu, I0_simu, R0_simu
            simu_1_ret = odeint(deriv, simu_1_y0, t_1, args=(N, prediction['beta_pred'], gamma))
            S, I, R = simu_1_ret.T
            R0 = round((-(S[-1]-S[-2])/(R[-1]-R[-2])),3)
            return R0
        else:
            return 0
    return 0





