import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
#import seaborn as sns; sns.set()
from helpers import *
from SIR_Model import *
from scipy.integrate import odeint
import matplotlib.animation as animation
import datetime as dt
from plotSimulation import *
from datacleaning import *
from datetime import datetime, timedelta
from statistics import mean
from sklearn.linear_model import Ridge,Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import seaborn as sns
from scipy import stats

def main():

    ## sidebar
    data=load_data('total_data.pkl')
    data1 = data[data['Date'] == data['Date'].iloc[0]]
    countydf=pd.DataFrame(data1.Combined_Key.str.split(',',2).tolist(),columns = ['County','State','Country']).drop('Country',axis=1)
    countydf=pd.DataFrame(countydf.groupby('State')['County'].apply(lambda x: x.values.tolist())).reset_index()
    statelist = countydf['State'].tolist()
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to",

                            ( 'Hybrid Model','Data Exploratory'))


    if page == 'Hybrid Model' :
        #Adding new changes -- gunjan
        st.title('Explore County Level Data ')

        # load data
        removed_state =[' US',' Wuhan Evacuee','Washington','District of Columbia']
        for i in removed_state:
            statelist.remove(i)
        statesselected = st.selectbox("Select a State", statelist)
        countylist=sorted(((countydf[countydf['State']==statesselected]['County']).tolist()[0]),key=lambda s: s.strip('"'))

        if statesselected == countydf['State'][3]:
            countylist.remove("Unassigned")
        countyselected = st.selectbox('Select a County',countylist)
        name=countyselected+', '+statesselected.strip()+', '+'US'
        df2=data_cleaning(data,name)
        df = df2

        date=df2['Date'].dt.date.max()
        currentdf=df2[df2['Date'].dt.date ==date]
        pastdate = date-timedelta(days=30)
        pastdf=df2[df2['Date'].dt.date >= pastdate].reset_index(drop=True)
        #Show data for the date
        st.title('{}, {} on {}'.format(statesselected, countyselected, date))

        #Adding chart for today:

        '## Exploring past 30 days data'
        #Visualizating past chart
        df_temp = pastdf.rename(columns = {'I':'Active Infection Cases','R':'Recovered Cases'})
        e = pd.melt(frame = df_temp,

                    id_vars='Date',

                    value_vars=['Active Infection Cases','Recovered Cases'],

                    var_name = 'type',

                    value_name = 'count')

        e = alt.Chart(e).mark_area().encode(

            x=alt.X('Date:T', title='Date'),

            y=alt.Y('count:Q',title = 'Number of Cases'),

            color = alt.Color('type:O',legend = alt.Legend(title = None,orient = 'top-left'))

        ).configure_axis(
            grid=False

        )

        st.altair_chart(e, use_container_width=True)

        #df2['I'] = df2['I'] + 1
        shape = df2.shape[0]
        last_day = data['Date'].max()

        #Maximum prediction range is from 14 days to 2 months

        recent=len(df2)
        Date=df2['Date'][recent-1]
        dfdate=df2[df2['Date']==Date]
        confirmed = dfdate.loc[dfdate['Date']== Date,'Confirmed'].iloc[0]
        deaths=round(((dfdate.loc[dfdate['Date']== Date,'Deaths'].iloc[0]) / confirmed )*100,3)
        df2['R(t+1)'] = df2['R'].shift(-1)
        df2['I(t+1)'] = df2['I'].shift(-1)
        #df2['I(t+1)'] = df2['I(t+1)'] - 1
        df2['gamma'] = (df2['R(t+1)'] - df2['R'])/df2['I']

        #df2['gamma'] = 1/14
        df2['beta'] = df2['gamma'] + (df2['I(t+1)'] - df2['I'])/df2['I']
        #st.write(df2)
        #3_days moving average to smooth the beta and gamma before doing regression
        window_size = 3
        beta_to_series = pd.Series(df2['beta'])
        windows = beta_to_series.rolling(window_size)
        moving_average = windows.mean()
        df2['beta'] = moving_average
        gamma_to_series = pd.Series(df2['gamma'])
        windows = gamma_to_series.rolling(window_size)
        moving_average = windows.mean()
        df2['gamma'] = moving_average
        #df2['R0'] = df2['beta']/df2['gamma']
        df2['beta(t-3)'] = df2['beta'].shift(3)
        df2['beta(t-2)'] = df2['beta'].shift(2)
        df2['beta(t-1)'] = df2['beta'].shift(1)
        ####
        df2['beta(t-4)'] = df2['beta'].shift(4)
        df2['beta(t-5)'] = df2['beta'].shift(5)
        df2['beta(t-6)'] = df2['beta'].shift(6)
        df2['beta(t-7)'] = df2['beta'].shift(7)
        df2['gamma(t-4)'] = df2['gamma'].shift(4)
        df2['gamma(t-5)'] = df2['gamma'].shift(5)
        df2['gamma(t-6)'] = df2['gamma'].shift(6)
        df2['gamma(t-7)'] = df2['gamma'].shift(7)
        ###
        df2['gamma(t-3)'] = df2['gamma'].shift(3)
        df2['gamma(t-2)'] = df2['gamma'].shift(2)
        df2['gamma(t-1)'] = df2['gamma'].shift(1)
        ##
        train_df = df2[df2['Date'] <  df2.Date.iloc[-7]]
        test_df = df2[(df2['Date'] >= df2.Date.iloc[-7]) & (df2['Date'] <= df2.Date.iloc[-1])]
        #st.write(df2['gamma'])
        beta_X = df2[['beta(t-3)','beta(t-2)','beta(t-1)','beta(t-4)','beta(t-5)','beta(t-6)','beta(t-7)']][(shape-1):]
        gamma_X = df2[['gamma(t-3)','gamma(t-2)','gamma(t-1)','gamma(t-4)','gamma(t-5)','gamma(t-6)','gamma(t-7)']][(shape-1):]
        ##
        d = -40
        ridge =Ridge()
        lasso = Lasso()
        alpha = np.linspace(0,1,11)
        Parameter_ = {'alpha':alpha.tolist()}
        train_beta_y = df2['beta'][10:(shape-7)]
        train_beta_X = df2[['beta(t-3)','beta(t-2)','beta(t-1)','beta(t-4)','beta(t-5)','beta(t-6)','beta(t-7)']][10:(shape-7)]
        # train_beta_y_40 = df2['beta'][d:(shape-7)]
        # train_beta_X_40 = df2[['beta(t-3)','beta(t-2)','beta(t-1)','beta(t-4)','beta(t-5)','beta(t-6)','beta(t-7)']][d:(shape-7)]
        #beta_ridge = Ridge(alpha=1)
        beta_ridge = GridSearchCV(ridge,Parameter_,scoring='neg_mean_squared_error',cv=5)
        beta_ridge.fit(train_beta_X,train_beta_y)

        # beta_ridge_40 = GridSearchCV(ridge,Parameter_,scoring='neg_mean_squared_error',cv=5)
        # beta_ridge_40.fit(train_beta_X_40,train_beta_y_40)
        
        train_gamma_y = df2['gamma'][10:(shape-7)]
        train_gamma_X = df2[['gamma(t-3)','gamma(t-2)','gamma(t-1)','gamma(t-4)','gamma(t-5)','gamma(t-6)','gamma(t-7)']][10:(shape-7)]
        # train_gamma_y_40 = df2['gamma'][d:(shape-7)]
        # train_gamma_X_40 = df2[['gamma(t-3)','gamma(t-2)','gamma(t-1)','gamma(t-4)','gamma(t-5)','gamma(t-6)','gamma(t-7)']][d:(shape-7)]
        
        
        
        #gamma_ridge=Ridge(alpha=10**(-6))
        
        # gamma_ridge_40 = GridSearchCV(ridge,Parameter_,scoring='neg_mean_squared_error',cv=5)
        # gamma_ridge_40.fit(train_gamma_X_40,train_gamma_y_40)

        gamma_ridge = GridSearchCV(ridge,Parameter_,scoring='neg_mean_squared_error',cv=5)
        gamma_ridge.fit(train_gamma_X,train_gamma_y)


        beta_= pd.DataFrame(beta_X.loc[shape-1]).transpose()
        gamma_ = pd.DataFrame(gamma_X.loc[shape-1]).transpose()
        beta_begin_pred = beta_ridge.predict(beta_)
        gamma_begin_pred = gamma_ridge.predict(gamma_)
        # st.write(beta_ridge.best_params_)
        # st.write(gamma_ridge.best_params_)
        #st.write(beta_begin_pred)
        #st.write(beta_ridge.coef_)
        rr=round(beta_begin_pred[0]/gamma_begin_pred[0],3)
# ----------- PRINTING CURRENT DAY STATS -----------------

        #``````````````````````````````````````````````````
        st.write('Confirmed cases on present day: ', confirmed)
        deaths_num = dfdate.loc[dfdate['Date']== Date,'Deaths'].iloc[0]
        deaths_mark = f'**Death cases on present day: ** **{deaths_num} **'
        st.markdown(deaths_mark)
        #st.write('Death cases on present day', dfdate.loc[dfdate['Date']== Date,'Deaths'].iloc[0])
        st.write('Death % today: ', deaths)
        rr_mark = f'**Effective reproduction number(R0) of the current day: ** **{rr:.2f}**'
        st.markdown(rr_mark)
        #st.write('R0 current: ', rr)

        #`````````````````````````````````````````````````````````````
        '## Forecast'

        date_select_box = list((last_day + dt.timedelta(days=x)).strftime('%m-%d-%y') for x in range(14,74))
        selected_pred_day = st.selectbox('Select a date to forcast to untill M/D/Y', date_select_box)

        population = df2.Population[1]



        ''## New method to estimate beta and gamma(time dependent)''

        test_beta_y = df2['beta'][(shape-7):]
        test_beta_X = df2[['beta(t-3)','beta(t-2)','beta(t-1)','beta(t-4)','beta(t-5)','beta(t-6)','beta(t-7)']][(shape-7):]
        beta_pred = beta_ridge.predict(test_beta_X)
        test_gamma_y = df2['beta'][(shape-7):]
        test_gamma_X = df2[['gamma(t-3)','gamma(t-2)','gamma(t-1)','gamma(t-4)','gamma(t-5)','gamma(t-6)','gamma(t-7)']][(shape-7):]
        gamma_pred = gamma_ridge.predict(test_gamma_X)
        I_pred = []
        I_pred.append(df2.loc[shape-1]['I'])
        R_pred = []
        S_pred = []
        R_pred.append(df2.loc[shape-1]['R'])
        S_pred.append(population - I_pred[-1] - R_pred[-1])
        time_type = datetime.strptime(selected_pred_day, '%m-%d-%y')
        forecast_period = predict_period = (time_type - last_day).days

        for i in range(0,forecast_period):
            beta_= pd.DataFrame(beta_X.loc[shape-1+i]).transpose()
            gamma_ = pd.DataFrame(gamma_X.loc[shape-1+i]).transpose()
            beta_pred = beta_ridge.predict(beta_)
            gamma_pred = gamma_ridge.predict(gamma_)
            beta_X.loc[shape+i] = [beta_X['beta(t-2)'].loc[shape+i-1],beta_X['beta(t-1)'].loc[shape+i-1],beta_pred[0],beta_X['beta(t-3)'].loc[shape+i-1],beta_X['beta(t-4)'].loc[shape+i-1],beta_X['beta(t-5)'].loc[shape+i-1],beta_X['beta(t-6)'].loc[shape+i-1]]
            gamma_X.loc[shape+i] = [gamma_X['gamma(t-2)'].loc[shape+i-1],gamma_X['gamma(t-1)'].loc[shape+i-1],gamma_pred[0],gamma_X['gamma(t-3)'].loc[shape+i-1],gamma_X['gamma(t-4)'].loc[shape+i-1],gamma_X['gamma(t-5)'].loc[shape+i-1],gamma_X['gamma(t-6)'].loc[shape+i-1]]
            data_pred = round((1+beta_pred[0]-gamma_pred[0])*I_pred[-1])
            S_to_I = round((beta_pred[0]-gamma_pred[0])*I_pred[-1])
            I_to_R = round((gamma_pred[0])*I_pred[-1])
            S_pred.append(S_pred[-1] - S_to_I)
            R_pred.append(R_pred[-1] + I_to_R)
            I_pred.append(data_pred)
        Death = list(map(lambda x: round((x * deaths)/100), I_pred))
        pred_data = pd.DataFrame({'Time':list(range(len(I_pred))),'I':I_pred,'R':R_pred,'S':S_pred,'Death':Death})
        ##Calculate social dictancing factor
        beta_sd = (max(df2['beta']) - min(df2['beta'])) / mean(df2['beta'][:-1])
        current_social = 100*(beta_pred[0] - min(df2['beta'])) /(beta_sd *mean(df2['beta'][:-1]))
        st.write('Prediction after {} days :'.format(forecast_period))
        #st.write('Current social distancing factor is ', round(current_social,3))
        #st.write(beta_pred[0]/gamma_pred[0])

        recent=len(df2)
        Date=df2['Date'][recent-1]
        dfdate=df2[df2['Date']==Date]
        N = dfdate.loc[dfdate['Date']== Date,'Population'].iloc[0]

        #---------------------
        #moving average 3 days and last 7 days test plot
        def test_plot(beta_ridge,gamma_ridge,df):
            df_test_beta = test_df[['beta(t-3)','beta(t-2)','beta(t-1)','beta(t-4)','beta(t-5)','beta(t-6)','beta(t-7)']]
            test_beta = beta_ridge.predict(df_test_beta)
            #df_test_beta
            df_test_gamma = test_df[['gamma(t-3)','gamma(t-2)','gamma(t-1)','gamma(t-4)','gamma(t-5)','gamma(t-6)','gamma(t-7)']]
            test_gamma = gamma_ridge.predict(df_test_gamma)
            pred = []
            #pred.append(df['I'][len(df)-40])
            pred.append(df2['I'][98])
            for i in range(0,7):
                pred_data = round((1+test_beta[i]-test_gamma[i])*pred[-1])
                pred.append(pred_data)
            x = np.linspace(1,7,7)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(x,df2['I'][-7:],color = 'g',label = 'Actual last 7 days data')
            ax.plot(x,pred[1:],color = 'b',label = 'Predicted testing data')
            legend = ax.legend()
            for spine in ('top', 'right', 'bottom', 'left'):
                ax.spines[spine].set_visible(False)
            plt.show()
            st.pyplot()
        test_plot(beta_ridge,gamma_ridge,test_df)
        #-----------------------

        ###############################################
        def plot_test(beta_ridge,gamma_ridge,df):
            population = df['Population'].iloc[-1]
            shape = df.index[0]
            test_or_train = 0
            if shape == 0:
                test_or_train = 7
            beta_X = df[['beta(t-3)','beta(t-2)','beta(t-1)','beta(t-4)','beta(t-5)','beta(t-6)','beta(t-7)']].iloc[test_or_train]
            gamma_X = df[['gamma(t-3)','gamma(t-2)','gamma(t-1)','gamma(t-4)','gamma(t-5)','gamma(t-6)','gamma(t-7)']].iloc[test_or_train]
            I_pred = []
            I_pred.append(df.iloc[0]['I'])
            R_pred = []
            S_pred = []
            R_pred.append(df.iloc[0]['R'])
            S_pred.append(population - I_pred[-1] - R_pred[-1])
            beta_= pd.DataFrame(beta_X).transpose()
            gamma_ = pd.DataFrame(gamma_X).transpose()
            beta_X = beta_
            gamma_X = gamma_
            for i in range(test_or_train,len(df)):
                beta_= pd.DataFrame(beta_X.iloc[(i-test_or_train)]).transpose()
                gamma_ = pd.DataFrame(gamma_X.iloc[(i-test_or_train)]).transpose()
                beta_pred = beta_ridge.predict(beta_)
                gamma_pred = gamma_ridge.predict(gamma_)
                beta_X.loc[shape+i+1] = [beta_X['beta(t-2)'].loc[shape+i],beta_X['beta(t-1)'].loc[shape+i],beta_pred[0],
                                        beta_X['beta(t-3)'].loc[shape+i],beta_X['beta(t-4)'].loc[shape+i],beta_X['beta(t-5)'].loc[shape+i],
                                        beta_X['beta(t-6)'].loc[shape+i]]
                gamma_X.loc[shape+i+1] = [gamma_X['gamma(t-2)'].loc[shape+i],gamma_X['gamma(t-1)'].loc[shape+i],gamma_pred[0],
                                            gamma_X['gamma(t-3)'].loc[shape+i],gamma_X['gamma(t-4)'].loc[shape+i],gamma_X['gamma(t-5)'].loc[shape+i],
                                            gamma_X['gamma(t-6)'].loc[shape+i]]
                data_pred = round((1+beta_pred[0]-gamma_pred[0])*I_pred[-1])
                S_to_I = round((beta_pred[0]-gamma_pred[0])*I_pred[-1])
                I_to_R = round((gamma_pred[0])*I_pred[-1])
                S_pred.append(S_pred[-1] - S_to_I)
                R_pred.append(R_pred[-1] + I_to_R)
                I_pred.append(data_pred)
            Death = list(map(lambda x: round((x * deaths)/100), I_pred))
            pred_data = pd.DataFrame({'Time':list(range(len(I_pred))),'I':I_pred,'R':R_pred,'S':S_pred,'Death':Death})
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(df['Day'],df['I'],color = 'y',lw = 2,label = 'Actual data')
            ax.plot((pred_data['Time'] + df['Day'].min()), pred_data['I'], color = 'r',label = 'Predictive data')
            legend = ax.legend()
            for spine in ('top', 'right', 'bottom', 'left'):
                ax.spines[spine].set_visible(False)
            plt.show()
            st.pyplot()
        #plot_test(beta_ridge,gamma_ridge,test_df)
        #plot_test(beta_ridge,gamma_ridge,train_df)
        
        st.title('fitting plot using whole data set as training')
        def plot_fitting(beta_ridge,gamma_ridge,df):
            # df_beta = df[['beta(t-3)','beta(t-2)','beta(t-1)','beta(t-4)','beta(t-5)','beta(t-6)','beta(t-7)']][-40:]
            # df_gamma = df[['gamma(t-3)','gamma(t-2)','gamma(t-1)','gamma(t-4)','gamma(t-5)','gamma(t-6)','gamma(t-7)']][-40:]
            df_beta = df[['beta(t-3)','beta(t-2)','beta(t-1)','beta(t-4)','beta(t-5)','beta(t-6)','beta(t-7)']][7:]
            df_gamma = df[['gamma(t-3)','gamma(t-2)','gamma(t-1)','gamma(t-4)','gamma(t-5)','gamma(t-6)','gamma(t-7)']][7:]
            gamma_fitting = gamma_ridge.predict(df_gamma)
            beta_fitting = beta_ridge.predict(df_beta)
            lenth = len(beta_fitting)
            pred = []
            #pred.append(df['I'][len(df)-40])
            pred.append(df['I'][7])
            for i in range(0,lenth):
                pred_data = round((1+beta_fitting[i]-gamma_fitting[i])*pred[-1])
                pred.append(pred_data)
            x = np.linspace(1,lenth,lenth)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(x,df['I'][7:],color = 'y',lw = 2,label = 'Actual data')
            ax.plot(x,pred[1:],color = 'b',lw = 2,label = 'Predictive data')
            legend = ax.legend()
            for spine in ('top', 'right', 'bottom', 'left'):
                ax.spines[spine].set_visible(False)
            plt.show()
            st.pyplot()
            # st.write(pred)
            # st.write(df['I'][10:])
            #st.write(beta_fitting)
            # st.write(gamma_fitting)
        #plot_fitting(beta_ridge,gamma_ridge,df2)
        st.title('fitting plot using past 40 days data set as training')
        def plot_fitting_40(beta_ridge,gamma_ridge,df):
            df_beta = df[['beta(t-3)','beta(t-2)','beta(t-1)','beta(t-4)','beta(t-5)','beta(t-6)','beta(t-7)']][-40:]
            df_gamma = df[['gamma(t-3)','gamma(t-2)','gamma(t-1)','gamma(t-4)','gamma(t-5)','gamma(t-6)','gamma(t-7)']][-40:]
            gamma_fitting = gamma_ridge_40.predict(df_gamma)
            beta_fitting = beta_ridge_40.predict(df_beta)
            lenth = len(beta_fitting)
            pred = []
            pred.append(df['I'][len(df)-40])
            #pred.append(df['I'][7])
            for i in range(0,lenth):
                pred_data = round((1+beta_fitting[i]-gamma_fitting[i])*pred[-1])
                pred.append(pred_data)
            x = np.linspace(1,lenth,lenth)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(x,df['I'][-40:],color = 'y',lw = 2,label = 'Actual data')
            ax.plot(x,pred[1:],color = 'b',lw = 2,label = 'Predictive data')
            legend = ax.legend()
            for spine in ('top', 'right', 'bottom', 'left'):
                ax.spines[spine].set_visible(False)
            plt.show()
            st.pyplot()
            # st.write(pred)
            # st.write(df['I'][10:])
            #st.write(beta_fitting)
            # st.write(gamma_fitting) 
        
        #plot_fitting_40(beta_ridge,gamma_ridge,df2)
        gamma =gamma_pred[0]

        rr=round(beta_begin_pred[0]/gamma_begin_pred[0],3)
        simulation_period = 0
        t = np.linspace(0, simulation_period, 500)
        # The SIR model differential equations.

        def deriv(y, t, N, beta, gamma):

            S, I, R = y

            dSdt = -beta * S * I / N

            dIdt = beta * S * I / N - gamma * I

            dRdt = gamma * I

            return dSdt, dIdt, dRdt

        # Initial conditions vector
        I0_simu = pred_data['I'].iloc[-1]
        R0_simu = pred_data['R'].iloc[-1] +pred_data['Death'].iloc[-1]
        S0_simu = N - I0_simu - R0_simu



        pred_data_date = pred_data['Time'].iloc[0:(forecast_period + 1)] +df2['Day'].max()






        def plotting(df2,pred_data_date,pred_data,predict_period, forecast_period, simulation_period):
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(df2['Day']-1,df2['I'],color = 'y',lw = 2,label = 'Actual data')
            ax.plot(pred_data_date-1, pred_data['I'].iloc[0:(predict_period+2)], color = 'r',label = 'Predictive data')
            begin = df2['Date'].min()
            num_days = len(df2) + forecast_period + simulation_period + 5
            labels = list((begin + dt.timedelta(days=x)).strftime('%m-%d') for x in range(0,num_days,5))
            plt.xticks(list(range(0,num_days,5)), labels, rotation=45)

            legend = ax.legend()
            for spine in ('top', 'right', 'bottom', 'left'):
                ax.spines[spine].set_visible(False)
            plt.show()
            st.pyplot()


        #Add past df
        #plotting(df2,pred_data_date,pred_data,predict_period, forecast_period, simulation_period)

        '## Simulation'

        simulation_period = st.slider('Input how many days to simulate after prediction date',1,100,step = 1,value = 1)
        level = st.slider('Estimated percentage to impact the current preventive measures by',-100,100,step = 10,value = 0)
        if level == 100:
            level = 99
        social_after_preventive = current_social * (1 - level / 100)
        beta_preventive = round(((social_after_preventive *beta_sd *mean(df2['beta'][:-1])/100) + min(df2['beta'])),3)
        beta = beta_preventive
        ####lag
        lag = 5
        t_lag_plot = np.linspace(1,(lag+1),(lag+1))
        t_lag=np.linspace(0,1,500)
        y0_simu_lag = S0_simu,I0_simu,R0_simu
        beta_lag_begin = beta_pred[0]
        beta_lag_end = beta
        gamma_lag = gamma
        I_lag_end = []
        I_lag_end.append(I0_simu)
        for i in range(1,(lag+1)):
            beta_lag = beta_lag_begin - (i/lag)*(beta_lag_begin - beta_lag_end)
            ret_lag = odeint(deriv,y0_simu_lag,t_lag,args=(N,beta_lag,gamma_lag))
            S_lag,I_lag,R_lag = ret_lag.T
            I_lag_end.append(I_lag[-1])
            I0_simu_lag = I_lag[-1]
            R0_simu_lag = R_lag[-1]
            S0_simu_lag = N - I0_simu_lag - R0_simu_lag
            y0_simu_lag = S0_simu_lag,I0_simu_lag,R0_simu_lag
        #repeated code -- code cleaning
        gamma = gamma_pred[0]
        t = np.linspace(0, simulation_period, 500)

        if st.button('Start Simulation'):
            #Plot for simulation = 1 : please use the function
            S0_simu_2 = N - I_lag[-1]-R_lag[-1]
            y0_simu = S0_simu_2,I_lag[-1],R_lag[-1]
            ret_first_intervention = odeint(deriv, y0_simu, t, args=(N, beta_lag_end, gamma))

            S, I, R = ret_first_intervention.T
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(df2['Day']-1,df2['I'],color = 'y',lw = 2,label = 'Actual data')
            ax.plot(pred_data_date-1, pred_data['I'].iloc[0:(predict_period+2)], color = 'r',label = 'Predictive data')
            ax.plot((t_lag_plot + pred_data_date.iloc[-1]-2),I_lag_end,color = 'r',linestyle = 'dashed')
            ax.plot((t + (lag+1) + pred_data_date.iloc[-1]-2),I,color = 'r',linestyle='dashed',label = 'Simulation data')
            # change X axis to actual date instead of days
            begin = data['Date'].min()
            num_days = len(df2) + forecast_period + simulation_period + 5
            labels = list((begin + dt.timedelta(days=x)).strftime('%m-%d') for x in range(0,num_days,5))
            plt.xticks(list(range(0,num_days,5)), labels, rotation=45)
            ####
            legend = ax.legend()
            for spine in ('top', 'right', 'bottom', 'left'):
                ax.spines[spine].set_visible(False)
            plt.show()
            st.pyplot()
            D_t = 500 / simulation_period
            R0_simu = f'**Effective reproduction number(R0) of the begin of simulation: ** **{(beta / gamma):.2f}**'
            st.markdown(R0_simu)

            st.write('Susceptible cases by the end of simulation will be ',int(S[499]))
            st.write('Recovered cases by the end of simulation will be ',int(R[499]))

            I_int = int(I[499])
            st.write('Infected cases by the end of simulation will be ',I_int)

            D_int =int(deaths *(int(I[499])+int(R[499]))/100)
            deaths_mark = f'**Total Death cases by the end of simulation will be ** **{D_int} **'
            st.markdown(deaths_mark)
            #st.write('Total Death cases by the end of simulation will be ',D_int)
            #plotting_SIR_Recovery(plot_S, plot_I, plot_R,N, t,simulation_period)
    else:
        st.title('Explore County Level Data ')
        # load data
        statesselected = st.selectbox("Select a County", countydf['State'])
        countylist=(countydf[countydf['State']==statesselected]['County']).tolist()[0]
        countyselected = st.selectbox('Select a county for demo',countylist)
        name=countyselected+', '+statesselected.strip()+', '+'US'
        df=data_cleaning(data,name)



        # drawing
        base = alt.Chart(df).mark_bar().encode( x='monthdate(Date):O',).properties(width=500)
        red = alt.value('#f54242')
        a = base.encode(y='Confirmed').properties(title='Total Confirmed')
        st.altair_chart(a,use_container_width=True)
        b = base.encode(y='Deaths', color=red).properties(title='Total Deaths')
        st.altair_chart(b,use_container_width=True)



        c = base.encode(y='New Cases').properties(title='Daily New Cases')
        st.altair_chart(c,use_container_width=True)


        d = base.encode(y='New deaths', color=red).properties(title='Daily New Deaths')
        st.altair_chart(d,use_container_width=True)

        dates=df['Date'].dt.date.unique()

        selected_date = st.selectbox('Select a Date to Start',(dates))
        forecastdf=df[df['Date'].dt.date >=selected_date]

        if st.checkbox('Show Raw Data'):

            st.write(forecastdf)

        if st.checkbox('Visualization Chart'):
            df_temp = forecastdf.rename(columns = {'I':'Active Infection Cases','R':'Recovered Cases'})
            e = pd.melt(frame = df_temp,

                        id_vars='Date',

                        value_vars=['Active Infection Cases','Recovered Cases'],

                        var_name = 'type',

                        value_name = 'count')

            e = alt.Chart(e).mark_area().encode(

                x=alt.X('Date:T', title='Date'),

                y=alt.Y('count:Q',title = 'Number of Cases'),

                color = alt.Color('type:O',legend = alt.Legend(title = None,orient = 'top-left'))

            ).configure_axis(
                grid=False

            )

            st.altair_chart(e, use_container_width=True)

    st.title("About")
    st.info(
            "This app uses JHU data available in [Github]"

            "(https://github.com/CSSEGISandData/COVID-19) repository.\n\n")

main()