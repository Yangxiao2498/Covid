import streamlit as st
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
#import seaborn as sns; sns.set()
from helpers import *
from SIR_Model import *
from scipy.integrate import odeint
import matplotlib.animation as animation
from plotSimulation import *
from datacleaning import *
import datetime as dt
from statistics import mean
from sklearn.linear_model import Ridge,Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import seaborn as sns
from scipy import stats
from CovidModels import *
from CallBack import *


def main():

    ## sidebar
    dferr = pd.read_csv('errorlist.csv')
    error_list = list(dferr['errorlist'])
    # Adding new changes
    # Load data
    data = load_data('total_data.pkl')

    #set predicted period pp
    pp = 14

    data['Date'] = pd.to_datetime(data['Date'])
    data1 = data[data['Date'] == data['Date'].iloc[0]]
    countydf=pd.DataFrame(data1.Combined_Key.str.split(',',2).tolist(),columns = ['County','State','Country']).drop('Country',axis=1)
    countydf=pd.DataFrame(countydf.groupby('State')['County'].apply(lambda x: x.values.tolist())).reset_index()
    statelist = countydf['State'].tolist()

    last_day = data['Date'].max()
    last_day_formatted = last_day.date().strftime('%b %d %Y')

    # ---removed unrelated states
    # --add DC back to our list
    removed_state =[' US', ' Wuhan Evacuee',  'Washington', ' District of Columbia']
    for i in removed_state:
        statelist.remove(i)

    ## move DC position of the statelist
    statelist[-1] = ' District of Columbia'
    statelist = sorted(statelist)

    # Created new drop down menu
    statedropdown = [
        {"label": str(statelist[i]), "value": str(statelist[i])}
        for i in range(len(statelist))
    ]
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to",

                            ( 'Hybrid Model','Data Exploratory'))


    if page == 'Hybrid Model' :
        data = load_data('total_data.pkl')
        #Adding new changes -- gunjan
        st.title('Explore County Level Data ')

        statesselected = st.selectbox('Select a State',statelist)
        countylist = sorted(((countydf[countydf['State'] == statesselected]['County']).tolist()[0]), key=lambda s: s.strip('"'))

        # Remove duplicates in countylist
        countylist = sorted(list(set(countylist)))

        # Remove "Unassigned" county in countylist if exists
        countylist = [i for i in countylist if (i != 'Unassigned') & (i != 'Unknown')]

        countyselected = st.selectbox('Select a County',countylist)

        name = countyselected+', '+statesselected.strip()+', '+'US'
        if name in error_list:
            st.write(name,'is in the error list')
            pass
        else:
            df2 = data_cleaning(data,name)
            #set 14 days back
            df = df2[df2['Date'] < df2.Date.iloc[-14]]

            date=df2['Date'].dt.date.max()
            currentdf=df2[df2['Date'].dt.date ==date]
            pastdate = date-timedelta(days=30)
            pastdf=df2[df2['Date'].dt.date >= pastdate].reset_index(drop=True)
            #Show data for the date
            st.title('{}, {} on {}'.format(statesselected, countyselected, date))

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
            begin = df['Date'].max()


            json_obj = prediction(statesselected,countyselected)
            prediction_result = json.loads(json_obj)
            df = pd.read_json(prediction_result['df'])
            pred_data = pd.read_json(prediction_result['pred_data'])
            I_pred = pred_data['I']
            label_pred = list((begin + timedelta(days=x)) for x in range(0,14))
            label_pred_ = list(x.strftime('%m-%d') for x in label_pred)
            x = np.linspace(0,14,14)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            plt.plot(label_pred_,df2['I'][-14:],color = 'g',label = 'Actual last 14 days data')
            plt.plot(label_pred_,I_pred[1:],color = 'b',label = 'Predicted data')
            plt.xticks(list(range(0,14)), label_pred_, rotation=45, fontsize=12)
            legend = ax.legend()
            for spine in ('top', 'right', 'bottom', 'left'):
                ax.spines[spine].set_visible(False)
            #plt.show()
            st.pyplot()
            st.write('MSE',mean_squared_error(df2['I'][-14:], I_pred[1:]))

        ###############################################
        
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
