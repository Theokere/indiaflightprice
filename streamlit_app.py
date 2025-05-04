# -*- coding: utf-8 -*-
"""
Created on Sat May  3 20:59:40 2025

@author: teoka
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import category_encoders as ce

loaded_model = pickle.load(open("trained_model.sav",'rb'))
loaded_encoder = pickle.load(open("trained_encoder.sav",'rb'))

columns = ['airline','source_city','departure_time','stops','arrival_time','destination_city','class','duration','days_left']


    

def main():
    
    st.title(':small_airplane: Predict Flight Prices in India Top 6 Metro Cities')
    st.markdown('(Kaggle) Data source collected from "Ease My Trip" website. A total of 300261 distinct flight booking options was extracted from the site. Data was collected for 50 days, from February 11th to March 31st, 2022.')
    st.markdown('MAE is around 9.52%')
    
    airline = st.selectbox('Select your airline',['SpiceJet', 'AirAsia', 'Vistara', 'GO_FIRST', 'Indigo',
       'Air_India'])
    
    source_city = st.selectbox('Select your city of origin',['Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai'])
    
    departure_time = st.selectbox('Select your time of departure',['Evening', 'Early_Morning', 'Morning', 'Afternoon', 'Night',
       'Late_Night'])
    
    stops = st.selectbox('Number of stops in transit?',['zero', 'one', 'two_or_more'])
    
    arrival_time = st.selectbox('Select yor arrival time period',['Night', 'Morning', 'Early_Morning', 'Afternoon', 'Evening',
       'Late_Night'])
    
    destination_city = st.selectbox('Select your city of destination',['Mumbai', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai', 'Delhi'])
    
    flight_class = st.selectbox('Economy or Business?',['Economy', 'Business'])
    
    duration = st.number_input('How long (hours) is the flight?',min_value = 1,max_value = 50)
    
    days_left = st.number_input('Ticket bought how many days before the flight',min_value = 1, max_value = 50)
    
    input_data = pd.DataFrame([[airline, source_city, departure_time, stops, arrival_time, destination_city,flight_class]],columns=['airline','source_city','departure_time','stops','arrival_time','destination_city','flight_class'])

    encoded = loaded_encoder.transform(input_data)
    
    to_add = pd.DataFrame([[duration,days_left]],columns = ['duration','days_left'])
    
    input_data = pd.concat([encoded,to_add],axis=1)
    
    price_predict = ""
    
    if st.button('Submit'):        
        price_predict = loaded_model.predict(input_data)
        price_predict = format(price_predict[0],",")
        
    st.success(price_predict)
    
    st.page_link('https//www.hdbpriceai.com/',label = 'Preduct Singapore HDB Price',icon = '🏠')

if __name__ == "__main__":
    main()
    
