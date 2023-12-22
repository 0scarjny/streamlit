# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import joblib
import numpy as np


# Set general properties for our app
st.set_page_config(
    page_title= "Car Price Prediction App",
    page_icon = "🏎️",
    layout="wide"
    )

# Function to load the dataset and model, cached to improve load times
@st.cache_data
def load_data():
    file_path = 'data_all_cars_clean.csv'  
    return pd.read_csv(file_path, delimiter=';', on_bad_lines= "skip")

data1 = load_data()

@st.cache_data
def load_model():
    model_path = 'predictions.sav' 
    return joblib.load(model_path)

model = load_model()

# Main page layout and navigation

def home():
    st.image('caradvisor.jpg')

    st.title("AutoAdvisor: Pricing & Recommendations")
    st.write("Welcome to the AutoAdvisor! Choose an option below:")
    
    # Create links to the car price prediction and recommendation pages
    with st.expander("Car Price Prediction"):
        st.markdown("<a name='price-prediction'></a>", unsafe_allow_html=True)
        car_price_prediction()
    
    with st.expander("Car Recommendation"):
        st.markdown("<a name='car-recommendation'></a>", unsafe_allow_html=True)
        car_recommendation()



# Define the car price prediction page
def car_price_prediction():
    st.header("Car Features")

    # Extract unique options from the dataset
    unique_models = data1["model"].unique()
    transmission_options = data1["Transmission"].unique()
    type_options = data1["type"].unique()
    fuel_options = data1["Fuel"].unique()
    brand_options = data1["brand"].unique()

    # Initialize session states for inputs
    if 'selected_model' not in st.session_state:
        st.session_state['selected_model'] = unique_models[0]
    if 'mileage' not in st.session_state:
        st.session_state['mileage'] = 0
    if 'horsepower' not in st.session_state:
        st.session_state['horsepower'] = 0
    if 'selected_brand' not in st.session_state:
        st.session_state['selected_brand'] = brand_options[0]
    if 'selected_transmission' not in st.session_state:
        st.session_state['selected_transmission'] = transmission_options[0]
    if 'selected_type' not in st.session_state:
        st.session_state['selected_type'] = type_options[0]
    if 'selected_fuel' not in st.session_state:
        st.session_state['selected_fuel'] = fuel_options[0]
    if 'year' not in st.session_state:
        st.session_state['year'] = 1950

    # Introducing columns for user inputs
    col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)

    selected_brand = col1.selectbox("Brand", brand_options, index=list(brand_options).index(st.session_state['selected_brand']))
    st.session_state['selected_brand'] = selected_brand

    # Filter models based on selected brand
    filtered_models = data1[data1["brand"] == selected_brand]["model"].unique()
    
    selected_model = col2.selectbox("Model", filtered_models, index=0)
    st.session_state['selected_model'] = selected_model

    mileage = col3.number_input("Mileage", min_value=0, value=st.session_state['mileage'])
    st.session_state['mileage'] = mileage

    horsepower = col4.number_input("Horsepower", min_value=0, value=st.session_state['horsepower'])
    st.session_state['horsepower'] = horsepower

    selected_transmission = col5.selectbox("Transmission", transmission_options, index=list(transmission_options).index(st.session_state['selected_transmission']))
    st.session_state['selected_transmission'] = selected_transmission

    selected_type = col6.selectbox("Type", type_options, index=list(type_options).index(st.session_state['selected_type']))
    st.session_state['selected_type'] = selected_type

    selected_fuel = col7.selectbox("Fuel", fuel_options, index=list(fuel_options).index(st.session_state['selected_fuel']))
    st.session_state['selected_fuel'] = selected_fuel

    year = col8.slider("Production Year", min_value=1950, max_value=2023, value=st.session_state['year'])
    st.session_state['year'] = year
    
    
    
    #Button to predict price
    if st.button('Predict Price'):
        #Prepare numeric features for prediction
        numeric_data = pd.DataFrame([[mileage, horsepower, year, 0]], 
                                    columns=['Mileage', 'Horsepower', 'year', 'used'])

        #Manually create one-hot encoded columns based on the dataset's unique values

        fuel_columns = [f'Fuel_{fuel}' for fuel in data1['Fuel'].unique()]
        transmission_columns = [f'Transmission_{trans}' for trans in data1['Transmission'].unique()]
        brand_columns = [f'brand_{brand}' for brand in data1['brand'].unique()]
        model_columns = [f'model_{model}' for model in data1['model'].unique()]
        type_columns = [f'type_{type}' for type in data1['type'].unique()]

        one_hot_columns = fuel_columns + transmission_columns + brand_columns + model_columns + type_columns

        encoded_features = pd.DataFrame(np.zeros((1, len(one_hot_columns))), columns=one_hot_columns)

        encoded_features[f'Fuel_{selected_fuel}'] = 1
        encoded_features[f'Transmission_{selected_transmission}'] = 1
        encoded_features[f'brand_{selected_brand}'] = 1
        encoded_features[f'model_{selected_model}'] = 1
        encoded_features[f'type_{selected_type}'] = 1

        input_data_encoded = pd.concat([numeric_data, encoded_features], axis=1)

        #Predict the price
        predicted_price = model.predict(input_data_encoded)[0]
        lower_bound = int(predicted_price // 1000) * 1000  #Round down to nearest thousand
        upper_bound = int((predicted_price + 1000) // 1000) * 1000  #Round up to nearest thousand

        st.write(f"Your car price is between {lower_bound} and {upper_bound}")

def format_datatable(df):
    return df.style.format({
        'price': '{:.0f}',
        'Horsepower': '{:.0f}',
        'Mileage': '{:,.0f} km',
        'car_age': '{} years'
    }).set_properties(**{
        'background-color': 'white',
        'color': 'black',
        'border-color': 'black'
    })


# Define the page for car recommendation
def car_recommendation():
    
    st.title("Car Recommendation")
    
    st.write("Select the features you want on your new car.")

    unique_models = data1["model"].unique()
    transmission_options = data1["Transmission"].unique()
    type_options = data1["type"].unique()
    fuel_options = data1["Fuel"].unique()
    brand_options = data1["brand"].unique()
    
    col_hp, col_mileage, col_year = st.columns(3)
    with col_hp:
         hp = col_hp.slider("How powerful do you want your car to be?",
                            data1["Horsepower"].min(),
                            800,            #not using .max() for better readability
                            (10, 250))

    with col_mileage:
         mileage = col_mileage.slider("How many kilometers do you want your car to have done?",
                                      data1["Mileage"].min(),
                                      800000,       #same purpose here
                                      (0, 100000))
    with col_year:
         age = col_year.slider("How many years do you want your car to have?",
                                      0,
                                      data1["car_age"].max(),
                                      (0,10))

    #Creating columns for brand, model, transmission, type, and fuel selections
    col1, col2, col3, col4, col5 = st.columns(5)
     
    selected_brands = st.multiselect("Select Brands", brand_options)

    #Dynamically update model selection based on selected brands
    if selected_brands:
        filtered_models = data1[data1["brand"].isin(selected_brands)]["model"].unique()
    else:
        filtered_models = data1["model"].unique()

    selected_models = st.multiselect("Select Model", filtered_models)

    selected_transmissions = st.multiselect("Select Transmission Types", transmission_options)

    selected_types = st.multiselect("Select Vehicle Types", type_options)

    selected_fuels = st.multiselect("Select Fuel Types", fuel_options)

    #Recommendation
    conditions = []
    if selected_brands:
        conditions.append(data1["brand"].isin(selected_brands))
    if selected_transmissions:
        conditions.append(data1["Transmission"].isin(selected_transmissions))
    if selected_types:
        conditions.append(data1["type"].isin(selected_types))
    if selected_fuels:
        conditions.append(data1["Fuel"].isin(selected_fuels))

    conditions.append((data1["Horsepower"] >= hp[0]) & (data1["Horsepower"] <= hp[1]))
    conditions.append((data1["Mileage"] >= mileage[0]) & (data1["Mileage"] <= mileage[1]))
    conditions.append((data1["car_age"] >= age[0]) & (data1["car_age"] <= age[1]))

    #Combine conditions
    if conditions:
        combined_conditions = conditions[0]
        for condition in conditions[1:]:
            combined_conditions &= condition
        Recommendation_data = data1.loc[combined_conditions, :]
    else:
    # If no specific filters are selected, show all data
        Recommendation_data = data1

    #Display the recommendations
    if st.checkbox("Show Recommendations", False):
        st.subheader("Recommendation of Cars")

        display_data = Recommendation_data.drop(['used', 'oldtimer'], axis=1)

        num_recommendations = len(display_data)
        if num_recommendations > 0:
            st.write(f"Number of cars matching your criteria: {num_recommendations}")

            # Convert DataFrame to HTML
            html = display_data.to_html(escape=False, index=False)
            st.markdown(html, unsafe_allow_html=True)
        else:
            st.write("Unfortunately, there are no cars corresponding to your criteria at the moment.")

            
  

# Create our app
def main():
    home()

# Main function to run our app
if __name__ == "__main__":
    main()



