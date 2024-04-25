import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler


model = joblib.load('rf_regressor.pkl')


st.title("Random Forest Regressor")

# Input fields
inputs = {}
for column in ['% Iron Feed', '% Silica Feed', 'Starch Flow', 'Amina Flow', 'Ore Pulp Flow',
               'Ore Pulp pH', 'Ore Pulp Density', 'Flotation Column 01 Air Flow',
               'Flotation Column 02 Air Flow', 'Flotation Column 03 Air Flow',
               'Flotation Column 04 Air Flow', 'Flotation Column 05 Air Flow',
               'Flotation Column 06 Air Flow', 'Flotation Column 07 Air Flow',
               'Flotation Column 01 Level', 'Flotation Column 02 Level',
               'Flotation Column 03 Level', 'Flotation Column 04 Level',
               'Flotation Column 05 Level', 'Flotation Column 06 Level',
               'Flotation Column 07 Level']:
    inputs[column] = st.number_input(f"{column}", step=1e-3)


if st.button("Predict"):
    data = pd.DataFrame(inputs, index=[0])
    X = data[['% Iron Feed', '% Silica Feed', 'Starch Flow', 'Amina Flow', 'Ore Pulp Flow',
              'Ore Pulp pH', 'Ore Pulp Density', 'Flotation Column 01 Air Flow',
              'Flotation Column 02 Air Flow', 'Flotation Column 03 Air Flow',
              'Flotation Column 04 Air Flow', 'Flotation Column 05 Air Flow',
              'Flotation Column 06 Air Flow', 'Flotation Column 07 Air Flow',
              'Flotation Column 01 Level', 'Flotation Column 02 Level',
              'Flotation Column 03 Level', 'Flotation Column 04 Level',
              'Flotation Column 05 Level', 'Flotation Column 06 Level',
              'Flotation Column 07 Level']]

    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    predictions = model.predict(X_scaled)
    predictions_df = pd.DataFrame(predictions, columns=['% Iron Concentrate', '% Silica Concentrate'])
    st.write(predictions_df)