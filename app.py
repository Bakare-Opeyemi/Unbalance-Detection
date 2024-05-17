import pandas as pd
import numpy as np
import streamlit as st
import pickle

@st.cache_data
def load_models():
    """
    Load models
    """
    pickle_file_path = 'models/Random_forest.pkl'

    # Open the pickle file in read-binary mode
    with open(pickle_file_path, 'rb') as file:
        # Load the model from the pickle file
        model = pickle.load(file)

    return model

model = load_models()
decoder = {0:"balanced", 1:"unbalanced" }
cols=['V_in', 'Measured_RPM', 'Vibration_1', 'Vibration_2', 'Vibration_3']

def main(): 
    st.title("Unbalance Detector")
    html_temp = """
    <div style="background:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Unbalance Detection in a Rotating Shaft</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html = True)
    st.divider()

    html_temp2 = """
    <div style="background:#14302f ;padding:10px">
    <p style="color:white;text-align:left;">This project aims to identify the unbalanced state of a rotating shaft based on its vibration parameters</p>
    </div>
    """
    st.markdown(html_temp2, unsafe_allow_html = True)
    st.divider()

    voltage =  st.number_input("The input voltage to the motor controller V_in (in V)")
    measured_rpm =  st.number_input("The rotation speed of the motor (in RPM)")
    Vibration_1  =  st.number_input("The signal from the first vibration sensor")
    Vibration_2  =  st.number_input("The signal from the second vibration sensor")
    Vibration_3  = st.number_input("The signal from the third vibration sensor")

    if st.button("Predict"):
        
        data = {'V_in':voltage, 'Measured_RPM':measured_rpm, 
                'Vibration_1':Vibration_1, 'Vibration_2':Vibration_2, 'Vibration_3':Vibration_3, }
        df = pd.DataFrame([list(data.values())], columns = cols)
        prediction = model.predict(df)
        probability = max(model.predict_proba(df)[0])*100

        st.success('The rotating shaft is estimated to be '+ decoder[prediction[0]] + '. I have '+ str(probability) + '% confidence in this prediction')


    st.divider()
    footer= """
    <div style="background:#1e2f5c; padding:10px; text-align: center;">
    <p>Developed with  ❤  by the <a style='text-align: center; text-decoration: none;' href="https://www.linkedin.com/company/engineering-fit-org/" target=""> Engineering.fit( ) </a> Edison Research Lab</p>
    </div>
    """
    
    st.markdown(footer,unsafe_allow_html=True)



    

if __name__=='__main__': 
    main()