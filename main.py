# main.py

import streamlit as st
import numpy as np
import joblib

# -------------------------------
# Utility Functions for Loading Models
# -------------------------------

@st.cache_resource
def load_crop_objects():
    """
    Loads the crop prediction model and scaler.

    """
    crop_model = joblib.load("models/crop_model.pkl")
    crop_scaler = joblib.load("models/crop_scaler.pkl")
    return crop_model, crop_scaler

@st.cache_resource
def load_fertilizer_objects():
    """
    Loads the fertilizer prediction model, scaler, and target label encoder.
    
    """
    fert_model = joblib.load("models/fertilizer_model.pkl")
    fert_scaler = joblib.load("models/fertilizer_scaler.pkl")
    target_le = joblib.load("models/target_label_encoder.pkl")
    return fert_model, fert_scaler, target_le

# -------------------------------
# Prediction Functions
# -------------------------------

def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    """
    Predicts the best crop given soil nutrients and weather parameters.
    Uses the saved crop model and scaler.
    """
    model, scaler = load_crop_objects()
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    
    
    crop_dict = {
        1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
        8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
        14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
        19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
    }
    crop = crop_dict.get(prediction[0], "Unknown Crop")
    return crop

def predict_fertilizer(Temperature, Humidity, Moisture, Soil_Type, Crop_Type, Nitrogen, Potassium, Phosphorous):
    """
    Predicts the best fertilizer given environmental and soil conditions.
    Uses the saved fertilizer model, scaler, and target label encoder.
    """
    model, scaler, target_le = load_fertilizer_objects()
    features = np.array([[Temperature, Humidity, Moisture, Soil_Type, Crop_Type, Nitrogen, Potassium, Phosphorous]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    fertilizer = target_le.inverse_transform(prediction)[0]
    return fertilizer

# -------------------------------
# Main Streamlit App
# -------------------------------

def main():
    st.title("Agricultural Recommendation System")
    
    # Sidebar to select recommendation type
    app_mode = st.sidebar.selectbox("Choose Recommendation Type", 
                                    ["Crop Recommendation", "Fertilizer Recommendation"])
    
    # -------------------------------
    # Crop Recommendation Section
    # -------------------------------
    if app_mode == "Crop Recommendation":
        st.header("Crop Recommendation")
        st.markdown("Enter the soil nutrient and weather parameters to get the recommended crop.")
        
        # Use session_state to prepopulate inputs if they were entered before.
        N = st.number_input("Nitrogen (N)", min_value=0, max_value=200, 
                            value=st.session_state.get("crop_N", 100), key="crop_N_input")
        P = st.number_input("Phosphorous (P)", min_value=0, max_value=200, 
                            value=st.session_state.get("crop_P", 100), key="crop_P_input")
        K = st.number_input("Potassium (K)", min_value=0, max_value=200, 
                            value=st.session_state.get("crop_K", 100), key="crop_K_input")
        temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, 
                                      value=st.session_state.get("crop_temp", 25.0), key="crop_temp_input")
        humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, 
                                   value=st.session_state.get("crop_humid", 50.0), key="crop_humid_input")
        ph = st.number_input("pH", min_value=0.0, max_value=14.0, 
                             value=st.session_state.get("crop_ph", 7.0), key="crop_ph_input")
        rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, 
                                   value=st.session_state.get("crop_rainfall", 100.0), key="crop_rainfall_input")
        
        if st.button("Predict Crop"):
            result = predict_crop(N, P, K, temperature, humidity, ph, rainfall)
            # Save input values and predicted crop in session state
            st.session_state["crop_N"] = N
            st.session_state["crop_P"] = P
            st.session_state["crop_K"] = K
            st.session_state["crop_temp"] = temperature
            st.session_state["crop_humid"] = humidity
            st.session_state["crop_ph"] = ph
            st.session_state["crop_rainfall"] = rainfall
            st.session_state["predicted_crop"] = result
            st.success(f"Recommended Crop: **{result}**")
    
    # -------------------------------
    # Fertilizer Recommendation Section
    # -------------------------------
    elif app_mode == "Fertilizer Recommendation":
        st.header("Fertilizer Recommendation")
        st.markdown("Enter the environmental and soil conditions to get the recommended fertilizer. "
                    "Common fields are pre-filled from your crop recommendation, if available.")
        
        # Prepopulate Temperature, Humidity, Nitrogen, and Potassium from crop recommendation if available.
        Temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, 
                                        value=st.session_state.get("crop_temp", 25.0), key="temp_fert")
        Humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, 
                                     value=st.session_state.get("crop_humid", 50.0), key="humid_fert")
        
        Moisture = st.number_input("Moisture (ratio 0-1)", min_value=0.0, max_value=1.0, 
                                   value=0.5, key="moist_fert", step=0.1,
                                   help="Enter a value between 0 and 1 (e.g., 0.5 represents moderate moisture).")
        Nitrogen = st.number_input("Nitrogen (N)", min_value=0, max_value=100, 
                                   value=st.session_state.get("crop_N", 50), key="N_fert")
        Potassium = st.number_input("Potassium (K)", min_value=0, max_value=100, 
                                    value=st.session_state.get("crop_K", 50), key="K_fert")
        Phosphorous = st.number_input("Phosphorous", min_value=0, max_value=100, 
                                      value=50, key="P_fert")
        
        
        Soil_Type_options = ["Sandy", "Loamy", "Black", "Red", "Clayey"]
        Soil_Type = st.selectbox("Soil Type", options=Soil_Type_options, key="soil_fert")
        
        # Define a mapping from predicted crop (from crop recommendation) to fertilizer crop type.
        # For fruits (e.g., muskmelon, mango, orange, apple, papaya, banana, grapes, pomegranate),
        # the default is set to "Pulses" because the fertilizer model is trained on specific crop types.
        default_fert_crop_mapping = {
            "rice": "Paddy",
            "maize": "Maize",
            "jute": "Oil seeds",
            "cotton": "Cotton",
            "coconut": "Pulses",
            "papaya": "Pulses",
            "orange": "Pulses",
            "apple": "Pulses",
            "muskmelon": "Pulses",
            "watermelon": "Pulses",
            "grapes": "Pulses",
            "mango": "Pulses",
            "banana": "Pulses",
            "pomegranate": "Pulses",
            "lentil": "Pulses",
            "blackgram": "Pulses",
            "mungbean": "Pulses",
            "mothbeans": "Pulses",
            "pigeonpeas": "Pulses",
            "kidneybeans": "Pulses",
            "chickpea": "Pulses",
            "coffee": "Cotton"
        }
        # List of valid crop type options used in fertilizer training.
        Crop_Type_options = ["Maize", "Sugarcane", "Cotton", "Tobacco", "Paddy", 
                             "Barley", "Wheat", "Millets", "Oil seeds", "Pulses", "Ground Nuts"]
        
        # Determine default crop type based on predicted crop.
        predicted_crop = st.session_state.get("predicted_crop", None)
        default_crop_type = None
        if predicted_crop:
            # Check if the recommended crop is a fruit.
            fruits = ["papaya", "orange", "apple", "muskmelon", "mango", "banana", "grapes", "pomegranate"]
            if predicted_crop.lower() in fruits:
                default_crop_type = "Pulses"
            else:
                default_crop_type = default_fert_crop_mapping.get(predicted_crop.lower(), None)
        if default_crop_type and default_crop_type in Crop_Type_options:
            default_crop_index = Crop_Type_options.index(default_crop_type)
        else:
            default_crop_index = 0
        Crop_Type = st.selectbox("Crop Type", options=Crop_Type_options, 
                                 index=default_crop_index, key="crop_fert",
                                 help="This is pre-selected based on your crop recommendation. Change if needed.Note: Pulses is default for all fruits.Oil seeds for jute and cottom")
        
    
        soil_mapping = {"Sandy": 0, "Loamy": 1, "Black": 2, "Red": 3, "Clayey": 4}
        crop_mapping = {"Maize": 0, "Sugarcane": 1, "Cotton": 2, "Tobacco": 3, "Paddy": 4, 
                        "Barley": 5, "Wheat": 6, "Millets": 7, "Oil seeds": 8, "Pulses": 9, "Ground Nuts": 10}
        encoded_soil = soil_mapping.get(Soil_Type, 0)
        encoded_crop = crop_mapping.get(Crop_Type, 0)
        
        if st.button("Predict Fertilizer"):
            result = predict_fertilizer(Temperature, Humidity, Moisture, encoded_soil, encoded_crop, Nitrogen, Potassium, Phosphorous)
            st.success(f"Recommended Fertilizer: **{result}**")

if __name__ == "__main__":
    main()
