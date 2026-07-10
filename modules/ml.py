import pandas as pd
from joblib import load
import shap
import numpy as np
import pandas as pd
import shap
from joblib import load
import streamlit as st

@st.cache_resource 
def load_model_and_explainer(bundle_path="data/car_price_bundle.joblib"):
    """
    Loads the model bundle and initializes the SHAP explainer ONCE.
    """
    bundle = load(bundle_path)
    lmodel = bundle["model"]
    explainer = shap.TreeExplainer(lmodel)
    return bundle, explainer

def get_price_prediction(df_features, bundle):
    """
    Runs LightGBM prediction to output the fair market price.
    """
    lmodel = bundle["model"]
    predicted_price = lmodel.predict(df_features)[0]
    
    return predicted_price


FEATURE_MAPPING = {
    "myear": "Manufacturing Year",
    "tt": "Transmission Type",
    "displacement": "Engine Displacement (cc)",
    "length": "Car Length (mm)",
    "width": "Car Width (mm)",
    "kerb_weight": "Kerb Weight (kg)",
    "turning_radius": "Turning Radius (m)",
    "top_speed": "Top Speed (kmph)",
    "acceleration": "Acceleration (0-100 kmph time)",
    "alloy_wheel_size": "Alloy Wheel Size (inches)",
    "gear_count": "Number of Gears",
    "power_bhp": "Engine Power (bhp)",
    "power_rpm": "Power at RPM",
    "torque_nm": "Engine Torque (Nm)",
    "km_driven": "Kilometers Driven",
    "owner_type": "Ownership History"
}
def extract_shap_insights(df_features, explainer):
    # Computes SHAP values and extracts both the actual values and the price impact.
    shap_values = explainer.shap_values(df_features)[0]
    feature_names = df_features.columns.tolist()
    
    shap_dict = dict(zip(feature_names, shap_values))
    sorted_features = sorted(shap_dict.items(), key=lambda x: x[1])
    
    top_negative = sorted_features[:3]
    top_positive = sorted_features[-3:]
    
    # Construct Positives: Feature Name (Actual Value) [Impact]
    positives = []
    for k, v in top_positive:
        if v > 0:
            human_name = FEATURE_MAPPING.get(k, k)
            actual_value = df_features[k].iloc[0]
            positives.append(f"{human_name} is {actual_value} [Added ₹{int(v):,}]")
            
    # Construct Negatives: Feature Name (Actual Value) [Impact]
    negatives = []
    for k, v in top_negative:
        if v < 0:
            human_name = FEATURE_MAPPING.get(k, k)
            actual_value = df_features[k].iloc[0]
            negatives.append(f"{human_name} is {actual_value} [Reduced ₹{abs(int(v)):,}]")
    

    return positives,negatives