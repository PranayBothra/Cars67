import streamlit as st

def render_car_selector(hierarchy, key_suffix="main"):
    cols1 = st.columns(3)
    oems = sorted(hierarchy.keys())
    oem = cols1[0].selectbox("OEM", oems, key=f"oem_{key_suffix}")
    models = sorted(hierarchy[oem].keys()) if oem else []
    model = cols1[1].selectbox("Model", models, key=f"model_{key_suffix}")
    variants = sorted(hierarchy[oem][model].keys()) if model else []
    variant = cols1[2].selectbox("Variant", variants, key=f"variant_{key_suffix}")

    cols2 = st.columns(3)
    years = sorted(hierarchy[oem][model][variant], reverse=True) if variant else []
    myear = cols2[0].selectbox("Manufacturing Year", years, key=f"year_{key_suffix}")
    km_driven = cols2[1].number_input("KM Driven", min_value=0, value=20000, step=1000, key=f"km_{key_suffix}")
    owner_type = cols2[2].selectbox("Owner type of the car", ['first', 'second', 'third', 'fourth'], key=f"owner_{key_suffix}")

    return oem, model, variant, myear, km_driven, owner_type