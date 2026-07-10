import json
import streamlit as st
@st.cache_data
def load_and_parse_lookup():

    #Reads the JSON once and builds an O(1) lookup hierarchy.
    with open("data/variant_specs.json", "r", encoding="utf-8") as f:
        lookup = json.load(f)
    
    # Build a nested dictionary: hierarchy[oem][model][variant] = {years}
    hierarchy = {}
    
    for key in lookup.keys():
        oem, model, variant, year = key.split("|")
        year = int(year)
        
        # Initialize nested levels if they don't exist
        if oem not in hierarchy:
            hierarchy[oem] = {}
        if model not in hierarchy[oem]:
            hierarchy[oem][model] = {}
        if variant not in hierarchy[oem][model]:
            hierarchy[oem][model][variant] = set()
            
        hierarchy[oem][model][variant].add(year)
        
    return lookup, hierarchy