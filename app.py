import pandas as pd
import streamlit as st
from modules.data_utils import load_and_parse_lookup
from modules.ui_utils import render_car_selector
from modules.ml import load_model_and_explainer, get_price_prediction, extract_shap_insights
from modules.ai import get_price_explanation, generate_inspection_checklist, generate_vehicle_review, generate_car_comparison

st.set_page_config(
    page_title="Cars67 | Cars 6×4 / 7",
    page_icon=":material/directions_car:",
    layout="wide"
)
st.title(":blue[:material/directions_car: Cars67] | Cars 6×4 / 7 ")

mode = st.segmented_control("Mode", ["Predict Price", "Compare Vehicles"], default="Predict Price")
lookup, hierarchy = load_and_parse_lookup()
bundle, explainer = load_model_and_explainer()

# Format user inputs into the model-ready feature dataframe
def format_df(y, km, own, specs):
    df = pd.DataFrame([{"myear": y, "km_driven": km, "owner_type": own, **specs}])[bundle["feature_names"]]
    for c in bundle["cat_cols"]: df[c] = pd.Categorical(df[c], categories=bundle["categories"][c])
    for c in [x for x in bundle["feature_names"] if x not in bundle["cat_cols"]]: df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

st.divider()

if mode == "Predict Price":
    with st.expander(":orange[:material/tune: Configure Vehicle Specifications]", expanded=True):
        oem, model, variant, myear, km_drive, owner_type = render_car_selector(hierarchy, "run")
        
    if st.button("Predict Market Price", type="primary", use_container_width=True):
        car_base = f"{oem} {model} {variant}"
        df_features = format_df(myear, km_drive, owner_type, lookup.get(f"{oem}|{model}|{variant}|{myear}", {}))
        
        st.session_state.update({
            'price': get_price_prediction(df_features, bundle), 'df': df_features,
            'base': car_base, 'yr': myear, 'km': km_drive, 'oem': oem, 'mod': model
        })
        for k in ['exp', 'chk', 'rev']: st.session_state.pop(k, None)

    if 'price' in st.session_state:
        st.subheader(":green[:material/analytics: Analysis Results]")
        
        with st.container(border=True):
            st.markdown("####  :green[:material/payments: Predicted Price]")
            st.metric(f"{st.session_state['yr']} {st.session_state['base']}", f"₹{st.session_state['price']:,.0f}")
            
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader(":orange[:material/memory: AI Insights]")
        
        # AI-generated valuation rationale
        with st.container(border=True):
            st.markdown("####  :blue[:material/memory: Comprehensive AI Insights]")
            if 'exp' not in st.session_state:
                if st.button(":material/psychology: Price Explanation", use_container_width=True):
                    with st.spinner("Analyzing parameters..."):
                        pos, neg = extract_shap_insights(st.session_state['df'], explainer)
                        st.session_state['exp'] = get_price_explanation(f"{st.session_state['yr']} {st.session_state['base']}", st.session_state['price'], pos, neg)
                    st.rerun()
            if 'exp' in st.session_state:
                with st.expander(":blue[:material/psychology: Price Explanation]", expanded=True):
                    st.write(st.session_state['exp'])
                    
            # AI-generated pre-purchase inspection checklist
            if 'chk' not in st.session_state:
                if st.button(":material/fact_check: Inspection Checklist", use_container_width=True):
                    with st.spinner("Compiling checks..."):
                        st.session_state['chk'] = generate_inspection_checklist(st.session_state['base'], 2026 - int(st.session_state['yr']), st.session_state['km'])
                    st.rerun()
            if 'chk' in st.session_state:
                with st.expander(":orange[:material/fact_check: Inspection Checklist]", expanded=True):
                    st.write(st.session_state['chk'])
                    
            # AI-generated market reputation summary
            if 'rev' not in st.session_state:
                if st.button(":material/monitoring: Vehicle Review", use_container_width=True):
                    with st.spinner("Fetching data..."):
                        st.session_state['rev'] = generate_vehicle_review(st.session_state['oem'], st.session_state['mod'])
                    st.rerun()
            if 'rev' in st.session_state:
                with st.expander(":violet[:material/monitoring: Vehicle Review]", expanded=True):
                    st.write(st.session_state['rev'])

elif mode == "Compare Vehicles":
    with st.expander(":orange[:material/compare_arrows: Configure Vehicles]", expanded=True):
        c1, c2 = st.columns(2)
        with c1: st.markdown(":blue[**Vehicle A**]"); oa, ma, va, ya, ka, ow_a = render_car_selector(hierarchy, "a")
        with c2: st.markdown(":green[**Vehicle B**]"); ob, mb, vb, yb, kb, ow_b = render_car_selector(hierarchy, "b")

    if st.button("Predict & Compare", type="primary", use_container_width=True):
        sp_a, sp_b = lookup[f"{oa}|{ma}|{va}|{ya}"], lookup[f"{ob}|{mb}|{vb}|{yb}"]
        st.session_state.update({
            'p_a': get_price_prediction(format_df(ya, ka, ow_a, sp_a), bundle),
            'p_b': get_price_prediction(format_df(yb, kb, ow_b, sp_b), bundle),
            'sp_a': sp_a, 'sp_b': sp_b, 'oa': oa, 'ma': ma, 'ob': ob, 'mb': mb,
            'ya': ya, 'ka': ka, 'yb': yb, 'kb': kb
        })
        st.session_state.pop('cmp_rev', None)

    if 'p_a' in st.session_state:
        c1, c2 = st.columns(2)
        with c1:
            with st.container(border=True):
                st.markdown("####  :green[:material/directions_car: Vehicle A Value]")
                st.metric(f"{st.session_state['oa']} {st.session_state['ma']}", f"₹{st.session_state['p_a']:,.0f}")
        with c2:
            with st.container(border=True):
                st.markdown("####  :green[:material/directions_car: Vehicle B Value]")
                st.metric(f"{st.session_state['ob']} {st.session_state['mb']}", f"₹{st.session_state['p_b']:,.0f}")
                
        with st.container(border=True):
            st.markdown("####  :blue[:material/balance: AI Comparative Verdict]")
            # AI-generated side-by-side vehicle comparison
            if 'cmp_rev' not in st.session_state:
                if st.button("Generate Comparison Report"):
                    with st.spinner("Generating comparative analysis..."):
                        keys = ["power_bhp", "torque_nm", "displacement", "length"]
                        st.session_state['cmp_rev'] = generate_car_comparison(
                            f"{st.session_state['ya']} {st.session_state['oa']} {st.session_state['ma']} ({st.session_state['ka']} km)", 
                            {k: st.session_state['sp_a'][k] for k in keys if k in st.session_state['sp_a']}, st.session_state['p_a'],
                            f"{st.session_state['yb']} {st.session_state['ob']} {st.session_state['mb']} ({st.session_state['kb']} km)", 
                            {k: st.session_state['sp_b'][k] for k in keys if k in st.session_state['sp_b']}, st.session_state['p_b']
                        )
                    st.rerun()
            else:
                st.write(st.session_state['cmp_rev'])