import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Load the API key from Streamlit secrets
API_KEY = st.secrets["GEMINI_API_KEY"]

# Configure an HTTP session with connection pooling and automatic retries
http_session = requests.Session()
retries = Retry(total=2, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
http_session.mount(
    'https://',
    HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=10)
)

def call_api(model_name, prompt_text):
    # Executes a Gemini API request with standardized generation settings.

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={API_KEY}"

    generation_config = {
        "temperature": 0.3,
        # Token limit selected to accommodate reasoning and the complete response
        "maxOutputTokens": 4096
    }

    payload = {
        "contents": [{"parts": [{"text": prompt_text}]}],
        "generationConfig": generation_config
    }

    response = http_session.post(url, json=payload, timeout=20)
    return response.json()

def execute_model_cascade(tiers, fallback_text):
    # Executes a prioritized model cascade with automatic fallback on failures.

    for models, prompt in tiers:
        for model in models:
            try:
                data = call_api(model, prompt)

                if "candidates" in data:
                    return data["candidates"][0]["content"]["parts"][0]["text"]

                elif "error" in data:
                    err_msg = data["error"].get("message", "").lower()

                    # Retry with the next available model when quota limits are reached
                    if "quota" in err_msg or "limit" in err_msg or data["error"].get("code") == 429:
                        print(f"[Warning] Quota limit hit for {model}. Falling back...")
                        continue
                    else:
                        print(f"[Error] API returned error for {model}: {err_msg}")

            except Exception as e:
                # Log unexpected exceptions before attempting the next fallback model
                print(f"[Exception] Failed to call {model}: {str(e)}")
                continue

    return fallback_text

def get_price_explanation(car_details, predicted_price, shap_pos_list, shap_neg_list):
    # Generates an AI-powered explanation for the predicted vehicle valuation.

    # Primary models prioritized for analytical reasoning and structured explanations
    gemini_models = ["gemini-3.5-flash", "gemini-2.5-flash","gemini-3-flash", "gemini-3.1-flash-lite"]

    # Secondary models used as fallback if the primary tier is unavailable
    gemma_models = ["gemma-4-31b-it", "gemma-4-26b-a4b-it"]

    pos_str = ", ".join(shap_pos_list) if shap_pos_list else "Standard market features"
    neg_str = ", ".join(shap_neg_list) if shap_neg_list else "Standard wear and tear"
    formatted_price = f"₹{predicted_price:,.0f}"

    gemini_prompt = f"""
    You are an elite used car valuation expert.
    
    ### INPUT DATA
    <vehicle>
    Details: {car_details}
    Predicted Market Value: {formatted_price}
    Value Boosters (Positives): {pos_str}
    Value Reducers (Negatives): {neg_str}
    </vehicle>
    
    ### TASK
    Explain the AI's predicted price based ONLY on the input data.
    - Write exactly 2 concise paragraphs.
    - TONE: Professional, objective, and analytical. Act as a financial advisor.
    - FOCUS: Explain how the boosters justify the price and why the reducers pull it down.
    
    ### CRITICAL CONSTRAINTS
    - ABSOLUTELY NO EMOJIS in the output.
    - DO NOT output exact monetary values (₹) for individual boosters or reducers; describe their impact qualitatively.
    - DO NOT use technical jargon (e.g., "SHAP", "Algorithm", "Model").
    - DO NOT infer or assume specifications or ownership details not explicitly provided.
    """

    gemma_prompt = f"""
    You are a data-driven vehicle appraiser.

    ### INPUT DATA
    <vehicle>
    Details: {car_details}
    Predicted Market Value: {formatted_price}
    Value Boosters: {pos_str}
    Value Reducers: {neg_str}
    </vehicle>
    
    ### TASK
    Write exactly 3 bullet points explaining this valuation.
    - Bullet 1: State the final estimated value.
    - Bullet 2: Explain how the specific Value Boosters justify adding value to this price.
    - Bullet 3: Explain why the specific Value Reducers pull the price down.
    
    ### CRITICAL CONSTRAINTS
    - ABSOLUTELY NO EMOJIS in the output.
    - DO NOT write an introduction or conclusion.
    - DO NOT output exact rupee amounts (₹) for individual boosters/reducers.
    - DO NOT use technical ML jargon like SHAP.
    - DO NOT invent facts outside of the provided Input Data.
    """

    fallback = (
        f"Based on current market data, the estimated market value for this "
        f"{car_details} is **{formatted_price}**. This price is dynamically "
        f"calculated based on its specific age, mileage, and mechanical condition."
    )

    # Execute the primary-to-fallback model cascade
    return execute_model_cascade(
        [(gemini_models, gemini_prompt), (gemma_models, gemma_prompt)],
        fallback,
    )

def generate_car_comparison(car_a_name, car_a_specs, car_a_price, car_b_name, car_b_specs, car_b_price):
    # Generates an AI-driven comparative analysis for two configured vehicles.

    # Skip comparison when both vehicle configurations are identical
    if (car_a_name == car_b_name) and (car_a_price == car_b_price):
        return "### 🛑 Identical Vehicles Detected\nBoth selected vehicles have the exact same specifications, mileage, and predicted market value. There is no comparative analysis to perform."

    # Primary models prioritized for consistent Markdown formatting
    gemma_models = ["gemma-4-31b-it", "gemma-4-26b-a4b-it"]

    # Secondary models used as fallback if the primary tier is unavailable
    gemini_models = [ "gemini-3-flash", "gemini-3.5-flash","gemini-2.5-flash", "gemini-3.1-flash-lite"]
    
    prompt = f"""
    You are an objective and highly critical automotive advisor. 
    
    ### INPUT DATA
    <car_a>
    Vehicle: {car_a_name}
    Predicted Price: ₹{car_a_price:,.0f}
    Key Specs: {car_a_specs}
    </car_a>
    
    <car_b>
    Vehicle: {car_b_name}
    Predicted Price: ₹{car_b_price:,.0f}
    Key Specs: {car_b_specs}
    </car_b>
    
    ### TASK
    Compare these two configured used cars. Structure your response strictly in Markdown with these EXACT 3 sections:
    
    ### The Value Verdict
    (1 concise paragraph directly stating which car offers a better value-for-money ratio based on age, mileage, specs, and price. Be critical.)
    
    ### Target Buyer Profiles
    * **Buy Car A if:** (1 bullet point identifying the ideal buyer)
    * **Buy Car B if:** (1 bullet point identifying the ideal buyer)
    
    ### Technical Trade-offs
    (2-3 bullet points comparing their mechanical specs like engine power or torque.)
    
    ### CRITICAL CONSTRAINTS
    - ABSOLUTELY NO EMOJIS in the output.
    - DO NOT hallucinate specs. If a specification is unavailable, state that it cannot be compared.
    - DO NOT include any introductory greetings or concluding remarks.
    """
    
    fallback = "Comparison engine is currently overloaded. Please check back in a few minutes or analyze the predicted prices directly."

    # Execute the primary-to-fallback model cascade
    return execute_model_cascade([(gemma_models, prompt), (gemini_models, prompt)], fallback)

def generate_inspection_checklist(car_details, age, mileage):
    #Generates an AI-powered inspection checklist tailored to the vehicle's condition.

    # Primary models prioritized for structured checklist generation
    gemma_models = ["gemma-4-31b-it", "gemma-4-26b-a4b-it"]

    # Secondary models used as fallback if the primary tier is unavailable
    gemini_models = ["gemini-3.1-flash-lite", "gemini-2.5-flash-lite"]
    
    prompt = f"""
    You are a strict and expert automotive mechanic.
    
    ### INPUT DATA
    <vehicle>
    Details: {car_details}
    Age: {age} years old
    Mileage: {mileage} km
    </vehicle>
    
    ### TASK
    Create a highly specific inspection checklist tailored to this exact used car's age and mileage.
    
    Structure your response strictly in Markdown with these EXACT 3 sections (do not alter these headers):
    
    ### Engine and Mechanical
    (Exactly 3 highly specific, actionable bullet points here)
    
    ### Electronics and Interior
    (Exactly 3 highly specific, actionable bullet points here)
    
    ### Structural and Exterior
    (Exactly 3 highly specific, actionable bullet points here)
    
    ### CRITICAL CONSTRAINTS
    - ABSOLUTELY NO EMOJIS anywhere in the output.
    - DO NOT include any introductory text or concluding remarks.
    - DO NOT invent specific known flaws for this model; focus purely on age/mileage wear-and-tear principles.
    """

    fallback = "Checklist generator is currently overloaded. Please ensure you take a trusted mechanic with you to inspect the engine, structural integrity, and electronic systems."

    # Execute the primary-to-fallback model cascade
    return execute_model_cascade([(gemma_models, prompt), (gemini_models, prompt)], fallback)


def generate_vehicle_review(oem, model_name):
    # Generates an AI-powered summary of a vehicle's long-term market reputation.

    # Primary models prioritized for broader automotive knowledge
    gemini_models = ["gemini-3.1-flash-lite", "gemini-3-flash","gemini-2.5-flash-lite"]

    # Secondary models used as fallback if the primary tier is unavailable
    gemma_models = ["gemma-4-31b-it", "gemma-4-26b-a4b-it"]
    
    prompt = f"""
    You are a highly critical and objective automotive market analyst.
    
    ### INPUT DATA
    <vehicle_lineage>
    Make: {oem}
    Model: {model_name}
    </vehicle_lineage>
    
    ### TASK
    Provide a long-term market reputation summary. Structure your response strictly in Markdown with these EXACT 3 sections:
    
    ### Market Reputation
    (1 concise paragraph detailing how this car holds up over time, its resale value trends, and general public perception.)
    
    ### Key Strengths
    (3 specific bullet points highlighting its best attributes.)
    
    ### Known Flaws and Red Flags
    (3 specific bullet points detailing common mechanical failures or typical complaints.)
    
    ### CRITICAL CONSTRAINTS
    - ABSOLUTELY NO EMOJIS anywhere in the output.
    - Discuss ONLY issues that are commonly reported for this vehicle lineage.
    - If you are not highly confident about a specific part failure, describe it as a general category (e.g., 'suspension components') rather than inventing a specific issue.
    - DO NOT include any generic introductory or concluding remarks.
    """
    
    fallback = "Market reputation data is currently unavailable due to high server load."

    # Execute the primary-to-fallback model cascade
    return execute_model_cascade([(gemini_models, prompt), (gemma_models, prompt)], fallback)
