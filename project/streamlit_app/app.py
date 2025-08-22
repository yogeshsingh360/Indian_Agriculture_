import streamlit as st
import pandas as pd
from Helper import helper as hp
import numpy as np

# --- Initialize session state for 'submit' ---
if "submit" not in st.session_state:
    st.session_state.submit = False

# --- Main UI Function ---
def show_input_form():
    st.title("🌾 Crop Yield Prediction")

    # --- State & District Selection ---
    col1, col2 = st.columns(2)
    with col1:
        st.selectbox(
            "Select State",
            hp.options()["state_list"],
            key="state",
            placeholder="Select State",
            help="Select the state for which you want to predict crop yield.",
            index=None
        )

    selected_state = st.session_state.get("state")
    with col2:
        st.selectbox(
            "Select District",
            hp.options()["district_dict"].get(selected_state, []),
            key="district",
            placeholder="Select District",
            help="Select the district for which you want to predict crop yield.",
            index=None
        )

    # --- Year & Season Selection ---
    col3, col4 = st.columns(2)
    with col3:
        st.selectbox(
            "Select Year",
            list(range(2020, 2031)),
            key="year",
            placeholder="Select Year",
            help="Select the year for prediction.",
            index=None
        )
    with col4:
        st.selectbox(
            "Select Season",
            hp.options()["season_list"],
            key="season",
            placeholder="Select Season",
            help="Select the season for prediction.",
            index=None
        )

    # --- Crop Selection ---
    st.multiselect(
        "Select Crop(s)",
        hp.options()["crop_list"],
        key="crop",
        placeholder="Select Crop(s)",
        help="Select one or more crops for prediction.",
        default=None
    )

    # --- Area & Unit Selection ---
    col5, col6 = st.columns(2)
    with col5:
        st.selectbox(
            "Select Unit of Area",
            [
                'ha', 'sq_m', 'sq_km', 'acre', 'sq_ft', 'sq_yd', 'gaj', 'kanal',
                'bigha', 'biswa', 'killa', 'lessa', 'dhur', 'pura', 'chatak', 'marla',
                'katha', 'ground', 'cent', 'murabba', 'guntha', 'karam'
            ],
            key="UnitOfArea",
            placeholder="Select Unit",
            index=None
        )

    with col6:
        disabled_area = not bool(st.session_state.get("UnitOfArea"))
        st.number_input(
            "Enter Area",
            placeholder="Enter area value",
            key="area",
            disabled=disabled_area
        )

    # --- Submit Button Logic ---
    required_fields = ["state", "district", "year", "season", "crop", "UnitOfArea", "area"]
    all_filled = all(st.session_state.get(field) not in [None, "", []] for field in required_fields)

    submit_button = st.button("🚀 Submit", disabled=not all_filled)
    if submit_button:
        st.session_state.submit = True

# --- Output View ---
def show_result():
    
    lat,lon = hp.extraction_lat_lon_values(st.session_state.state,st.session_state.district)
    env_data = hp.api_data(st.session_state.year,st.session_state.season.lower(),lat,lon)
    data_for_prediction = {
                "crop_year": st.session_state.year,
                "season": st.session_state.season,
                "crop": st.session_state.crop,
                "area": np.log1p(hp.unit_conversion(st.session_state.area,st.session_state.UnitOfArea)),
                "temperature_2m_mean": env_data["temperature_2m_mean"].mean(),
                "precipitation_sum": env_data["precipitation_sum"].mean(),
                "relative_humidity_2m_mean": env_data["relative_humidity_2m_mean"].mean(),
                "wind_speed_10m_mean": env_data["wind_speed_10m_mean"].mean(),
                "latitude": lat,
                "longitude": lon,
        
             }
    st.write(hp.predicction(data_for_prediction))
# --- App Flow ---
if not st.session_state.submit:
    show_input_form()
else:
    show_result()




        