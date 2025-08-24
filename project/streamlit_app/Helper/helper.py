import requests
import pandas as pd
import time
import os
from catboost import Pool, CatBoostRegressor
import numpy as np
BASE_DIR = os.path.dirname(__file__)
CSV_PATH = os.path.join(BASE_DIR, "final_data_for_web.csv")
data_df = data_df = pd.read_csv(CSV_PATH)
# Load model
MODEL_BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(MODEL_BASE_DIR, "catboost_best_model.cbm")

model = CatBoostRegressor()
model.load_model(MODEL_PATH)
def extraction_lat_lon_values(state,district):
    location_row = data_df[
                (data_df["state_name"] == state.lower()) &
                (data_df["district_name"] == district.lower())
            ]
    
    lat = location_row["latitude"].values[0]
    lon = location_row["longitude"].values[0]
    return lat,lon
def options():
    
    # Preload dropdown options
    state_list = sorted(data_df["state_name"].dropna().unique())
    district_dict = {
        state: sorted(data_df[data_df["state_name"] == state]["district_name"].dropna().unique())
        for state in state_list
    }
    season_list = sorted(data_df["season"].dropna().unique())
    crop_list = sorted(data_df["crop"].dropna().unique())
    
    season_list = [ season.title() for season in season_list]
    state_list = [ state.title() for state in state_list]
    district_dict = {state.title(): [district.title() for district in districts] for state, districts in district_dict.items()}
    
    return {
        "state_list": state_list,
        "district_dict": district_dict,
        "season_list": season_list,
        "crop_list": crop_list
    }


def api_data(year,season,lat,lon):
    base_url = "https://archive-api.open-meteo.com/v1/archive"
    
    if season == "kharif":
        start_date = f"{year}-06-01"
        end_date = f"{year}-08-31"
    elif season == "rabi":
        start_date = f"{year}-02-01"
        end_date = f"{year}-03-31"
    elif season == "winter":
        start_date = f"{year}-11-01"
        end_date = f"{year+1}-01-31"
    elif season == "summer":
        start_date = f"{year}-04-01"
        end_date = f"{year}-05-31"
    elif season == "autumn":
        start_date = f"{year}-09-01"
        end_date = f"{year}-10-31"
    elif season == "whole year":
        start_date = f"{year}-02-01"
        end_date = f"{year+1}-01-31"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "precipitation_sum,relative_humidity_2m_mean,temperature_2m_mean,wind_speed_10m_mean"
    }

    try:
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        results = []
        if 'daily' in data:
            for i in range(len(data['daily']['time'])):
                results.append({
                    "latitude": lat,
                    "longitude": lon,
                    "date": data['daily']['time'][i],
                    "precipitation_sum": data['daily']['precipitation_sum'][i],
                    "relative_humidity_2m_mean": data['daily']['relative_humidity_2m_mean'][i],
                    "temperature_2m_mean": data['daily']['temperature_2m_mean'][i],
                    "wind_speed_10m_mean": data['daily']['wind_speed_10m_mean'][i]
                    
                })
        return pd.DataFrame(results)

    except Exception as e:
        return []
def predicction(data) :
    pred_list = []
    for i in data["crop"]:
        input_data = pd.DataFrame([{
            "crop_year": data["crop_year"],
            "season": data["season"].lower().replace(" ", ""),
            "crop": i.lower().replace(" ", ""),
            "area": data["area"],
            "temperature_2m_mean": data["temperature_2m_mean"],
            "precipitation_sum": data["precipitation_sum"],
            "relative_humidity_2m_mean": data["relative_humidity_2m_mean"],
            "wind_speed_10m_mean": data["wind_speed_10m_mean"],
            "latitude": data["latitude"],
            "longitude": data["longitude"],
        }])
        input_pool = Pool(input_data, cat_features=["season", "crop"])
        pred_log = model.predict(input_pool)
        pred = np.expm1(pred_log[0])
        pred_list.append({"Crop" : i,"Yield" : round(pred,3)})
    return pred_list 
def unit_conversion(land_area,area_unit):
    conversion_factors = {
        'ha': 1,
        'sq_m': 1 / 10000,          
        'sq_km': 100,                
        'acre': 0.4046856422,        
        'sq_ft': 0.0000092903,       
        'sq_yd': 0.0000836127,
        'gaj'  : 0.0000836,
        'kanal': 0.0505857,
        'bigha' : 0.1011714,
        'biswa':0.0050586,
        'killa' : 0.4046856,
        'lessa' :0.0002529,
        'dhur' : 0.0002529,
        'pura' :0.4046856,
        'chatak':0.0004181,
        'marla':0.0025293,
        'katha': 0.0050586,
        'ground':0.0223000,
        'cent':0.0040465,
        'murabba':10.1171411,
        'guntha':0.0101171,
        'karam':0.0002810} 
    return land_area*conversion_factors[area_unit]
def conversion_factor_Ha_to_X(unit):
    conversion_factors = {
        'ha': 1,
        'sq_m': 1 / 10000,          
        'sq_km': 100,                
        'acre': 0.4046856422,        
        'sq_ft': 0.0000092903,       
        'sq_yd': 0.0000836127,
        'gaj'  : 0.0000836,
        'kanal': 0.0505857,
        'bigha' : 0.1011714,
        'biswa':0.0050586,
        'killa' : 0.4046856,
        'lessa' :0.0002529,
        'dhur' : 0.0002529,
        'pura' :0.4046856,
        'chatak':0.0004181,
        'marla':0.0025293,
        'katha': 0.0050586,
        'ground':0.0223000,
        'cent':0.0040465,
        'murabba':10.1171411,
        'guntha':0.0101171,
        'karam':0.0002810} 
    return 1/conversion_factors[unit]
    
    
     