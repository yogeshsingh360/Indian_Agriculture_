import requests
import pandas as pd
import time
from tqdm import tqdm
from catboost import Pool, CatBoostRegressor
import numpy as np
# Load model
model = CatBoostRegressor()
model.load_model("catboost_best_model.cbm")

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
    pred_dict = {}
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
        pred_dict[i] = round(pred, 3)
    return pred_dict 
     