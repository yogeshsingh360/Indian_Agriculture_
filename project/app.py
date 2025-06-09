from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import os
from flask import jsonify
from flask_cors import CORS
import helper

app = Flask(__name__)
CORS(app)
# Load cleaned dataset
data_df = pd.read_csv("final_data_for_web.csv")
# Preload dropdown options
state_list = sorted(data_df["state_name"].dropna().unique())
district_dict = {
    state: sorted(data_df[data_df["state_name"] == state]["district_name"].dropna().unique())
    for state in state_list
}
season_list = sorted(data_df["season"].dropna().unique())
crop_list = sorted(data_df["crop"].dropna().unique())

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    selected_state = None
    selected_district = None

    if request.method == "POST":
        try:
            selected_state = request.form.get("state")
            selected_district = request.form.get("district")
            crop_year = int(request.form.get("crop_year"))
            season = request.form.get("season")
            crop = request.form.getlist("crop")
            area = float(request.form.get("area"))

            # 🔍 Extract latitude and longitude
            location_row = data_df[
                (data_df["state_name"] == selected_state) &
                (data_df["district_name"] == selected_district)
            ]

            if location_row.empty:
                prediction = "Error: Location not found in data."
                return render_template("index.html",
                                       state_list=state_list,
                                       district_dict=district_dict,
                                       season_list=season_list,
                                       crop_list=crop_list,
                                       prediction=prediction)

            lat = location_row["latitude"].values[0]
            lon = location_row["longitude"].values[0]

            # ✅ Get weather API data
            api_data = helper.api_data(crop_year, season, lat, lon)

            input_data = {
                "crop_year": crop_year,
                "season": season,
                "crop": crop,
                "area": np.log1p(area),
                "temperature_2m_mean": api_data["temperature_2m_mean"].mean(),
                "precipitation_sum": api_data["precipitation_sum"].mean(),
                "relative_humidity_2m_mean": api_data["relative_humidity_2m_mean"].mean(),
                "wind_speed_10m_mean": api_data["wind_speed_10m_mean"].mean(),
                "latitude": lat,
                "longitude": lon,
            }

            # ✅ Prediction
            pred_data = helper.predicction(input_data)

            return render_template("result.html",
                                   prediction=pred_data,
                                   selected_state=selected_state,
                                   selected_district=selected_district,
                                   data=api_data.to_dict(orient="records"))

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template("index.html",
                           state_list=state_list,
                           district_dict=district_dict,
                           season_list=season_list,
                           crop_list=crop_list,
                           )


@app.route("/get_districts", methods=["POST"])
def get_districts():
    state = request.json.get("state")
    districts = district_dict.get(state, [])
    return jsonify(districts)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  
    app.run(host="0.0.0.0", port=port, debug=True)

