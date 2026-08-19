# 🌾 Indian Agriculture & Weather Analytics (2001–2020)

An end-to-end Machine Learning and Business Intelligence platform analyzing 20 years of Indian agricultural yields integrated with high-resolution historical climate data across 33 States/UTs and 685+ districts.

📖 **For the complete, in-depth technical report and quantitative evaluation, see [INDIAN_AGRICULTURE_PROJECT_REPORT.md](file:///c:/Users/91742/Desktop/trails/Indian_Agriculture_/INDIAN_AGRICULTURE_PROJECT_REPORT.md).**

---

## 🚀 Key Achievements & Quantitative Milestones
- **Dataset Scale**: 249,492 clean records synthesized across 2001–2020.
- **Top Machine Learning Model**: **Optuna-Tuned CatBoost Regressor** achieving:
  - **$R^2$ Score**: **0.8649 (86.5%)** on unseen out-of-time future crop years ($2019–2020$).
  - **Test RMSE**: **0.1898**
  - **Test MAE**: **0.2236**
  - **Minimal Overfitting**: $\text{Train } R^2 = 0.9215$ vs $\text{Test } R^2 = 0.8649$ ($\Delta = 0.056$).
- **Integrated Weather Features**: Temperature, precipitation, humidity, and wind speed dynamically fetched via Open-Meteo Historical Archive API.
- **Regional Land Support**: Automatic bidirectional conversion across **22 Indian land measurement units** (*Hectare, Acre, Bigha, Biswa, Guntha, Ground, Kanal, Marla, Killa, Katha, etc.*).
- **Interactive Web App**: Dual-page Streamlit application with live prediction engine and a 6-tab interactive analytics dashboard.

---

## 🛠️ Used APIs & Data Sources
1. **Open-Meteo Historical Weather Archive API**: `https://archive-api.open-meteo.com/v1/archive`
2. **India Data Portal APY Dataset**: `https://ckandev.indiadataportal.com/dataset/area-production-yield-apy/resource`

---

## 📂 Partitioning Strategy (2001–2010 vs 2011–2020)
The dataset was partitioned into two chronological tranches (2001–2010 and 2011–2020) to address district boundary redistricting across India over two decades, ensuring accurate spatial geocoding and meteorological joins.

---

## 🏃 Quick Start
```bash
# Clone & install dependencies
pip install -r project/requirements.txt

# Launch the Streamlit web application
cd project/streamlit_app
streamlit run app.py
```
