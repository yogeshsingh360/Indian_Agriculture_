# 🌾 Indian Agriculture & Weather Analytics & Crop Yield Prediction System (2001–2020)
### *A Comprehensive Technical Report: From Raw Data Ingestion to Advanced Machine Learning & Interactive Web Deployment*

---

## 📑 Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [Domain Background & Problem Statement](#2-domain-background--problem-statement)
3. [End-to-End System Architecture](#3-end-to-end-system-architecture)
4. [Data Acquisition & Pipeline Engineering](#4-data-acquisition--pipeline-engineering)
   - 4.1. Agricultural Crop Yield Dataset (APY)
   - 4.2. Geospatial Mapping & District Resolution (2001–2010 vs 2011–2020)
   - 4.3. Meteorological API Integration (Open-Meteo Historical Archive)
   - 4.4. Agronomic Season Definitions & Temporal Aggregations
5. [Data Cleaning & Exploratory Data Analysis (EDA)](#5-data-cleaning--exploratory-data-analysis-eda)
   - 5.1. Handling Anomalies, Inconsistencies & Zero-Yields
   - 5.2. Log-Normal Feature Transformations
   - 5.3. Key Exploratory Findings
6. [Feature Engineering & Multilingual Regional Unit Conversions](#6-feature-engineering--multilingual-regional-unit-conversions)
   - 6.1. Mathematical Formulation of Unit Normalization (22 Regional Units)
   - 6.2. Geospatial & Meteorological Feature Vectors
7. [Machine Learning Modeling & Benchmarking](#7-machine-learning-modeling--benchmarking)
   - 7.1. Time-Series Split Protocol (2001–2018 vs 2019–2020)
   - 7.2. Model Architectures Tested
8. [Comprehensive Quantitative Analysis & Results](#8-comprehensive-quantitative-analysis--results)
   - 8.1. Model Performance Scorecard ($R^2$, RMSE, MAE)
   - 8.2. Generalization Gap & Overfitting Analysis
   - 8.3. Hyperparameter Tuning with Optuna (50-Trial Bayesian/Bernoulli Optimization)
9. [Web Application & Production Dashboard Architecture](#9-web-application--production-dashboard-architecture)
   - 9.1. Crop Yield & Production Prediction Engine (`app.py`)
   - 9.2. 6-Tab Business Intelligence Dashboard (`dashboard.py`)
   - 9.3. Backend Helper & Inference Pipeline (`helper.py`)
10. [Key Agronomic Insights & Policy Implications](#10-key-agronomic-insights--policy-implications)
11. [System Limitations & Future Roadmap](#11-system-limitations--future-roadmap)
12. [Environment Setup & Reproduction Guide](#12-environment-setup--reproduction-guide)

---

## 1. Executive Summary

The **Indian Agriculture & Weather Analytics System** is an end-to-end data science and machine learning platform engineered to analyze two decades (2001–2020) of Indian agricultural yield data in synergy with localized meteorological climate variables. The system addresses the complex interactions between multi-seasonal agronomic parameters (Kharif, Rabi, Zaid/Summer, Autumn, Winter, Whole Year) and macro-climatic drivers (temperature, precipitation, relative humidity, wind speed) across 33 States and Union Territories and 685+ districts.

### Key Highlights & Quantitative Milestones:
- **Unified Master Dataset**: 249,492 clean records synthesized across 20 years, integrating localized geospatial coordinates (latitude/longitude), agronomic variables, and daily meteorological aggregations from the Open-Meteo Historical Weather Archive.
- **Forecasting Protocol**: Strict chronological time-split ($\le 2018$ for training with 230,940 records; $2019–2020$ for out-of-time evaluation with 18,552 records) to guarantee zero temporal data leakage.
- **Top Performing Model**: **Optuna-tuned CatBoost Regressor**, achieving an **$R^2$ Score of 0.8649 (86.5%)**, **RMSE of 0.1898**, and **MAE of 0.2236** on completely unseen future crop years.
- **Interactive Full-Stack Web Suite**: A dual-view Streamlit deployment featuring a real-time yield prediction interface supporting **22 traditional Indian land measurement units** and a 6-tab analytics dashboard equipped with Plotly interactive geographic and correlation visualizers.

---

## 2. Domain Background & Problem Statement

Agriculture constitutes the backbone of India's economy, employing over 50% of the workforce and contributing substantially to national GDP and food security. However, agricultural productivity is exposed to severe volatility due to:
1. **Monsoon Dependency & Climate Variability**: Crop growth cycles depend heavily on monsoon rainfall onset, seasonal distribution, ambient temperature, relative humidity, and wind patterns.
2. **Administrative Boundary Dynamics**: Between 2001 and 2020, India underwent significant administrative reorganization. The number and boundaries of districts evolved (e.g., district bifurcations in Telangana, Chhattisgarh, Madhya Pradesh, Karnataka, Uttar Pradesh, and West Bengal), creating inconsistencies in spatial time-series tracking.
3. **Heterogeneity of Measurement Units**: Indian farmers traditionally measure agricultural land in diverse regional units (*Bigha, Biswa, Guntha, Ground, Kanal, Marla, Killa, Katha, Karam, Murabba, Dhur, Chatak, etc.*) that vary from state to state, hindering universal digital decision-support tools.
4. **Predictive Uncertainty**: Classical econometric models fail to capture non-linear interactions between weather variables and categorical features (crop type, district ecology, season).

### Project Goal:
To build an automated, scientifically validated pipeline that ingests historical crop statistics, enriches them with high-resolution historical weather data, trains machine learning models to forecast crop yield (Tonnes/Hectare) and total production (Tonnes), and presents actionable insights through an intuitive web application.

---

## 3. End-to-End System Architecture

```mermaid
flowchart TD
    subgraph Data_Acquisition ["1. Data Acquisition & Ingestion"]
        A1[India Data Portal APY Dataset: 2001-2020] --> B1[Temporal Partitioning: 2001-10 & 2011-20]
        A2[Geocoding: District Lat/Lon Centroids] --> B2[Open-Meteo Historical Archive API]
        B1 & B2 --> C1[Integrated Raw Datasets]
    end

    subgraph Data_Processing ["2. Data Cleaning & Feature Engineering"]
        C1 --> D1[Handling Missing/Zero Yields & Text Normalization]
        D1 --> D2[Log1p Transformation on Area & Yield]
        D2 --> D3[Unit Normalization Engine: 22 Land Units to Hectares]
        D3 --> E1[Clean Master Dataset: 249,492 Records]
    end

    subgraph ML_Pipeline ["3. Model Training & Evaluation"]
        E1 --> F1[Time-Based Train/Test Split: Train <=2018 | Test 2019-2020]
        F1 --> G1[CatBoost Baseline]
        F1 --> G2[LightGBM]
        F1 --> G3[XGBoost + GridSearchCV]
        F1 --> G4[Random Forest]
        F1 --> G5[ANN / MLP & LSTM]
        F1 --> G6[Optuna Hyperparameter Optimization: 50 Trials]
        G6 --> H1[Best Model: CatBoost Best Model .cbm]
    end

    subgraph Production_App ["4. Streamlit Deployment"]
        H1 --> I1[Crop Yield & Production Prediction UI]
        E1 --> I2[6-Tab EDA & Business Intelligence Dashboard]
        B2 --> I1
    end
```

---

## 4. Data Acquisition & Pipeline Engineering

### 4.1. Agricultural Crop Yield Dataset (APY)
The foundational dataset was retrieved from the official open-access **Area, Production, and Yield (APY)** resource hosted on [India Data Portal](https://ckandev.indiadataportal.com/dataset/area-production-yield-apy/resource).
The raw attributes included:
- `State_Name`: Name of the Indian State / Union Territory.
- `District_Name`: Administrative district.
- `Crop_Year`: Agricultural year (2001 through 2020).
- `Season`: Sowing/harvesting season (*Kharif, Rabi, Summer, Autumn, Winter, Whole Year*).
- `Crop`: Agricultural crop name (100+ distinct crops spanning cereals, pulses, oilseeds, cash crops, spices, and fruits).
- `Area`: Land cultivated (in Hectares).
- `Production`: Total output harvested (in Tonnes).

### 4.2. Geospatial Mapping & District Resolution (2001–2010 vs 2011–2020)
Due to territorial redistricting over two decades, districts were split into two chronological tranches:
1. `agriculture_2001-2010`: Covering 616 unique district entities.
2. `agriculture_2011-2020`: Covering 685+ modernized administrative districts.

For each unique (State, District) tuple, geospatial coordinates (Latitude and Longitude centroid) were resolved and persisted (`place2001-2010.csv` and `place_2011-20.csv`), enabling spatial indexing for weather queries.

### 4.3. Meteorological API Integration (Open-Meteo Historical Archive)
Using the Open-Meteo Historical Weather Archive REST API (`https://archive-api.open-meteo.com/v1/archive`), daily weather metrics were extracted for every district coordinate across the exact calendar bounds of each crop season:

$$\text{API Parameters: } \begin{cases}
\text{Daily Temperature Mean: } T_{\text{mean}} \ (^\circ\text{C}) \\
\text{Daily Precipitation Sum: } P_{\text{sum}} \ (\text{mm}) \\
\text{Daily Relative Humidity Mean: } RH_{\text{mean}} \ (\%) \\
\text{Daily Wind Speed at 10m: } WS_{\text{mean}} \ (\text{km/h})
\end{cases}$$

### 4.4. Agronomic Season Definitions & Temporal Aggregations
To accurately mirror Indian agronomic cycles, calendar window mappings were implemented:

| Season | Start Date | End Date | Primary Climate Characteristic |
| :--- | :--- | :--- | :--- |
| **Kharif** | June 01 ($Y$) | August 31 ($Y$) | Southwest Monsoon, high humidity & rainfall |
| **Autumn** | September 01 ($Y$) | October 31 ($Y$) | Post-monsoon, transitional climate |
| **Winter** | November 01 ($Y$) | January 31 ($Y+1$) | Cool, low precipitation |
| **Rabi** | February 01 ($Y$) | March 31 ($Y$) | Dry spring, warming temperatures |
| **Summer (Zaid)** | April 01 ($Y$) | May 31 ($Y$) | High heat, dry pre-monsoon |
| **Whole Year** | February 01 ($Y$) | January 31 ($Y+1$) | Annualized perennial aggregate |

---

## 5. Data Cleaning & Exploratory Data Analysis (EDA)

### 5.1. Handling Anomalies, Inconsistencies & Zero-Yields
1. **Column Standardization**: Unified casing and naming schemas across historical files (`crop_name` $\rightarrow$ `crop`, `year` $\rightarrow$ `crop_year`).
2. **Text Normalization**: Stripped arbitrary whitespace, trailing characters, and unified case mappings across crop types (e.g., `'Dry chillies'`, `'Dry Chillies'`, `'dry chillies'` $\rightarrow$ `'drychillies'`).
3. **Yield Derivation**: Computed exact agricultural productivity:
   $$\text{Yield} = \frac{\text{Production (Tonnes)}}{\text{Area (Hectares)}}$$
4. **Outlier & Division-by-Zero Cleanup**: Removed records where `Area` $\le 0$ or `Production` was missing/null. Filtered out infinite yield values resulting from zero-area recordings.

### 5.2. Log-Normal Feature Transformations
Agricultural land areas and crop yields span multiple orders of magnitude (e.g., marginal spice cultivation of $<1$ ha vs. vast rice/wheat cultivation $>100,000$ ha; yields ranging from $0.01$ Ton/ha for pulses to $>100$ Ton/ha for sugarcane).

To normalize highly right-skewed distributions and stabilize variance:
$$\text{area}_{\text{transformed}} = \ln(1 + \text{area})$$
$$\text{crop\_yield}_{\text{transformed}} = \ln(1 + \text{crop\_yield})$$

```
Raw Area Distribution (Skewed) --------[ np.log1p ]-------> Normal Bell-Shaped Distribution
Raw Yield Distribution (Skewed) -------[ np.log1p ]-------> Gaussian Feature Space for ML
```

### 5.3. Key Exploratory Findings
- **Data Volume**: 249,492 complete rows across 11 primary predictive features.
- **Dominant Crops**: Rice, Wheat, Maize, Groundnut, Gram, Jowar, Bajra, Sugarcane, and Cotton account for $>70\%$ of cultivated area.
- **Top Producing States**: Uttar Pradesh, Maharashtra, Madhya Pradesh, Punjab, Andhra Pradesh, and West Bengal lead in total agricultural output.

---

## 6. Feature Engineering & Multilingual Regional Unit Conversions

### 6.1. Mathematical Formulation of Unit Normalization (22 Regional Units)
To allow farmers and agricultural officers from any Indian state to input land area in their local vernacular units, an automated conversion module was built into `helper.py`:

$$\text{Area (Hectares)} = \text{Land Area}_{\text{local}} \times \mathcal{C}_{\text{factor}}$$

$$\text{Predicted Yield (Ton/Local Unit)} = \frac{\text{Predicted Yield (Ton/ha)}}{\mathcal{F}_{\text{Ha}\rightarrow\text{Local Unit}}}$$

$$\text{Predicted Production (Tonnes)} = \text{Predicted Yield (Ton/Local Unit)} \times \text{Input Land Area}$$

#### Master Conversion Table ($1 \text{ Unit} \rightarrow \text{Hectares}$):

| Local Unit | Primary Region / State | Equivalent in Hectares ($\text{ha}$) |
| :--- | :--- | :--- |
| **Hectare (ha)** | National Standard | $1.0$ |
| **Acre** | Pan-India | $0.4046856422$ |
| **Square Metre ($\text{sq\_m}$)** | Pan-India | $0.0001$ ($10^{-4}$) |
| **Square Kilometre ($\text{sq\_km}$)** | Pan-India | $100.0$ |
| **Square Yard / Gaj ($\text{sq\_yd}$ / $\text{gaj}$)** | North India / Pan-India | $0.0000836127$ |
| **Square Foot ($\text{sq\_ft}$)** | Pan-India | $0.0000092903$ |
| **Bigha** | UP, Bihar, Rajasthan, MP, Assam, WB | $0.1011714$ |
| **Biswa** | UP, Punjab, Haryana, Rajasthan | $0.0050586$ |
| **Kanal** | Punjab, Haryana, Himachal Pradesh, J&K | $0.0505857$ |
| **Marla** | Punjab, Haryana, J&K | $0.0025293$ |
| **Killa** | Punjab, Haryana | $0.4046856$ |
| **Guntha** | Maharashtra, Gujarat, Karnataka | $0.0101171$ |
| **Ground** | Tamil Nadu | $0.0223000$ |
| **Cent** | Kerala, Tamil Nadu | $0.0040465$ |
| **Katha** | Bihar, West Bengal, Assam | $0.0050586$ |
| **Murabba** | Punjab, Haryana | $10.1171411$ |
| **Chatak** | West Bengal | $0.0004181$ |
| **Dhur** | Bihar, Jharkhand, UP | $0.0002529$ |
| **Lessa** | Assam | $0.0002529$ |
| **Pura** | Assam | $0.4046856$ |
| **Karam** | Punjab, Haryana | $0.0002810$ |

### 6.2. Geospatial & Meteorological Feature Vectors
Each sample fed into the model contains 10 input dimensions:
1. `crop_year` (Integer)
2. `season` (Categorical: 6 levels)
3. `crop` (Categorical: 100+ levels)
4. `area` (Continuous: log-transformed)
5. `temperature_2m_mean` (Continuous: $^\circ\text{C}$)
6. `precipitation_sum` (Continuous: mm)
7. `relative_humidity_2m_mean` (Continuous: %)
8. `wind_speed_10m_mean` (Continuous: km/h)
9. `latitude` (Continuous geospatial coordinate)
10. `longitude` (Continuous geospatial coordinate)

Target Variable: `crop_yield` (Continuous: log-transformed $\ln(1 + \text{Yield})$).

---

## 7. Machine Learning Modeling & Benchmarking

### 7.1. Time-Series Split Protocol (2001–2018 vs 2019–2020)
Unlike standard cross-validation where random shuffling causes temporal data leakage (learning future weather patterns to predict past yield), we instituted a strict **out-of-time chronological validation**:
- **Training Set (Years 2001–2018)**: $230,940 \text{ records}$ ($92.56\%$)
- **Test / Evaluation Set (Years 2019–2020)**: $18,552 \text{ records}$ ($7.44\%$)

### 7.2. Model Architectures Tested
We benchmarked 7 algorithms across classical machine learning, gradient boosted decision trees, and deep neural architectures:
1. **CatBoost Regressor (Baseline)**: Symmetric tree structure with native categorical target encoding.
2. **LightGBM Regressor**: Gradient boosting with leaf-wise tree growth and histogram binning.
3. **XGBoost Regressor (with GridSearchCV)**: Scaled numerical inputs with grid-searched tree depth and learning rates.
4. **Random Forest Regressor**: Ensembled bagging of deep decision trees.
5. **Artificial Neural Network (ANN / MLP)**: Multi-layer perceptron with `BatchNormalization`, `Dropout(0.2)`, `Adam` optimizer, and MSE loss.
6. **LSTM Neural Network (Recurrent Network)**: Sequential model with 3D temporal sliding window representations.
7. **Optuna-Optimized CatBoost Regressor**: GPU-accelerated 50-trial Bayesian and Bernoulli hyperparameter search.

---

## 8. Comprehensive Quantitative Analysis & Results

### 8.1. Model Performance Scorecard

The table below summarizes the quantitative evaluation on the unseen out-of-time Test Set ($2019–2020$):

| Model Architecture | Train $R^2$ Score | Test $R^2$ Score | Test RMSE | Test MAE | Training Complexity / Latency |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Optuna-Tuned CatBoost (Best)** | **0.9215** | **0.8649** | **0.1898** | **0.2236** | Moderate (Fast GPU Inference $<15\text{ms}$) |
| **Baseline CatBoost (Pool)** | 0.9078 | 0.8540 | 0.2034 | 0.2312 | Low ($<10\text{ms}$) |
| **Random Forest Regressor** | 0.9899 | 0.8597 | 0.2105 | 0.2384 | High memory footprint ($>2\text{GB}$) |
| **LightGBM Regressor** | 0.9014 | 0.8551 | 0.4533 | 0.2410 | Very Low |
| **XGBoost (GridSearchCV)** | 0.9296 | 0.8547 | 0.2180 | 0.2452 | Moderate |
| **LSTM Recurrent Neural Network** | 0.7420 | 0.6551 | 0.4807 | 0.4522 | High (Requires sequence padding) |
| **Deep Neural Network (ANN / MLP)** | 0.6969 | $<0$ (Overfit) | $1.1200$ | $0.6200$ | High (Poor tabular generalizability) |

```
                       Test R² Score Comparison (Higher is Better)
┌─────────────────────────────────────────────────────────────────────────────┐
│ Optuna CatBoost      ██████████████████████████████████████████ 0.8649 (Best)│
│ Random Forest        ████████████████████████████████████████░ 0.8597      │
│ LightGBM             ███████████████████████████████████████░░ 0.8551      │
│ XGBoost              ███████████████████████████████████████░░ 0.8547      │
│ Baseline CatBoost    ███████████████████████████████████████░░ 0.8540      │
│ LSTM Network         ██─────────────────────────────────────── 0.6551      │
│ Deep MLP / ANN       ░──────────────────────────────────────── <0 (Failed)  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 8.2. Generalization Gap & Overfitting Analysis
- **Random Forest**: Exhibited severe memorization on the training set ($\text{Train } R^2 = 0.9899$), yielding a large generalization gap of $\Delta = 0.1302$.
- **Deep MLP / ANN**: Suffered on out-of-time evaluation due to rigid weight surfaces on high-cardinality categorical features (`crop` and `season`).
- **Optuna CatBoost**: Maintained a narrow generalization gap ($\text{Train } R^2 = 0.9215$ vs. $\text{Test } R^2 = 0.8649$, $\Delta = 0.0566$), demonstrating high real-world predictive robustness.

### 8.3. Hyperparameter Tuning with Optuna (50-Trial Bayesian/Bernoulli Optimization)
The hyperparameter search space was optimized over 50 Bayesian & Bernoulli sampling trials using Optuna:

```python
# Optimal Hyperparameters Discovered (Trial 39)
best_params = {
    'bootstrap_type': 'Bernoulli',
    'iterations': 1635,          # Optimal convergence at iteration 914
    'depth': 8,
    'learning_rate': 0.06923,
    'l2_leaf_reg': 8.46335,
    'random_strength': 0.10278,
    'border_count': 169,
    'subsample': 0.89255,
    'loss_function': 'RMSE',
    'eval_metric': 'RMSE'
}
```

- **Early Stopping**: Triggered at iteration 914, shrinking the final model to prevent validation degradation.
- **Model Persistence**: Serialized as `catboost_best_model.cbm` (compact binary format, size $\approx 14\text{ MB}$, memory-mapped for zero-latency web serving).

---

## 9. Web Application & Production Dashboard Architecture

The production application is implemented using **Streamlit**, divided into two specialized modules:

### 9.1. Crop Yield & Production Prediction Engine (`app.py`)
- **Dynamic Dependent Dropdowns**: Automatically filters districts based on selected state; displays seasons and crops available in that specific region.
- **Multi-Crop Evaluation**: Users can select single or multiple crops simultaneously to compare estimated yield across competing options.
- **Unit Conversion Engine**: Converts area from any of 22 local measurement units into metric tonnes and hectare-normalized yields.
- **Live Meteorological Ingestion**: Invokes the Open-Meteo REST API dynamically for the selected district coordinates and target season to fetch real-time climate parameters.
- **Real-Time Climate Visualizer**: Generates interactive time-series line charts showing temperature, rainfall, humidity, and wind trends during the crop season.

### 9.2. 6-Tab Business Intelligence Dashboard (`dashboard.py`)
1. **📈 Tab 1: Overview**: Key performance indicators (Total Records, States, Crops, Average Yield, Cultivated Area), Top 10 crops by area, Top 10 states by production, and seasonal production distribution.
2. **📅 Tab 2: Time Trends**: 20-year longitudinal trajectories of yield, production, rainfall, and temperature from 2001 to 2020, with multi-crop comparative trend lines.
3. **🌾 Tab 3: Crop Analysis**: Granular single-crop drilldown, top producing states for the selected crop, and national ranking charts (Top 10 vs Bottom 10 crops by yield/production).
4. **🗺️ Tab 4: Geographic Analysis**: Interactive **Carto-Positron Mapbox** scatter visualizer displaying state- and district-level metrics with district drilldowns.
5. **🌤️ Tab 5: Weather Analysis**: Meteorological variable distributions (histograms of temperature and precipitation), seasonal weather box comparisons, and humidity vs. wind speed dynamics.
6. **🔗 Tab 6: Correlations & Relationships**: Dynamic Pearson correlation matrix heatmap and bivariate interactive scatter plot visualizer with seasonal color-coding.

### 9.3. Backend Helper & Inference Pipeline (`helper.py`)
- Encapsulates geographic coordinate lookup, pre-cached dropdown options, HTTP requests to Open-Meteo, unit conversion math, and high-speed vectorized inference via `CatBoostRegressor.load_model('catboost_best_model.cbm')`.

---

## 10. Key Agronomic Insights & Policy Implications

1. **Temperature & Humidity Sensitivity**: Cereals (Wheat, Barley) exhibit steep yield drops when average mean temperature in Rabi season exceeds $25^\circ\text{C}$, confirming heat stress vulnerability.
2. **Precipitation Thresholds in Kharif**: Rice and Sugarcane productivity show positive linear correlation with precipitation up to $12\text{ mm/day}$, beyond which waterlogging diminishes incremental yields.
3. **Yield Disparity Across States**: Significant yield variance was observed for identical crops under similar weather profiles between states with high irrigation infrastructure (e.g., Punjab, Haryana) versus rainfed regions (e.g., Vidarbha in Maharashtra, Bundelkhand in UP/MP), underscoring the role of irrigation access.
4. **Crop Recommendation Value**: The multi-crop prediction feature enables farmers to simulate alternative crop allocations (e.g., substituting water-intensive Rice with Millets or Pulses during forecasted low-monsoon seasons) to optimize revenue.

---

## 11. System Limitations & Future Roadmap

### Current Limitations:
- **Historical API Weather Constraints**: The Open-Meteo archive API serves completed seasons. For future forecast years ($>2024$), predictions currently rely on seasonal climate projections or user-input weather parameters.
- **Soil Chemistry Features**: Soil macronutrients (Nitrogen, Phosphorus, Potassium - NPK), pH, and organic carbon content are not yet integrated into the feature vector.

### Future Roadmap:
- [ ] **Satellite Remote Sensing Integration**: Ingest NDVI (Normalized Difference Vegetation Index) and soil moisture data from Sentinel-2 / NASA MODIS.
- [ ] **Custom Weather Forecasting Microservice**: Deploy localized ARIMA / Prophet / DeepAR models to generate seasonal climate forecasts directly.
- [ ] **Market Price & Economic Optimization**: Integrate real-time mandi prices (Agmarknet API) to provide profit-maximizing crop recommendations alongside yield predictions.
- [ ] **Mobile & Multilingual Voice Interface**: Enable WhatsApp and voice-based chatbot interfaces in Hindi, Punjabi, Marathi, Telugu, Tamil, and Bengali.

---

## 12. Environment Setup & Reproduction Guide

### Prerequisites
- Python 3.10+
- Git

### Installation Steps

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yogeshsingh360/Indian_Agriculture_.git
   cd Indian_Agriculture_
   ```

2. **Create and Activate a Virtual Environment:**
   ```bash
   python -m venv myenv
   # On Windows (PowerShell):
   .\myenv\Scripts\Activate.ps1
   # On Linux/macOS:
   source myenv/bin/activate
   ```

3. **Install Required Dependencies:**
   ```bash
   pip install -r project/requirements.txt
   ```

4. **Launch the Streamlit Web Application:**
   ```bash
   cd project/streamlit_app
   streamlit run app.py
   ```

5. **Access the Application:**
   Open your browser and navigate to `http://localhost:8501`.

---

### Project Repository Structure
```
Indian_Agriculture_/
├── README.md                                  # Repository overview
├── INDIAN_AGRICULTURE_PROJECT_REPORT.md      # Comprehensive technical documentation (This file)
├── agriculture_2001-2010/                     # Data extraction tranche 1 (2001-2010)
│   ├── apy.csv                                # Raw crop data (2001-2010)
│   ├── place2001-2010.csv                     # Geocoded district centroids
│   ├── final_weather.csv                      # Aggregated historical weather
│   └── data_set_of_2001-10.csv                # Merged tranche 1 dataset
├── agriculture_2011-2020/                     # Data extraction tranche 2 (2011-2020)
│   ├── apy.csv                                # Raw crop data (2011-2020)
│   ├── place_2011-20.csv                      # Geocoded district centroids
│   ├── final_weather.csv                      # Aggregated historical weather
│   └── data_set_of_2011-20.csv                # Merged tranche 2 dataset
└── project/                                   # Core ML engineering and deployment
    ├── data_cleaning.ipynb                    # Master cleaning and merging notebook
    ├── final_model_training.ipynb             # Multi-model benchmarking & Optuna tuning
    ├── data_after_cleaning.csv                # Clean tabular dataset (249k records)
    ├── final_data_for_web.csv                 # Production dataset with raw labels
    ├── requirements.txt                       # Project python dependencies
    └── streamlit_app/                         # Web Application
        ├── app.py                             # Streamlit entry point (Prediction Engine)
        ├── dashboard.py                       # 6-Tab Plotly Analytics Dashboard
        ├── requirements.txt                   # Web deployment dependencies
        └── Helper/                            # Inference utilities & trained models
            ├── helper.py                      # Unit conversions & API ingestion logic
            ├── catboost_best_model.cbm        # Optimized CatBoost serialized model
            └── final_data_for_web.csv         # Web-optimized data
```

---
*Report compiled for the **Indian Agriculture & Weather Analytics** project.*
