import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

@st.cache_data
def load_data():
    base_dir = os.path.dirname(__file__)
    csv_path = os.path.join(base_dir, "final_data_for_web.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(base_dir, "Helper", "final_data_for_web.csv")

    df = pd.read_csv(csv_path)

    # Clean infinite values in yield
    df["crop_yield"] = df["crop_yield"].replace([np.inf, -np.inf], np.nan)
    df["crop_yield"] = df["crop_yield"].fillna(0)

    # Calculate production (tonnes) = area (ha) * yield (tonnes/ha)
    df["production"] = df["area"] * df["crop_yield"]

    # Format text fields for clean display
    df["state_name"] = df["state_name"].astype(str).str.strip().str.title()
    df["district_name"] = df["district_name"].astype(str).str.strip().str.title()
    df["season"] = df["season"].astype(str).str.strip().str.title()
    df["crop"] = df["crop"].astype(str).str.strip()

    return df


def show_dashboard():
    st.title("📊 Indian Agriculture & Weather Analysis (2001–2020)")
    st.caption("Interactive exploratory data analysis of crop production, yields, and meteorological conditions across India.")

    df = load_data()

    # --- Sidebar Filters ---
    st.sidebar.header("🔍 Dashboard Filters")

    min_year = int(df["crop_year"].min())
    max_year = int(df["crop_year"].max())

    selected_years = st.sidebar.slider(
        "Year Range",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year)
    )

    all_states = sorted(df["state_name"].unique())
    selected_states = st.sidebar.multiselect(
        "Filter by State(s)",
        options=all_states,
        default=[]
    )

    all_seasons = sorted(df["season"].unique())
    selected_seasons = st.sidebar.multiselect(
        "Filter by Season(s)",
        options=all_seasons,
        default=[]
    )

    all_crops = sorted(df["crop"].unique())
    selected_crops = st.sidebar.multiselect(
        "Filter by Crop(s)",
        options=all_crops,
        default=[]
    )

    # Apply filters
    filtered_df = df[
        (df["crop_year"] >= selected_years[0]) &
        (df["crop_year"] <= selected_years[1])
    ]

    if selected_states:
        filtered_df = filtered_df[filtered_df["state_name"].isin(selected_states)]
    if selected_seasons:
        filtered_df = filtered_df[filtered_df["season"].isin(selected_seasons)]
    if selected_crops:
        filtered_df = filtered_df[filtered_df["crop"].isin(selected_crops)]

    if filtered_df.empty:
        st.warning("No records found matching the selected filter criteria. Please adjust your filters.")
        return

    # --- Dashboard Tabs ---
    tab_overview, tab_trends, tab_crops, tab_geo, tab_weather, tab_correlations = st.tabs([
        "📈 Overview",
        "📅 Time Trends",
        "🌾 Crop Analysis",
        "🗺️ Geographic Analysis",
        "🌤️ Weather Analysis",
        "🔗 Correlations"
    ])

    # ==========================================
    # TAB 1: OVERVIEW
    # ==========================================
    with tab_overview:
        st.subheader("Key Agricultural & Climate Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Records", f"{len(filtered_df):,}")
        col2.metric("Years Covered", f"{selected_years[0]} - {selected_years[1]}")
        col3.metric("States / UTs", f"{filtered_df['state_name'].nunique()}")
        col4.metric("Crops Included", f"{filtered_df['crop'].nunique()}")

        col5, col6, col7, col8 = st.columns(4)
        col5.metric("Avg Yield", f"{filtered_df['crop_yield'].mean():.2f} Ton/ha")
        col6.metric("Total Area", f"{filtered_df['area'].sum() / 1e6:.2f} M ha")
        col7.metric("Avg Temp", f"{filtered_df['temperature_2m_mean'].mean():.1f} °C")
        col8.metric("Avg Precip", f"{filtered_df['precipitation_sum'].mean():.1f} mm")

        st.markdown("---")

        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("##### Top 10 Crops by Cultivated Area")
            top_crops_area = (
                filtered_df.groupby("crop")["area"]
                .sum()
                .reset_index()
                .sort_values(by="area", ascending=False)
                .head(10)
            )
            fig_area = px.bar(
                top_crops_area,
                x="area",
                y="crop",
                orientation="h",
                labels={"area": "Total Area (ha)", "crop": "Crop"},
                color="area",
                color_continuous_scale="Blues"
            )
            fig_area.update_layout(yaxis=dict(autorange="reversed"), showlegend=False, height=380, margin=dict(l=10, r=10, t=20, b=20))
            st.plotly_chart(fig_area, use_container_width=True)

        with col_right:
            st.markdown("##### Top 10 States by Total Production")
            top_state_prod = (
                filtered_df.groupby("state_name")["production"]
                .sum()
                .reset_index()
                .sort_values(by="production", ascending=False)
                .head(10)
            )
            fig_prod = px.bar(
                top_state_prod,
                x="production",
                y="state_name",
                orientation="h",
                labels={"production": "Total Production (Tonnes)", "state_name": "State"},
                color="production",
                color_continuous_scale="Greens"
            )
            fig_prod.update_layout(yaxis=dict(autorange="reversed"), showlegend=False, height=380, margin=dict(l=10, r=10, t=20, b=20))
            st.plotly_chart(fig_prod, use_container_width=True)

        st.markdown("##### Seasonal Breakdown")
        season_summary = (
            filtered_df.groupby("season")
            .agg(
                Total_Area=("area", "sum"),
                Total_Production=("production", "sum"),
                Avg_Yield=("crop_yield", "mean"),
                Records=("crop", "count")
            )
            .reset_index()
        )
        fig_season = px.pie(
            season_summary,
            names="season",
            values="Total_Production",
            title="Production Share by Season",
            hole=0.4
        )
        fig_season.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=20))
        st.plotly_chart(fig_season, use_container_width=True)

    # ==========================================
    # TAB 2: TIME TRENDS
    # ==========================================
    with tab_trends:
        st.subheader("Historical Trends (2001–2020)")

        yearly_data = (
            filtered_df.groupby("crop_year")
            .agg(
                Avg_Yield=("crop_yield", "mean"),
                Total_Production=("production", "sum"),
                Total_Area=("area", "sum"),
                Mean_Temp=("temperature_2m_mean", "mean"),
                Mean_Precip=("precipitation_sum", "mean"),
                Mean_Humidity=("relative_humidity_2m_mean", "mean")
            )
            .reset_index()
        )

        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.markdown("##### Average Crop Yield Over Time")
            fig_yield_trend = px.line(
                yearly_data,
                x="crop_year",
                y="Avg_Yield",
                markers=True,
                labels={"crop_year": "Year", "Avg_Yield": "Average Yield (Ton/ha)"},
                line_shape="linear"
            )
            fig_yield_trend.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=20))
            st.plotly_chart(fig_yield_trend, use_container_width=True)

        with col_t2:
            st.markdown("##### Total Production Over Time")
            fig_prod_trend = px.bar(
                yearly_data,
                x="crop_year",
                y="Total_Production",
                labels={"crop_year": "Year", "Total_Production": "Total Production (Tonnes)"},
                color="Total_Production",
                color_continuous_scale="Viridis"
            )
            fig_prod_trend.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=20), showlegend=False)
            st.plotly_chart(fig_prod_trend, use_container_width=True)

        col_t3, col_t4 = st.columns(2)
        with col_t3:
            st.markdown("##### Mean Temperature Over Time")
            fig_temp_trend = px.line(
                yearly_data,
                x="crop_year",
                y="Mean_Temp",
                markers=True,
                labels={"crop_year": "Year", "Mean_Temp": "Mean Temperature (°C)"},
                color_discrete_sequence=["#e6550d"]
            )
            fig_temp_trend.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=20))
            st.plotly_chart(fig_temp_trend, use_container_width=True)

        with col_t4:
            st.markdown("##### Mean Precipitation Over Time")
            fig_precip_trend = px.line(
                yearly_data,
                x="crop_year",
                y="Mean_Precip",
                markers=True,
                labels={"crop_year": "Year", "Mean_Precip": "Precipitation (mm)"},
                color_discrete_sequence=["#3182bd"]
            )
            fig_precip_trend.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=20))
            st.plotly_chart(fig_precip_trend, use_container_width=True)

        # Multi-crop trend explorer
        st.markdown("##### Compare Crop Yield Trends Over Time")
        top_popular_crops = filtered_df["crop"].value_counts().head(6).index.tolist()
        trend_crops = st.multiselect(
            "Select crops to compare:",
            options=all_crops,
            default=[c for c in top_popular_crops if c in filtered_df["crop"].unique()][:3]
        )
        if trend_crops:
            crop_trend_df = (
                filtered_df[filtered_df["crop"].isin(trend_crops)]
                .groupby(["crop_year", "crop"])["crop_yield"]
                .mean()
                .reset_index()
            )
            fig_multicrop = px.line(
                crop_trend_df,
                x="crop_year",
                y="crop_yield",
                color="crop",
                markers=True,
                labels={"crop_year": "Year", "crop_yield": "Average Yield (Ton/ha)", "crop": "Crop"}
            )
            fig_multicrop.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=20))
            st.plotly_chart(fig_multicrop, use_container_width=True)

    # ==========================================
    # TAB 3: CROP ANALYSIS
    # ==========================================
    with tab_crops:
        st.subheader("Deep Dive by Crop")

        available_crops = sorted(filtered_df["crop"].unique())
        selected_single_crop = st.selectbox(
            "Select a Crop to analyze in detail:",
            options=available_crops,
            index=available_crops.index("Rice") if "Rice" in available_crops else 0
        )

        single_crop_df = filtered_df[filtered_df["crop"] == selected_single_crop]

        col_c1, col_c2, col_c3, col_c4 = st.columns(4)
        col_c1.metric("Selected Crop", selected_single_crop)
        col_c2.metric("Avg Yield", f"{single_crop_df['crop_yield'].mean():.2f} Ton/ha")
        col_c3.metric("Total Cultivated Area", f"{single_crop_df['area'].sum() / 1e3:.1f}k ha")
        col_c4.metric("Total Production", f"{single_crop_df['production'].sum() / 1e3:.1f}k Ton")

        col_c_left, col_c_right = st.columns(2)
        with col_c_left:
            st.markdown(f"##### {selected_single_crop} Yield Trend (2001–2020)")
            single_crop_yearly = (
                single_crop_df.groupby("crop_year")["crop_yield"]
                .mean()
                .reset_index()
            )
            fig_single_trend = px.line(
                single_crop_yearly,
                x="crop_year",
                y="crop_yield",
                markers=True,
                labels={"crop_year": "Year", "crop_yield": "Average Yield (Ton/ha)"}
            )
            fig_single_trend.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=20))
            st.plotly_chart(fig_single_trend, use_container_width=True)

        with col_c_right:
            st.markdown(f"##### Top Producing States for {selected_single_crop}")
            top_states_crop = (
                single_crop_df.groupby("state_name")["production"]
                .sum()
                .reset_index()
                .sort_values(by="production", ascending=False)
                .head(8)
            )
            fig_crop_states = px.bar(
                top_states_crop,
                x="production",
                y="state_name",
                orientation="h",
                labels={"production": "Production (Tonnes)", "state_name": "State"},
                color="production",
                color_continuous_scale="Purples"
            )
            fig_crop_states.update_layout(yaxis=dict(autorange="reversed"), showlegend=False, height=340, margin=dict(l=10, r=10, t=20, b=20))
            st.plotly_chart(fig_crop_states, use_container_width=True)

        st.markdown("---")
        st.markdown("##### National Crop Performance Rankings")
        rank_metric = st.radio("Rank crops by:", ["Average Yield (Ton/ha)", "Total Production (Tonnes)"], horizontal=True)

        if rank_metric == "Average Yield (Ton/ha)":
            crop_ranking = (
                filtered_df.groupby("crop")
                .agg(Metric=("crop_yield", "mean"), Records=("crop_year", "count"))
                .query("Records >= 30")
                .reset_index()
                .sort_values(by="Metric", ascending=False)
            )
            x_label = "Average Yield (Ton/ha)"
        else:
            crop_ranking = (
                filtered_df.groupby("crop")
                .agg(Metric=("production", "sum"))
                .reset_index()
                .sort_values(by="Metric", ascending=False)
            )
            x_label = "Total Production (Tonnes)"

        col_rank1, col_rank2 = st.columns(2)
        with col_rank1:
            st.markdown("Top 10 Performing Crops")
            fig_top10 = px.bar(
                crop_ranking.head(10),
                x="Metric",
                y="crop",
                orientation="h",
                labels={"Metric": x_label, "crop": "Crop"},
                color="Metric",
                color_continuous_scale="Greens"
            )
            fig_top10.update_layout(yaxis=dict(autorange="reversed"), showlegend=False, height=350, margin=dict(l=10, r=10, t=20, b=20))
            st.plotly_chart(fig_top10, use_container_width=True)

        with col_rank2:
            st.markdown("Bottom 10 Crops by Metric")
            fig_bot10 = px.bar(
                crop_ranking.tail(10),
                x="Metric",
                y="crop",
                orientation="h",
                labels={"Metric": x_label, "crop": "Crop"},
                color="Metric",
                color_continuous_scale="Reds"
            )
            fig_bot10.update_layout(yaxis=dict(autorange="reversed"), showlegend=False, height=350, margin=dict(l=10, r=10, t=20, b=20))
            st.plotly_chart(fig_bot10, use_container_width=True)

    # ==========================================
    # TAB 4: GEOGRAPHIC ANALYSIS
    # ==========================================
    with tab_geo:
        st.subheader("Geographic & State-Level Analysis")

        # Map visualization using actual latitude & longitude
        st.markdown("##### Agricultural Data Distribution Across India")
        map_metric = st.selectbox(
            "Color map points by:",
            ["crop_yield", "precipitation_sum", "temperature_2m_mean", "relative_humidity_2m_mean"],
            format_func=lambda x: {
                "crop_yield": "Crop Yield (Ton/ha)",
                "precipitation_sum": "Precipitation (mm)",
                "temperature_2m_mean": "Temperature (°C)",
                "relative_humidity_2m_mean": "Relative Humidity (%)"
            }.get(x, x)
        )

        map_df = (
            filtered_df.groupby(["state_name", "district_name", "latitude", "longitude"])
            .agg({
                "crop_yield": "mean",
                "precipitation_sum": "mean",
                "temperature_2m_mean": "mean",
                "relative_humidity_2m_mean": "mean",
                "production": "sum"
            })
            .reset_index()
        )

        fig_map = px.scatter_mapbox(
            map_df,
            lat="latitude",
            lon="longitude",
            color=map_metric,
            size="production",
            size_max=18,
            hover_name="district_name",
            hover_data={"state_name": True, "latitude": False, "longitude": False, map_metric: ":.2f"},
            color_continuous_scale="Viridis",
            zoom=3.8,
            center={"lat": 22.5937, "lon": 78.9629},
            mapbox_style="carto-positron"
        )
        fig_map.update_layout(height=450, margin=dict(l=0, r=0, t=10, b=10))
        st.plotly_chart(fig_map, use_container_width=True)

        st.markdown("---")

        # State Comparison Drilldown
        st.markdown("##### State-Level Agricultural Profile")
        state_summary = (
            filtered_df.groupby("state_name")
            .agg(
                Total_Production=("production", "sum"),
                Total_Area=("area", "sum"),
                Avg_Yield=("crop_yield", "mean"),
                Avg_Rainfall=("precipitation_sum", "mean"),
                Avg_Temp=("temperature_2m_mean", "mean"),
                Total_Districts=("district_name", "nunique")
            )
            .reset_index()
            .sort_values(by="Total_Production", ascending=False)
        )

        st.dataframe(
            state_summary.rename(columns={
                "state_name": "State",
                "Total_Production": "Total Production (Tonnes)",
                "Total_Area": "Total Area (ha)",
                "Avg_Yield": "Avg Yield (Ton/ha)",
                "Avg_Rainfall": "Avg Rainfall (mm)",
                "Avg_Temp": "Avg Temp (°C)",
                "Total_Districts": "Districts Count"
            }),
            use_container_width=True,
            hide_index=True
        )

        # District-level exploration for a selected state
        st.markdown("##### District-Level Breakdown for a State")
        drill_state = st.selectbox("Select State for District Breakdown:", sorted(filtered_df["state_name"].unique()))
        drill_df = filtered_df[filtered_df["state_name"] == drill_state]

        district_agg = (
            drill_df.groupby("district_name")
            .agg(
                Total_Production=("production", "sum"),
                Avg_Yield=("crop_yield", "mean"),
                Avg_Rainfall=("precipitation_sum", "mean")
            )
            .reset_index()
            .sort_values(by="Total_Production", ascending=False)
            .head(15)
        )

        fig_district = px.bar(
            district_agg,
            x="district_name",
            y="Total_Production",
            color="Avg_Yield",
            labels={"district_name": "District", "Total_Production": "Production (Tonnes)", "Avg_Yield": "Avg Yield"},
            color_continuous_scale="Teal"
        )
        fig_district.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=40))
        st.plotly_chart(fig_district, use_container_width=True)

    # ==========================================
    # TAB 5: WEATHER ANALYSIS
    # ==========================================
    with tab_weather:
        st.subheader("Meteorological Variable Analysis")

        col_w1, col_w2 = st.columns(2)
        with col_w1:
            st.markdown("##### Temperature Distribution Across Records")
            fig_temp_hist = px.histogram(
                filtered_df,
                x="temperature_2m_mean",
                nbins=40,
                labels={"temperature_2m_mean": "Temperature (°C)"},
                color_discrete_sequence=["#e6550d"]
            )
            fig_temp_hist.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=20))
            st.plotly_chart(fig_temp_hist, use_container_width=True)

        with col_w2:
            st.markdown("##### Precipitation Distribution")
            fig_precip_hist = px.histogram(
                filtered_df,
                x="precipitation_sum",
                nbins=40,
                labels={"precipitation_sum": "Precipitation (mm)"},
                color_discrete_sequence=["#3182bd"]
            )
            fig_precip_hist.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=20))
            st.plotly_chart(fig_precip_hist, use_container_width=True)

        col_w3, col_w4 = st.columns(2)
        with col_w3:
            st.markdown("##### Seasonal Weather Comparison")
            season_weather = (
                filtered_df.groupby("season")
                .agg(
                    Mean_Temp=("temperature_2m_mean", "mean"),
                    Mean_Precip=("precipitation_sum", "mean"),
                    Mean_Humidity=("relative_humidity_2m_mean", "mean"),
                    Mean_Wind=("wind_speed_10m_mean", "mean")
                )
                .reset_index()
            )
            fig_season_w = px.bar(
                season_weather,
                x="season",
                y=["Mean_Temp", "Mean_Humidity"],
                barmode="group",
                labels={"value": "Value", "season": "Season", "variable": "Metric"}
            )
            fig_season_w.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=20))
            st.plotly_chart(fig_season_w, use_container_width=True)

        with col_w4:
            st.markdown("##### Relative Humidity vs Wind Speed")
            sample_weather = filtered_df.sample(n=min(len(filtered_df), 3000), random_state=42)
            fig_hw = px.scatter(
                sample_weather,
                x="wind_speed_10m_mean",
                y="relative_humidity_2m_mean",
                color="season",
                labels={
                    "wind_speed_10m_mean": "Wind Speed (m/s)",
                    "relative_humidity_2m_mean": "Relative Humidity (%)",
                    "season": "Season"
                },
                opacity=0.6
            )
            fig_hw.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=20))
            st.plotly_chart(fig_hw, use_container_width=True)

    # ==========================================
    # TAB 6: CORRELATION & RELATIONSHIP ANALYSIS
    # ==========================================
    with tab_correlations:
        st.subheader("Agricultural & Weather Relationships")
        st.info("💡 Note: The charts and matrices below display observed statistical correlations and associations across historical data (2001–2020). Correlation does not imply direct causation.")

        num_cols = [
            "crop_yield",
            "area",
            "production",
            "temperature_2m_mean",
            "precipitation_sum",
            "relative_humidity_2m_mean",
            "wind_speed_10m_mean"
        ]

        corr_matrix = filtered_df[num_cols].corr()

        col_corr1, col_corr2 = st.columns([1.2, 1])
        with col_corr1:
            st.markdown("##### Correlation Matrix")
            fig_corr = px.imshow(
                corr_matrix,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                labels=dict(x="Variable", y="Variable", color="Correlation")
            )
            fig_corr.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=20))
            st.plotly_chart(fig_corr, use_container_width=True)

        with col_corr2:
            st.markdown("##### Variable Relationship Explorer")
            x_var = st.selectbox(
                "X-Axis Variable:",
                options=[c for c in num_cols if c != "crop_yield"],
                index=num_cols.index("precipitation_sum") - 1,
                format_func=lambda x: {
                    "temperature_2m_mean": "Temperature (°C)",
                    "precipitation_sum": "Precipitation (mm)",
                    "relative_humidity_2m_mean": "Relative Humidity (%)",
                    "wind_speed_10m_mean": "Wind Speed (m/s)",
                    "area": "Cultivated Area (ha)",
                    "production": "Production (Tonnes)"
                }.get(x, x)
            )

            y_var = st.selectbox(
                "Y-Axis Variable:",
                options=["crop_yield", "production"],
                index=0,
                format_func=lambda y: "Crop Yield (Ton/ha)" if y == "crop_yield" else "Production (Tonnes)"
            )

            sample_df = filtered_df.sample(n=min(len(filtered_df), 2000), random_state=42)
            fig_scatter = px.scatter(
                sample_df,
                x=x_var,
                y=y_var,
                color="season",
                labels={x_var: x_var.replace("_", " ").title(), y_var: y_var.replace("_", " ").title()},
                opacity=0.65
            )
            fig_scatter.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=20))
            st.plotly_chart(fig_scatter, use_container_width=True)
