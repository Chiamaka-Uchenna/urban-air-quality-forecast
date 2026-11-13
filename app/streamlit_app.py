"""Next-day air quality forecasts for ten cities: interactive app with live data.

Run with:  streamlit run app/streamlit_app.py
"""
import math

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from common import BLUE, GREY, ORANGE, bar, lines, load, predict, show, style

st.set_page_config(page_title="Tomorrow's air quality in ten cities", layout="wide")
model, s = load("model.json"), load("summary.json")
cities = s["cities"]
gbm = next(m for m in s["metrics"] if m["model"] == "Gradient boosting")
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def level(value):
    if value <= 15:
        return "Within WHO daily guideline"
    if value <= 37.5:
        return "Moderate"
    return "High" if value <= 75 else "Very high"


@st.cache_data(ttl=3600, show_spinner="Fetching the latest pollution and weather data...")
def fetch_live():
    """One request for hourly PM2.5 and one for the weather forecast, covering all ten cities."""
    latitude = ",".join(str(c["lat"]) for c in cities)
    longitude = ",".join(str(c["lon"]) for c in cities)
    air = requests.get("https://air-quality-api.open-meteo.com/v1/air-quality", timeout=60, params={
        "latitude": latitude, "longitude": longitude, "hourly": "pm2_5", "past_days": 9, "forecast_days": 2, "timezone": "auto"}).json()
    weather = requests.get("https://api.open-meteo.com/v1/forecast", timeout=60, params={
        "latitude": latitude, "longitude": longitude, "past_days": 1, "forecast_days": 2, "timezone": "auto",
        "daily": "temperature_2m_mean,precipitation_sum,wind_speed_10m_mean,relative_humidity_2m_mean"}).json()
    return air, weather


def forecast_city(index, air, weather):
    """Build the same features as the notebook from live data and score the model."""
    hourly = pd.DataFrame(air["hourly"]).dropna()
    hourly["day"] = hourly["time"].str[:10]
    daily = hourly.groupby("day")["pm2_5"].agg(["mean", "count"])
    daily = daily[daily["count"] >= 18]["mean"]
    w = weather["daily"]
    today, tomorrow = w["time"][1], w["time"][2]
    history = daily[daily.index <= today]
    if len(history) < 7 or history.index[-1] != today:
        return None
    pm = history.to_numpy()
    logs = [math.log(max(v, 0.5)) for v in pm]
    week = pd.Series(logs[-7:])
    target = pd.Timestamp(tomorrow)

    def weather_on(k):
        return {"temp": w["temperature_2m_mean"][k], "rain": w["precipitation_sum"][k] or 0.0,
                "wind": w["wind_speed_10m_mean"][k], "humidity": w["relative_humidity_2m_mean"][k]}

    now, nxt = weather_on(1), weather_on(2)
    values = {
        "l1": logs[-1], "l2": logs[-2], "l3": logs[-3], "m7": math.log(pm[-7:].mean()), "s7": float(week.std()), "d1": logs[-1] - logs[-2],
        "temp_today": now["temp"], "humidity_today": now["humidity"], "wind_today": now["wind"], "rain_today": math.log1p(now["rain"]),
        "temp_tom": nxt["temp"], "humidity_tom": nxt["humidity"], "wind_tom": nxt["wind"], "rain_tom": math.log1p(nxt["rain"]),
        "doy_sin": math.sin(2 * math.pi * target.dayofyear / 365.25), "doy_cos": math.cos(2 * math.pi * target.dayofyear / 365.25),
        "dow": target.dayofweek,
    }
    values.update({f"city_{j}": int(j == index) for j in range(len(cities))})
    features = [values[name] for name in model["features"]]
    if any(v is None for v in features):
        return None
    value = math.exp(values["l1"] + predict(model, features))
    return {"today": float(pm[-1]), "forecast": value, "low": value * math.exp(s["interval"]["low"]), "high": value * math.exp(s["interval"]["high"]),
            "date": target, "history": history.tail(9), "weather": nxt, "copernicus": float(daily[tomorrow]) if tomorrow in daily.index else None}


st.title("How clean will the air be tomorrow?")
st.caption("Live next-day forecasts of fine particle pollution (PM2.5) for ten major cities, from a model trained on four years of Copernicus data.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("City-days analysed", f"{s['city_days']:,}")
c2.metric("Cities above the WHO guideline", "10 of 10")
c3.metric("Forecast error against baseline", f"-{s['improvement']:.0%}")
c4.metric("Typical forecast error", f"{gbm['Median error (%)']:.0f}%")

tab_live, tab_cities, tab_model = st.tabs(["Live forecast", "How polluted are these cities", "How good is the forecast"])

with tab_live:
    live = {}
    try:
        air, weather = fetch_live()
        for i, city in enumerate(cities):
            result = forecast_city(i, air[i], weather[i])
            if result:
                live[city["name"]] = result
    except Exception:
        live = {}
    if not live:
        st.warning("The live data service could not be reached, so no forecast is shown. The other tabs show the model's tested performance.")
    else:
        st.caption("Forecasts for tomorrow in each city's local time, computed from the latest Copernicus and weather data. Daily average PM2.5 in micrograms per cubic metre.")
        thresholds = {c["name"]: c["threshold"] for c in cities}
        table = pd.DataFrame([{"City": name, "Today": r["today"], "Tomorrow (forecast)": r["forecast"], "Likely low": r["low"], "Likely high": r["high"],
                               "Times WHO guideline": r["forecast"] / 15, "Outlook": level(r["forecast"]) + (" · warning" if r["forecast"] > thresholds[name] else "")}
                              for name, r in live.items()]).sort_values("Tomorrow (forecast)", ascending=False)
        st.dataframe(table.style.format({"Today": "{:.0f}", "Tomorrow (forecast)": "{:.0f}", "Likely low": "{:.0f}", "Likely high": "{:.0f}", "Times WHO guideline": "{:.1f}x"}),
                     hide_index=True, width="stretch")
        name = st.selectbox("City", list(live), index=list(live).index("Lagos") if "Lagos" in live else 0)
        r = live[name]
        left, right = st.columns([1, 2])
        left.metric(f"{name}, {r['date']:%A %d %B}", f"{r['forecast']:.0f} µg/m³", level(r["forecast"]), delta_color="off")
        right.write(f"Likely range {r['low']:.0f} to {r['high']:.0f}. Today is {r['today']:.0f}. That is {r['forecast'] / 15:.1f} times the WHO 24-hour guideline of 15.")
        right.write(f"Tomorrow's weather: wind {r['weather']['wind']:.0f} km/h, rain {r['weather']['rain']:.1f} mm."
                    + (f" Copernicus's own forecast for tomorrow is {r['copernicus']:.0f}." if r["copernicus"] is not None else ""))
        figure = go.Figure([go.Bar(x=list(r["history"].index), y=list(r["history"].values), name="Observed daily mean", marker_color=BLUE),
                            go.Bar(x=[r["date"].strftime("%Y-%m-%d")], y=[r["forecast"]], name="Forecast for tomorrow", marker_color=ORANGE,
                                   error_y=dict(type="data", symmetric=False, array=[r["high"] - r["forecast"]], arrayminus=[r["forecast"] - r["low"]]))])
        figure.add_hline(y=15, line_dash="dot", line_color=GREY, annotation_text="WHO 24-hour guideline")
        figure.update_yaxes(title="PM2.5 (µg/m³)")
        show(style(figure))

with tab_cities:
    stats = s["stats"][::-1]
    left, right = st.columns(2)
    with left:
        st.markdown("**Average PM2.5** (dotted line: WHO annual guideline of 5)")
        show(bar([r["city"] for r in stats], [r["mean_pm25"] for r in stats], "Mean PM2.5 (µg/m³)", "", horizontal=True, reference=5))
    with right:
        st.markdown("**Days above the WHO 24-hour guideline of 15**")
        show(bar([r["city"] for r in stats], [r["days_above_guideline"] * 100 for r in stats], "Share of days (%)", "", horizontal=True))
    st.markdown("**When is the air worst** (monthly mean PM2.5; colour is relative to each city's own average)")
    monthly = s["monthly"]
    figure = go.Figure(go.Heatmap(z=monthly["relative"], x=MONTHS, y=monthly["cities"], text=[[f"{v:.0f}" for v in row] for row in monthly["values"]],
                                  texttemplate="%{text}", colorscale="Blues", zmin=0.3, zmax=2.2, xgap=2, ygap=2, colorbar=dict(title="Ratio")))
    figure.update_yaxes(autorange="reversed")
    show(style(figure, 440))
    left, right = st.columns(2)
    with left:
        st.markdown("**Wind clears the air**")
        show(bar(s["by_wind"]["labels"], s["by_wind"]["values"], "", "PM2.5 relative to city average", reference=1))
    with right:
        st.markdown("**So does rain**")
        show(bar(["Dry day", "Rain over 1 mm"], [s["by_rain"]["dry"], s["by_rain"]["rain"]], "", "PM2.5 relative to city average", reference=1))

with tab_model:
    st.caption("Trained on data to October 2025 and tested on the following 12 months. Persistence assumes tomorrow will be the same as today.")
    left, right = st.columns(2)
    with left:
        st.markdown("**Forecast error by city** (mean absolute error, µg/m³)")
        rows = s["by_city"][::-1]
        figure = go.Figure([go.Bar(y=[r["city"] for r in rows], x=[r["persistence_mae"] for r in rows], name="Persistence", orientation="h", marker_color=BLUE),
                            go.Bar(y=[r["city"] for r in rows], x=[r["model_mae"] for r in rows], name="Gradient boosting", orientation="h", marker_color=ORANGE)])
        show(style(figure, 440, barmode="group"))
        st.markdown("**Scores on the held-out year**")
        st.dataframe(pd.DataFrame(s["metrics"]).round(3), hide_index=True, width="stretch")
    with right:
        recent_city = st.selectbox("Forecast against outcome, last 120 days of the test period", [c["name"] for c in cities], index=5)
        recent = s["recent"][recent_city]
        show(lines([("Actual", recent["dates"], recent["actual"], BLUE, "solid"), ("Forecast made the day before", recent["dates"], recent["forecast"], ORANGE, "solid")],
                   "", "PM2.5 (µg/m³)", height=370))
        st.markdown("**As a warning for high pollution days**")
        st.dataframe(pd.DataFrame(s["alerts"]).round(3), hide_index=True, width="stretch")
    st.markdown("**What drives the forecast**")
    importance = s["importance"][::-1]
    show(bar([r["feature"] for r in importance], [r["value"] for r in importance], "Mean absolute SHAP value", "", horizontal=True, height=460))
    st.caption("PM2.5 comes from the Copernicus Atmosphere Monitoring Service global analysis, a satellite-constrained model on a grid of about 40 km. "
               "It is not a ground measurement and understates roadside peaks.")
