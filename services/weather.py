import datetime
import requests
from services.config import Config
from services.graph_api import GraphApi

DAYS_HI = ["सोमवार", "मंगलवार", "बुधवार", "गुरुवार", "शुक्रवार", "शनिवार", "रविवार"]


def _ms_to_kmh(ms: float) -> int:
    return int(round((ms or 0) * 3.6))


def send_weather(sender_phone_number_id, recipient_phone_number, location):
    lat = location.get("latitude")
    lon = location.get("longitude")

    url = (
        f"https://api.openweathermap.org/data/3.0/onecall?lat={lat}&lon={lon}"
        f"&units=metric&exclude=minutely,hourly,alerts&appid={Config.weather_api_key}"
    )

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    daily = response.json().get("daily", [])[:7]

    message = "🌦️ *7 दिन का मौसम*\n"
    total_rain = 0
    dry_streak = 0

    for index, day in enumerate(daily):
        rain_mm = round(day.get("rain", 0) or 0)
        total_rain += rain_mm

        clouds_pct = int(round(day.get("clouds", 0) or 0))
        wind_kmh = _ms_to_kmh(day.get("wind_speed", 0) or 0)

        if index == 0:
            day_label = "आज"
        elif index == 1:
            day_label = "कल"
        else:
            date = datetime.datetime.fromtimestamp(day.get("dt", 0))
            day_label = DAYS_HI[date.weekday()]

        if rain_mm == 0 and index == dry_streak:
            dry_streak += 1

        message += (
            f"\n➤ *{day_label}*\n"
            f"🌧️ {rain_mm} mm\n"
            f"☁️ {clouds_pct}%\n"
            f"💨 {wind_kmh} km/h\n"
        )

    if daily and (daily[0].get("rain", 0) or 0) > 0:
        advice = "आज बारिश की संभावना है। संभव हो तो सिंचाई न करें।"
    elif dry_streak >= 5:
        advice = "लगातार सूखे दिन दिख रहे हैं। सिंचाई की योजना बनाएं।"
    else:
        advice = "हल्की बारिश की संभावना है। खेत की स्थिति पर नजर रखें।"

    message += (
        f"\n📊 *सारांश*\n"
        f"🌧️ कुल बारिश: {total_rain} mm\n"
        f"☀️ लगातार सूखे दिन: {dry_streak}\n"
        f"🌱 सलाह: {advice}"
    )

    return GraphApi.message_text(sender_phone_number_id, recipient_phone_number, message)
