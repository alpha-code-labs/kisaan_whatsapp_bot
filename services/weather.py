import datetime
import requests
from services.config import Config
from services.graph_api import GraphApi

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


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

    message = "7-day weather summary:\n"
    total_rain = 0
    dry_streak = 0

    for index, day in enumerate(daily):
        rain = round(day.get("rain", 0))
        total_rain += rain

        if index == 0:
            day_label = "Today"
        elif index == 1:
            day_label = "Tomorrow"
        else:
            date = datetime.datetime.fromtimestamp(day.get("dt", 0))
            day_label = DAYS[date.weekday()]

        if rain == 0 and index == dry_streak:
            dry_streak += 1

        message += f"\n{day_label}\nRain: {rain} mm\n"

    if daily and daily[0].get("rain", 0) > 0:
        advice = "Rain expected today. Avoid irrigation if possible."
    elif dry_streak >= 5:
        advice = "Dry streak detected. Consider irrigation planning."
    else:
        advice = "Light rain expected. Monitor field conditions."

    message += (
        f"\nSummary:\n"
        f"Total rain (7 days): {total_rain} mm\n"
        f"Dry days in a row: {dry_streak}\n"
        f"Advice: {advice}"
    )

    return GraphApi.message_text(sender_phone_number_id, recipient_phone_number, message)