import json
import os
from openai import OpenAI
from services.config import Config

_client = OpenAI(api_key=Config.openai_api_key)


def detect_crop(query):
    crops_path = os.path.join(Config.data_dir, "crops.json")
    with open(crops_path, "r", encoding="utf-8") as f:
        crops = json.load(f)

    crop_list = ", ".join(crops.keys())

    response = _client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": (
                f"List of crops: {crop_list}.\n"
                f"User query: {query}.\n"
                "Identify if the query mentions any of these crops. "
                "Respond with the exact name or 'none'. Output only with one word either the crop name or none."
            )
        }],
        temperature=0
    )

    result = response.choices[0].message.content.strip()
    return "none" if result.lower() == "none" else result