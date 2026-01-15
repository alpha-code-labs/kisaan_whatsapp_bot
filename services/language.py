from openai import OpenAI
from services.config import Config

_client = OpenAI(api_key=Config.openai_api_key)


def normalize_to_hinglish(text, language):
    response = _client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Convert the following text into clean Hinglish suitable for NLP. Preserve agricultural terms."
            },
            {"role": "user", "content": text}
        ]
    )
    return response.choices[0].message.content


def normalize_to_english(text, language):
    if language == "en":
        return text

    response = _client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Convert the following text into clean English suitable for NLP. Preserve agricultural terms."
            },
            {"role": "user", "content": text}
        ]
    )
    return response.choices[0].message.content