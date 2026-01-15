from openai import OpenAI
from services.config import Config

_client = OpenAI(api_key=Config.openai_api_key)


def analyze_image(image_buffer, mime_type="image/jpeg"):
    import base64

    base64_image = base64.b64encode(image_buffer).decode("ascii")
    image_data_url = f"data:{mime_type};base64,{base64_image}"

    response = _client.responses.create(
        model="gpt-5.2-pro",
        input=[{
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": (
                        "Analyze the image and identify what agricultural entities are visibly present. "
                        "Use tags/keywords. Do not infer causes or treatments. Return tags separated by commas."
                    )
                },
                {"type": "input_image", "image_url": image_data_url}
            ]
        }]
    )

    tags = (response.output_text or "").strip()
    return {"tags": tags}