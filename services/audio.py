import os
import subprocess
import tempfile
from openai import OpenAI
from services.config import Config
from services.language import normalize_to_hinglish, normalize_to_english

_client = OpenAI(api_key=Config.openai_api_key)


def _write_temp_file(buffer_bytes, suffix):
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(buffer_bytes)
    return path


def _convert_ogg_to_wav(input_path):
    output_path = input_path.replace(".ogg", ".wav")
    cmd = ["ffmpeg", "-y", "-i", input_path, "-ar", "16000", "-ac", "1", output_path]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_path


def _transcribe_audio(wav_path):
    with open(wav_path, "rb") as f:
        response = _client.audio.transcriptions.create(
            file=f,
            model="whisper-1",
            response_format="verbose_json"
        )
    return {"text": response.text, "language": response.language}


def process_voice_note(audio_buffer):
    ogg_path = None
    wav_path = None

    try:
        ogg_path = _write_temp_file(audio_buffer, ".ogg")
        wav_path = _convert_ogg_to_wav(ogg_path)
        transcript = _transcribe_audio(wav_path)

        hinglish = normalize_to_hinglish(transcript["text"], transcript["language"])
        english = normalize_to_english(transcript["text"], transcript["language"])

        return {
            "transcript": transcript["text"],
            "hinglish": hinglish,
            "english": english,
            "detectedLanguage": transcript["language"]
        }
    finally:
        if ogg_path and os.path.exists(ogg_path):
            os.remove(ogg_path)
        if wav_path and os.path.exists(wav_path):
            os.remove(wav_path)