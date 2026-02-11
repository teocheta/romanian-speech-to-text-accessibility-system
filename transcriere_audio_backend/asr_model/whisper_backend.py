import whisper

model = whisper.load_model("base")


def transcribe_with_whisper(audio_path: str) -> str:
    print(f"Whisper transcrie: {audio_path}")
    result = model.transcribe(audio_path, language='ro')
    return result.get("text", "")
