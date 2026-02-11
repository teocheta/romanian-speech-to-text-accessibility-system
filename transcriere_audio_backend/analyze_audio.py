from pydub import AudioSegment


def analyze_audio(file_path):
    audio = AudioSegment.from_file(file_path)

    duration_sec = len(audio) / 1000
    loudness_db = audio.dBFS

    print(f"Durată: {duration_sec:.2f} sec | Volum mediu: {loudness_db:.2f} dBFS")

    low_quality = False
    issues = []

    if duration_sec < 1:
        issues.append("Înregistrare prea scurtă (<1s)")
        low_quality = True

    if loudness_db < -35:
        issues.append("Volum foarte slab (< -35 dBFS)")
        low_quality = True

    return {
        "duration_sec": round(duration_sec, 2),
        "loudness_db": round(loudness_db, 2),
        "issues": issues,
        "low_quality": low_quality
    }
