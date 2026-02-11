from flask import Blueprint, request, jsonify
import os
import time
from werkzeug.utils import secure_filename

from analyze_audio import analyze_audio
from asr_model.transcribe_audio import transcribe_audio
from asr_model.whisper_backend import transcribe_with_whisper

upload_bp = Blueprint('upload', __name__, url_prefix='/upload')

UPLOAD_ROOT = 'uploads'


@upload_bp.route('', methods=['POST'])
def upload_audio():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # 1. Preia sesiunea
    session_id = request.form.get('session')
    if not session_id:
        return jsonify({'error': 'Missing session ID'}), 400

    # 2. Creează folderul sesiunii dacă nu există
    session_folder = os.path.join(UPLOAD_ROOT, secure_filename(session_id))
    os.makedirs(session_folder, exist_ok=True)

    # 3. Creează un nume unic pentru fișier
    timestamp = int(time.time() * 1000)
    filename = f"{timestamp}_{secure_filename(file.filename)}"

    file_path = os.path.join(session_folder, filename)
    file.save(file_path)

    print(f"Fișier salvat: {file_path}")

    # 4. Află modelul selectat
    model = request.form.get('model', 'custom')
    print(f"Model selectat pentru transcriere: {model}")

    audio_info = analyze_audio(file_path)
    # 5. Transcriere cu modelul corespunzător
    if model == 'whisper':
        transcript = transcribe_with_whisper(file_path)
    else:
        transcript = transcribe_audio(file_path)
    #  6. Salvăm și în transcript.txt
    transcript_path = os.path.join(session_folder, 'transcript.txt')
    with open(transcript_path, 'a', encoding='utf-8') as f:
        f.write(transcript + '\n')

    return jsonify({
        'message': 'Fișier încărcat cu succes.',
        'transcript': transcript,
        'path': file_path,
        'info': audio_info
    })
