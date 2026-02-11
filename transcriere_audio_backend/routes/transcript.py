from flask import Blueprint, jsonify
import os


transcript_bp = Blueprint('transcript', __name__)  # fără prefix
@transcript_bp.route('/transcript/<session_id>', methods=['GET'])
def get_transcript(session_id):
    path = os.path.join('uploads', session_id, 'transcript.txt')
    if not os.path.exists(path):
        return jsonify({'transcript': ''})

    with open(path, 'r', encoding='utf-8') as f:
        return jsonify({'transcript': f.read()})
