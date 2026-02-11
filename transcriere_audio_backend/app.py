from flask import Flask
from flask_cors import CORS

from routes.transcript import transcript_bp
from routes.upload import upload_bp
import os

app = Flask(__name__)
CORS(app, origins=["http://localhost:8081"])

app.register_blueprint(upload_bp)
app.register_blueprint(transcript_bp)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
