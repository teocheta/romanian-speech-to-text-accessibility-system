# Romanian Speech-to-Text Accessibility System

A full-stack AI-powered speech-to-text system designed to improve classroom accessibility for students with hearing impairments.

Developed as a Bachelor’s Thesis project in Computer Science.

---

## Project Overview

This project implements a Romanian Automatic Speech Recognition (ASR) system using a custom deep learning architecture (CNN + BiLSTM + CTC) and integrates it into a complete transcription workflow consisting of:

- AI Model (training + inference)
- Backend API service
- Frontend user interface

The goal is to provide accessible, near real-time transcription support in educational environments.

---

## System Architecture

Frontend → Backend API → AI Model → Transcription Output

### 1. AI Module (`/speech_to_text`)

Implemented in **PyTorch**, the model follows a hybrid deep learning architecture:

- CNN layers for feature extraction from MFCC-based audio representations
- Bidirectional LSTM layers for temporal sequence modeling
- Fully connected projection layer
- Log-Softmax output compatible with CTC-based training

### Model Highlights

- Handles variable-length audio inputs
- Learns phonetic and contextual speech patterns
- Designed specifically for Romanian language speech

Evaluation was performed using **Word Error Rate (WER)**.

---

### 2. Backend (`/transcriere_audio_backend`)

- Handles audio file uploads
- Processes audio for inference
- Integrates custom ASR model
- Provides REST API endpoints for transcription
- Supports transcript export

---

### 3. Frontend (`/transcriere_audio`)

- User-friendly interface
- Audio recording or file upload
- Live transcription display
- Designed for classroom usability

---

## Technologies Used

- Python
- PyTorch
- NumPy / Pandas
- Librosa (audio preprocessing)
- Deep Learning (CNN + BiLSTM + CTC)
- REST API
- React Native (Expo)
- TypeScript


---

## Datasets

The model was trained and evaluated on Romanian speech data including:

- Mozilla Common Voice (Romanian subset)
- Librivox Romanian audiobooks
- Additional curated audio samples




