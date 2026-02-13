# 🚨 Accident Detection System

An AI-powered system that uses computer vision (ResNet18) to detect traffic accidents in video footage and automatically triggers emergency alerts via Twilio (SMS & Voice).

## ✨ Features

- **AI-Powered Detection**: Uses a ResNet18 model fine-tuned for accident detection.
- **Real-Time Analysis**: Processes video footage to identify accidents, collisions, and anomalous traffic behavior.
- **Emergency Alerts**: Automatically sends SMS and makes voice calls to emergency contacts via Twilio when high-risk incidents are detected.
- **Interactive Dashboard**: User-friendly Streamlit interface for video upload, monitoring, and configuring sensitivity thresholds.
- **Risk Assessment**: Calculates confidence scores and classifies incidents by severity (Low, Moderate, High, Critical).

## 🛠️ Tech Stack

- **Frontend**: React 
- **AI/ML**: PyTorch, Torchvision (ResNet18), OpenCV
- **Communication**: Twilio API (SMS & Voice)
- **Language**: Python

## 📋 Prerequisites

- Python 3.8+
- A Twilio Account (for alerting features)
- verify `ffmpeg` is installed (optional, but good for video processing)

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd AccPred
   ```

2. **Create a virtual environment (Recommended)**
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # Mac/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## ⚙️ Configuration

1. Create a `.env` file in the root directory.
2. Add your Twilio credentials:

   ```env
   TWILIO_ACCOUNT_SID=your_account_sid
   TWILIO_AUTH_TOKEN=your_auth_token
   TWILIO_FROM_NUMBER=your_twilio_number
   EMERGENCY_NUMBER=verified_target_number
   ```

   > **Note**: For the trial Twilio account, the `EMERGENCY_NUMBER` must be a verified caller ID.

## 🚦 Usage

1. **Start the Application**
   ```bash
   streamlit run app.py
   ```

2. **Using the Dashboard**
   - **Configuration**: Use the sidebar to set the "Camera Location" and adjust the "Detection Confidence Threshold".
   - **Upload**: Upload a traffic video file (`.mp4`, `.avi`, `.mov`, `.mkv`).
   - **Analyze**: Click **Start Analysis**.
   - **Results**: The system will display the video, highlighting any detected accidents. If a high-risk event is found, it will trigger the SOS system (Twilio).

## 📂 Project Structure

- `app.py`: Main Streamlit dashboard application.
- `app_prediction.py`: Core logic for accident prediction (`AccidentPredictionEnsemble`) and Twilio integration (`TwilioAlert`).
- `api_server.py`: Flask API server (alternative backend).
- `requirements.txt`: Python dependencies.
- `debug_video.py`, `inspect_model.py`: Utility scripts for debugging.

## ⚠️ Disclaimer

This project is a **prototype** developed for demonstration purposes. It uses heuristics and a basic AI model. For a production-grade safety system, it would require training on a large-scale real-world accident dataset, integration with live camera feeds, and rigorous validation.
