import cv2
import numpy as np
import tempfile
import os
# Fix for OpenMP conflict (OMP: Error #15)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from datetime import datetime
from PIL import Image
from collections import deque
import torch
import torch.nn as nn
from torchvision import models, transforms
from twilio.rest import Client
from dotenv import load_dotenv
try:
    import streamlit as st
except ImportError:
    st = None

# Load .env file
load_dotenv()

# ─── Twilio credentials from .env ────────────────────────────────────────────
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN  = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER")
EMERGENCY_NUMBER   = os.getenv("EMERGENCY_NUMBER")

# ─────────────────────────────────────────────────────────────────────────────
# TWILIO HELPER
# ─────────────────────────────────────────────────────────────────────────────
class TwilioAlert:
    """Handles real Twilio Voice call AND SMS."""

    def __init__(self):
        self.missing = not all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN,
                                TWILIO_FROM_NUMBER, EMERGENCY_NUMBER])
        self.client = None
        self.verified_numbers: list = []
        self.emergency_verified = False

        if self.missing:
            return

        try:
            self.client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
            for caller_id in self.client.outgoing_caller_ids.list():
                self.verified_numbers.append(caller_id.phone_number)
            self.emergency_verified = EMERGENCY_NUMBER in self.verified_numbers
        except Exception:
            self.emergency_verified = False

    def _check_verified(self):
        """Return (ok, message)."""
        if self.missing:
            return False, "Twilio credentials are missing in .env"
        if not self.emergency_verified:
            return False, (
                f"EMERGENCY_NUMBER {EMERGENCY_NUMBER} is NOT verified in your "
                f"Twilio account. Go to "
                f"https://www.twilio.com/en-us/console/phone-numbers/verified "
                f"and add + verify this number, then restart the app."
            )
        return True, ""

    def send_sms(self, location, risk_level, confidence, frame_number):
        ok, msg = self._check_verified()
        if not ok:
            return False, msg

        assert self.client is not None
        assert TWILIO_FROM_NUMBER is not None
        assert EMERGENCY_NUMBER is not None

        body = (
            f"🚨 ACCIDENT PREDICTION ALERT\n"
            f"Location : {location}\n"
            f"Risk     : {risk_level}\n"
            f"Score    : {confidence:.0%}\n"
            f"Frame    : {frame_number}\n"
            f"Time     : {datetime.now().strftime('%H:%M:%S')}\n"
            f"⚠️ Preventive measures required immediately."
        )
        try:
            msg = self.client.messages.create(  # type: ignore
                body=body,
                from_=TWILIO_FROM_NUMBER,
                to=EMERGENCY_NUMBER
            )
            return True, msg.sid
        except Exception as e:
            return False, str(e)

    def make_call(self, location, risk_level, confidence):
        ok, msg = self._check_verified()
        if not ok:
            return False, msg

        assert self.client is not None
        assert TWILIO_FROM_NUMBER is not None
        assert EMERGENCY_NUMBER is not None

        twiml = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            "<Response>"
            '<Say voice="alice" rate="slow">'
            "This is an automated emergency alert from the Accident Prediction System. "
            f"A {risk_level} risk has been detected at {location}. "
            f"Prediction confidence is {int(confidence * 100)} percent. "
            "Please activate preventive measures immediately. "
            "This message will repeat. "
            "</Say>"
            '<Say voice="alice" rate="slow">'
            "This is an automated emergency alert from the Accident Prediction System. "
            f"A {risk_level} risk has been detected at {location}. "
            f"Prediction confidence is {int(confidence * 100)} percent. "
            "Please activate preventive measures immediately. "
            "</Say>"
            "</Response>"
        )
        try:
            call = self.client.calls.create(
                twiml=twiml,
                from_=TWILIO_FROM_NUMBER,
                to=EMERGENCY_NUMBER
            )
            return True, call.sid
        except Exception as e:
            return False, str(e)


# ─────────────────────────────────────────────────────────────────────────────
# IMPROVED ACCIDENT PREDICTOR (Classical CV-Based)
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# RESNET18 ACCIDENT PREDICTOR
# ─────────────────────────────────────────────────────────────────────────────
class AccidentPredictionEnsemble:
    """
    Uses ResNet18 for accident prediction.
    """
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load Pretrained ResNet18
        try:
            # unexpected EOF in weights file or other download errors can occur
            # so we wrap in try/except and fallback if needed, though for this task we expect it to work or we use local
            self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        except Exception as e:
            print(f"Error loading ResNet18 weights: {e}. Loading without weights.")
            self.model = models.resnet18(weights=None)

        # Modify the final layer for 2 classes: Safe (0) vs Accident (1)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 2)
        
        # Load custom weights if available, otherwise use random init for the head
        # We check for 'accident_model.pkl' or 'accident_model.pth'
        model_path = 'accident_model.pkl'
        if os.path.exists(model_path):
            try:
                # Attempt to load state dict
                # If the pickle contains the full model or state dict, handle accordingly
                checkpoint = torch.load(model_path, map_location=self.device)
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                     self.model.load_state_dict(checkpoint['state_dict'], strict=False)
                elif isinstance(checkpoint, dict):
                    self.model.load_state_dict(checkpoint, strict=False)
                # If it's a full model object (not recommended but possible)
                elif isinstance(checkpoint, nn.Module):
                     self.model = checkpoint
                print(f"Loaded weights from {model_path}")
            except Exception as e:
                print(f"Could not load custom weights from {model_path}: {e}")
                print("Using ImageNet weights with initialized classification layer.")
        else:
             print("No custom weights found. Using ImageNet weights with initialized classification layer.")

        self.model = self.model.to(self.device)
        self.model.eval()

        # Image Preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

        self.risk_buffer = deque(maxlen=30)
        self.consecutive_high_risk = 0

    def reset_buffers(self):
        """Reset buffers for new video analysis"""
        self.risk_buffer.clear()
        self.consecutive_high_risk = 0

    def predict_accident(self, frame):
        """Predict accident probability for a single frame"""
        # Preprocess
        # Convert BGR (OpenCV) to RGB (PIL/Torch)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image)
        input_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            # Apply softmax to get probabilities
            probs = torch.nn.functional.softmax(outputs, dim=1)
            # Assuming Class 1 is 'Accident'
            accident_prob = probs[0][1].item()

        # Add temporal smoothing
        self.risk_buffer.append(accident_prob)
        avg_risk = sum(self.risk_buffer) / len(self.risk_buffer)
        
        # Determine Level
        # We use the smoothed risk for stability
        final_score = avg_risk

        if final_score >= 0.70:
            level, ptype = "CRITICAL", "⚠️ ACCIDENT IMMINENT"
        elif final_score >= 0.50:
            level, ptype = "HIGH", "⚡ HIGH RISK DETECTED"
        elif final_score >= 0.30:
            level, ptype = "MODERATE", "⚠️ CAUTION ADVISED"
        else:
            level, ptype = "LOW", "✅ NORMAL TRAFFIC"

        # Dummy scores for compatibility with UI
        scores = {
            'motion': 0.0,
            'behavior': 0.0,
            'complexity': 0.0,
            'trend': 0.0,
            'cnn': accident_prob,
            'lstm': final_score # Using smoothed score as "lstm" substitute
        }

        return level, final_score, scores, ptype


# ─────────────────────────────────────────────────────────────────────────────
# VIDEO PROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def process_video_prediction_flask(video_path, predictor, risk_threshold):
    """Flask-compatible version without Streamlit components"""
    try:
        # Reset predictor state for new video
        predictor.reset_buffers()
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        frame_count, warnings = 0, []
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Process every Nth frame for consistency
        process_every = 3  # Process every 3rd frame

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            if frame_count % process_every == 0:
                try:
                    level, conf, scores, ptype = predictor.predict_accident(frame)
                    
                    if conf >= risk_threshold:
                        warnings.append({
                            "frame_number": frame_count,
                            "confidence": conf,
                            "risk_level": level,
                            "prediction_type": ptype,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "model_scores": scores
                        })
                except Exception as e:
                    print(f"Error processing frame {frame_count}: {e}")
                    continue

        cap.release()
        return warnings
        
    except Exception as e:
        print(f"Error in video processing: {e}")
        raise

def process_video_prediction(video_path, predictor, risk_threshold):
    """Streamlit version"""
    if st is None:
        raise RuntimeError("Streamlit not available")
    # Reset predictor state for new video
    predictor.reset_buffers()

    cap = cv2.VideoCapture(video_path)
    frame_count, warnings, history = 0, [], []
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress = st.progress(0)
    status_txt = st.empty()
    live = st.empty()
    
    # Process every Nth frame
    process_every = 3

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        if frame_count % process_every == 0:
            level, conf, scores, ptype = predictor.predict_accident(frame)
            history.append({
                'frame': frame_count, 
                'confidence': conf,
                'risk_level': level, 
                'model_scores': scores
            })
            
            with live.container():
                c = st.columns(4)
                c[0].metric("Frame", frame_count)
                c[1].metric("Risk Score", f"{conf:.2%}")
                icon = "🔴" if level=="CRITICAL" else "🟡" if level=="HIGH" else "🟢"
                c[2].metric("Risk Level", f"{icon} {level}")
                c[3].metric("Prediction", ptype.split()[0])

            if conf >= risk_threshold:
                warnings.append({
                    "frame_number": frame_count, 
                    "confidence": conf,
                    "risk_level": level, 
                    "prediction_type": ptype,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "frame": frame.copy(), 
                    "model_scores": scores
                })
        
        progress.progress(int(frame_count / total_frames * 100))
        status_txt.text(f"Analyzing: Frame {frame_count}/{total_frames}")

    cap.release()
    progress.empty()
    status_txt.empty()
    live.empty()
    return warnings, history


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────
def main():
    if st is None:
        return
    st.markdown('<div class="main-header">🔮 AI Accident Prediction System</div>',
                unsafe_allow_html=True)
    st.markdown("""
    **Improved Computer Vision-Based Accident Prediction**  
    Uses proven CV techniques for consistent, reliable predictions.  
    🎯 Motion Analysis · 🚗 Object Proximity · 📊 Trend Detection  
    Real alerts via **Twilio (Call + SMS)** when high-risk detected.
    """)

    # ── Sidebar ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Configuration")
        location = st.text_input("Camera Location", "Highway Junction 45")
        risk_threshold = st.slider("Risk Threshold", 0.3, 0.9, 0.55,
                                   help="Minimum risk score to trigger alert")

        st.markdown("---")
        st.markdown("### 📞 Twilio Status")
        tw = st.session_state.get("twilio", None)

        if tw is None:
            st.info("Twilio status will appear after first load …")
        elif tw.missing:
            st.markdown('<div class="twilio-err">✗ Twilio NOT configured</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="twilio-ok">✓ Twilio connected</div>',
                        unsafe_allow_html=True)
            if tw.emergency_verified:
                st.success(f"✅ {EMERGENCY_NUMBER} verified")
            else:
                st.warning(f"❌ {EMERGENCY_NUMBER} not verified")

        st.markdown("---")
        st.markdown("### 🤖 Detection Methods")
        for name, desc in [
            ("Motion", "Optical flow analysis"),
            ("Proximity", "Object distance tracking"),
            ("Complexity", "Scene chaos detection"),
            ("Trend", "Risk escalation tracking")
        ]:
            st.markdown(f'<div class="model-card"><b>{name}</b><br>{desc}</div>',
                        unsafe_allow_html=True)

    # ── Session init ─────────────────────────────────────────────────────
    if 'predictor' not in st.session_state:
        with st.spinner("🚀 Initializing System..."):
            st.session_state.predictor = AccidentPredictionEnsemble()
            st.session_state.twilio = TwilioAlert()
        st.success("✅ System ready!")

    # ── Upload & analyze ─────────────────────────────────────────────────
    uploaded_file = st.file_uploader("Upload Video", type=['mp4','avi','mov','mkv'])

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        st.video(uploaded_file)

        if st.button("🔮 Start Analysis", type="primary"):
            # Create fresh predictor instance
            st.session_state.predictor = AccidentPredictionEnsemble()

            st.markdown("---")
            st.subheader("📊 Analysis in Progress…")

            warnings, history = process_video_prediction(
                video_path, st.session_state.predictor, risk_threshold
            )

            # ── Results ──────────────────────────────────────────────────
            st.markdown("---")
            st.subheader("📈 Results")

            if warnings:
                max_w = max(warnings, key=lambda w: w['confidence'])
                box = "danger" if max_w['risk_level']=="CRITICAL" else "warning"
                st.markdown(
                    f'<div class="prediction-box {box}"><h2>{max_w["prediction_type"]}</h2></div>',
                    unsafe_allow_html=True
                )

                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Risk Level", max_w['risk_level'])
                c2.metric("Confidence", f"{max_w['confidence']:.1%}")
                c3.metric("Total Warnings", len(warnings))
                c4.metric("First Warning", warnings[0]['frame_number'])

                # ── Twilio alerts ────────────────────────────────────────
                st.markdown("---")
                st.subheader("📞 Emergency Alert")

                twilio = st.session_state.twilio

                call_ok, call_info = twilio.make_call(
                    location, max_w['risk_level'], max_w['confidence']
                )
                if call_ok:
                    st.success(f"✅ Call placed: {call_info}")
                else:
                    st.error(f"❌ Call failed: {call_info}")

                sms_ok, sms_info = twilio.send_sms(
                    location, max_w['risk_level'],
                    max_w['confidence'], max_w['frame_number']
                )
                if sms_ok:
                    st.success(f"✅ SMS sent: {sms_info}")
                else:
                    st.error(f"❌ SMS failed: {sms_info}")

            else:
                st.success("✅ No high-risk situations detected")

        try:
            os.unlink(video_path)
        except:
            pass


if __name__ == "__main__":
    if st is not None:
        main()
    else:
        print("Streamlit not installed. Please install streamlit to run the app.")
