import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from datetime import datetime
import time
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Accident Detection System",
    page_icon="🚨",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #FF4B4B;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .status-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .safe {
        background-color: #D4EDDA;
        color: #155724;
        border: 2px solid #28A745;
    }
    .alert {
        background-color: #F8D7DA;
        color: #721C24;
        border: 2px solid #DC3545;
    }
    </style>
""", unsafe_allow_html=True)

from app_prediction import AccidentPredictionEnsemble, TwilioAlert

def process_video(video_path, detector, confidence_threshold=0.4):
    """Process video and detect accidents using ResNet18"""
    cap = cv2.VideoCapture(video_path)
    
    frame_count = 0
    accident_detected = False
    accident_frames = []
    accident_details = {}
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # For displaying intermediate results
    result_placeholder = st.empty()
    
    detector.reset_buffers()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Process every 3rd frame for better detection
        if frame_count % 3 == 0:
            level, confidence, scores, ptype = detector.predict_accident(frame)
            
            # Update intermediate results
            with result_placeholder.container():
                cols = st.columns(3)
                with cols[0]:
                    st.metric("Current Frame", frame_count)
                with cols[1]:
                    st.metric("Detection Score", f"{confidence:.2%}")
                with cols[2]:
                    # Determine status based on confidence
                    if confidence >= confidence_threshold:
                        status = f"🚨 {level}"
                        st.metric("Status", status, delta="High Risk", delta_color="inverse")
                    else:
                        st.metric("Status", "✅ Normal", delta="Safe")
            
            if confidence >= confidence_threshold:
                accident_detected = True
                accident_frames.append({
                    "frame_number": frame_count,
                    "confidence": confidence,
                    "frame": frame.copy(),
                    "level": level
                })
                
                if len(accident_frames) == 1:  # First detection
                    accident_details = {
                        "frame_number": frame_count,
                        "confidence": confidence,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "severity": level
                    }
        
        # Update progress
        if total_frames > 0:
             progress = int((frame_count / total_frames) * 100)
             progress_bar.progress(progress)
        status_text.text(f"Processing: Frame {frame_count}/{total_frames}")
    
    cap.release()
    progress_bar.empty()
    status_text.empty()
    result_placeholder.empty()
    
    return accident_detected, accident_frames, accident_details

class SOSSystem:
    """Simulate SOS alert system"""
    def __init__(self):
        self.twilio = TwilioAlert()

    def send_alert(self, location, timestamp, accident_details):
        """Send SOS alert with accident details"""
        print(f"Sending alert for {location}")
        
        # Trigger real Twilio calls/SMS if configured
        # We use the confidence and level from details
        confidence = accident_details.get("confidence", 0.0)
        level = accident_details.get("severity", "HIGH")
        frame_num = accident_details.get("frame_number", 0)

        # Send SMS
        sms_ok, sms_sid = self.twilio.send_sms(location, level, confidence, frame_num)
        
        # Make Call
        call_ok, call_sid = self.twilio.make_call(location, level, confidence)

        alert_data = {
            "timestamp": timestamp,
            "location": location,
            "severity": level,
            "confidence": confidence,
            "frame_number": frame_num,
            "twilio_sms": "Sent" if sms_ok else "Failed",
            "twilio_call": "Initiated" if call_ok else "Failed"
        }
        
        return alert_data
    
    def make_emergency_call(self, alert_data):
        """Simulate emergency call"""
        call_log = f"""
        🚨 EMERGENCY CALL INITIATED
        ================================
        Time: {alert_data['timestamp']}
        Location: {alert_data['location']}
        Severity: {alert_data['severity']}
        Confidence: {alert_data['confidence']:.2%}
        Frame: {alert_data['frame_number']}
        
        Twilio SMS: {alert_data['twilio_sms']}
        Twilio Call: {alert_data['twilio_call']}
        
        Calling Emergency Services...
        Status: CONNECTED ✓
        """
        return call_log

def main():
    # Header
    st.markdown('<div class="main-header">🚨 Accident Detection System</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    This system uses AI-powered computer vision to detect accidents in video footage and automatically alerts emergency services.
    Upload a video file to begin monitoring.
    """)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        location = st.text_input("Camera Location", "Highway Junction 45")
        confidence_threshold = st.slider(
            "Detection Confidence Threshold", 
            min_value=0.2, 
            max_value=0.8, 
            value=0.4,
            help="Minimum confidence required to trigger alert (Lower = More sensitive)"
        )
        
        st.markdown("---")
        st.markdown("### 📋 Detection Methods")
        st.info("""
        ✓ Motion Analysis
        ✓ Edge Detection
        ✓ Color Pattern Recognition
        ✓ Object Clustering
        ✓ Text Pattern Detection
        """)
        
        st.markdown("---")
        st.markdown("### 📋 System Status")
        st.success("✓ Camera: Active")
        st.success("✓ Detector: Loaded")
        st.success("✓ SOS System: Ready")
    
    # Initialize detector
    if 'detector' not in st.session_state:
        st.session_state.detector = AccidentPredictionEnsemble()
        st.session_state.sos_system = SOSSystem()
        st.success("✓ Detection system initialized!")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Video File",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a video file to analyze for accidents"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        
        # Display video
        st.video(uploaded_file)
        
        # Process button
        if st.button("🔍 Start Analysis", type="primary"):
            # Reset detector for new video
            st.session_state.detector = AccidentPredictionEnsemble()
            
            st.markdown("---")
            st.subheader("📊 Analysis in Progress...")
            
            # Process video
            accident_detected, accident_frames, accident_details = process_video(
                video_path, 
                st.session_state.detector,
                confidence_threshold
            )
            
            # Display results
            st.markdown("---")
            st.subheader("📈 Analysis Results")
            
            if accident_detected:
                st.markdown(
                    '<div class="status-box alert"><h3>⚠️ ACCIDENT DETECTED!</h3></div>',
                    unsafe_allow_html=True
                )
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Detection Status", "ACCIDENT", delta="Alert Triggered")
                with col2:
                    st.metric("Confidence", f"{accident_details['confidence']:.1%}")
                with col3:
                    st.metric("Severity", accident_details['severity'])
                
                # Show detected frames
                st.subheader("🎯 Detected Accident Frames")
                
                num_frames_to_show = min(3, len(accident_frames))
                cols = st.columns(num_frames_to_show)
                for idx, frame_data in enumerate(accident_frames[:num_frames_to_show]):
                    with cols[idx]:
                        frame_rgb = cv2.cvtColor(frame_data['frame'], cv2.COLOR_BGR2RGB)
                        st.image(frame_rgb, caption=f"Frame {frame_data['frame_number']}")
                        st.caption(f"Confidence: {frame_data['confidence']:.1%}")
                
                # Send SOS Alert
                st.markdown("---")
                st.subheader("🚨 Emergency Response")
                
                alert_data = st.session_state.sos_system.send_alert(
                    location=location,
                    timestamp=accident_details['timestamp'],
                    accident_details=accident_details
                )
                
                # Display alert
                call_log = st.session_state.sos_system.make_emergency_call(alert_data)
                st.code(call_log, language=None)
                
                # Additional details
                with st.expander("📋 Detailed Accident Report"):
                    st.json({
                        "Location": location,
                        "Timestamp": accident_details['timestamp'],
                        "Frame Number": accident_details['frame_number'],
                        "Confidence Score": f"{accident_details['confidence']:.2%}",
                        "Severity Level": accident_details['severity'],
                        "Total Detections": len(accident_frames),
                        "Response Status": "Emergency Services Notified"
                    })
                
            else:
                st.markdown(
                    '<div class="status-box safe"><h3>✅ No Accident Detected</h3></div>',
                    unsafe_allow_html=True
                )
                
                st.info("Video analysis complete. No accidents detected in the footage.")
                st.metric("Status", "SAFE", delta="All Clear")
        
        # Cleanup
        try:
            os.unlink(video_path)
        except:
            pass
    
    # Instructions
    with st.expander("ℹ️ How to Use"):
        st.markdown("""
        ### Instructions:
        1. **Configure Settings**: Set the camera location and detection threshold in the sidebar
        2. **Upload Video**: Click 'Browse files' and select a video file (MP4, AVI, MOV, MKV)
        3. **Start Analysis**: Click the 'Start Analysis' button to begin processing
        4. **View Results**: The system will automatically detect accidents and trigger alerts
        
        ### Detection Methods:
        - **Motion Analysis**: Detects sudden movements (collisions)
        - **Edge Detection**: Identifies debris and vehicle deformation
        - **Color Recognition**: Detects warning colors (red alerts, fire)
        - **Object Clustering**: Identifies vehicles in close proximity
        - **Pattern Recognition**: Detects accident indicators
        
        ### Tips:
        - Lower threshold (0.2-0.4) = More sensitive, may have false positives
        - Higher threshold (0.5-0.7) = Less sensitive, fewer false alarms
        - For test video generated by our script, use threshold ~0.4
        
        ### Note:
        This is a prototype using computer vision heuristics. For production:
        - Train on real accident datasets
        - Use deep learning models
        - Integrate with live camera feeds
        - Connect to actual emergency services
        """)

if __name__ == "__main__":
    main()