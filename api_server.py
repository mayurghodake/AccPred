from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import tempfile

app = Flask(__name__)
CORS(app)  # Enable CORS for React

from app_prediction import AccidentPredictionEnsemble, process_video_prediction_flask, TwilioAlert

# IMPORTANT: Don't initialize predictor globally - create fresh for each request
# This was causing state contamination issues
# predictor = AccidentPredictionEnsemble()  # ❌ DON'T DO THIS

# Initialize Twilio once (stateless helper)
twilio_alert = TwilioAlert()


@app.route('/api/analyze', methods=['POST'])
def analyze_video():
    """Endpoint to analyze uploaded video"""
    print("\n" + "="*70)
    print("📹 New Analysis Request Received")
    print("="*70)
    
    try:
        # Validate request
        if 'video' not in request.files:
            print("❌ No video in request.files")
            return jsonify({'error': 'No video uploaded'}), 400

        video_file = request.files['video']
        print(f"📁 Video filename: {video_file.filename}")
        
        if video_file.filename == '':
            print("❌ Video filename is empty")
            return jsonify({'error': 'No video selected'}), 400

        # Get configuration from request (optional)
        risk_threshold = float(request.form.get('risk_threshold', 0.55))  # Updated default
        location = request.form.get('location', 'Highway Junction 45')
        
        print(f"⚙️ Config: threshold={risk_threshold}, location={location}")

        # Save video temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            video_file.save(tmp.name)
            video_path = tmp.name
            print(f"💾 Saved video to: {video_path}")

        try:
            # CRITICAL: Create FRESH predictor for each request
            # This prevents state contamination between requests
            print("🤖 Creating fresh predictor instance...")
            predictor = AccidentPredictionEnsemble()
            
            # Run prediction
            print("🔍 Starting video processing...")
            warnings = process_video_prediction_flask(
                video_path, predictor, risk_threshold
            )
            print(f"✅ Processing complete: {len(warnings)} warnings detected")

            if warnings:
                # Get highest confidence warning
                max_warning = max(warnings, key=lambda w: w['confidence'])
                
                print(f"⚠️ RISK DETECTED: {max_warning['risk_level']} at {max_warning['confidence']:.1%}")
                
                # Send Twilio alerts for HIGH/CRITICAL risks
                alerts = {'call': {}, 'sms': {}}
                
                if max_warning['risk_level'] in ['HIGH', 'CRITICAL']:
                    print("📞 Triggering Twilio alerts...")
                    
                    # Voice call
                    call_success, call_info = twilio_alert.make_call(
                        location, 
                        max_warning['risk_level'], 
                        max_warning['confidence']
                    )
                    alerts['call'] = {
                        'success': call_success,
                        'sid': call_info if call_success else None,
                        'error': call_info if not call_success else None
                    }
                    
                    # SMS
                    sms_success, sms_info = twilio_alert.send_sms(
                        location,
                        max_warning['risk_level'],
                        max_warning['confidence'],
                        max_warning['frame_number']
                    )
                    alerts['sms'] = {
                        'success': sms_success,
                        'sid': sms_info if sms_success else None,
                        'error': sms_info if not sms_success else None
                    }
                    
                    if call_success and sms_success:
                        print("✅ Twilio alerts sent successfully")
                    else:
                        print("⚠️ Some Twilio alerts failed")
                else:
                    print(f"ℹ️ Risk level {max_warning['risk_level']} - no alerts sent")
                
                result = {
                    'detected': True,
                    'riskLevel': max_warning['risk_level'],
                    'confidence': max_warning['confidence'],
                    'frame': max_warning['frame_number'],
                    'timestamp': max_warning['timestamp'],
                    'modelScores': max_warning['model_scores'],
                    'totalWarnings': len(warnings),
                    'alerts': alerts,
                    'location': location
                }
            else:
                print("✅ No high-risk situations detected")
                result = {
                    'detected': False,
                    'message': 'No high-risk situations detected in video'
                }

            print(f"📤 Returning result: detected={result.get('detected', False)}")
            return jsonify(result)

        except Exception as e:
            print(f"❌ Error during video processing: {str(e)}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Video processing failed: {str(e)}'}), 500

        finally:
            # Cleanup
            try:
                os.unlink(video_path)
                print(f"🗑️ Cleaned up: {video_path}")
            except Exception as e:
                print(f"⚠️ Cleanup warning: {e}")

    except Exception as e:
        print(f"❌ Error in analyze_video: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@app.route('/api/status', methods=['GET'])
def system_status():
    """Get system status"""
    print("ℹ️ Status check requested")
    
    # Check Twilio status
    twilio_status = 'connected' if not twilio_alert.missing else 'not_configured'
    twilio_verified = twilio_alert.emergency_verified if not twilio_alert.missing else False
    
    status = {
        'camera': 'active',
        'models': 'loaded',
        'twilio': twilio_status,
        'twilioVerified': twilio_verified,
        'version': 'fixed_v2'  # Indicate using fixed version
    }
    
    print(f"📊 Status: {status}")
    return jsonify(status)


@app.route('/api/test', methods=['GET'])
def test_endpoint():
    """Simple test endpoint"""
    return jsonify({
        'status': 'ok',
        'message': 'Flask backend is running',
        'version': 'fixed_v2'
    })


@app.route('/api/config', methods=['GET', 'POST'])
def config_endpoint():
    """Get or update configuration"""
    if request.method == 'GET':
        return jsonify({
            'defaultThreshold': 0.55,  # Updated default
            'thresholds': {
                'CRITICAL': 0.70,  # Updated
                'HIGH': 0.55,      # Updated
                'MODERATE': 0.40   # Updated
            },
            'processingInterval': 3,  # Process every 3rd frame
            'temporalConsistency': True,
            'bufferReset': True  # Confirm we're resetting buffers
        })
    
    # POST: Update config (future feature)
    return jsonify({'message': 'Configuration update not implemented yet'}), 501


@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 Starting Accident Prediction Flask Backend")
    print("="*70)
    print("📍 Server: http://localhost:5000")
    print("🔧 Mode: Debug")
    print("📦 Version: Fixed v2 (Consistent Detection)")
    print("="*70 + "\n")
    
    # Check Twilio status on startup
    if twilio_alert.missing:
        print("⚠️ WARNING: Twilio credentials not configured")
        print("   Alerts will NOT be sent. Check your .env file")
    elif not twilio_alert.emergency_verified:
        print("⚠️ WARNING: Emergency number not verified in Twilio")
        print("   Alerts will FAIL until number is verified")
    else:
        print("✅ Twilio configured and verified")
    
    print("\n" + "-"*70)
    print("Ready to accept requests!")
    print("-"*70 + "\n")
    
    app.run(debug=True, port=5000, host='0.0.0.0')