from flask import Flask, request, jsonify
import json
import os
import cv2
import base64
import numpy as np
import threading
import time
from io import BytesIO
from PIL import Image

# Import your gesture recognition functionality
# This is structured to use your code without modifying it
import sys
import importlib.util

# Define the path to gesture_recognition.py
GESTURE_RECOGNITION_PATH = 'gesture_recognition.py'

# Import the module dynamically
spec = importlib.util.spec_from_file_location("gesture_recognition", GESTURE_RECOGNITION_PATH)
gesture_recognition = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gesture_recognition)

app = Flask(__name__)

# Global variables
recognition_thread = None
stop_recognition = threading.Event()
current_recognition_country = None
last_recognized_gesture = None
recognition_active = False

# Constants from your original code
GESTURE_FILE = 'gestures.json'

# Load gesture data
if os.path.exists(GESTURE_FILE):
    with open(GESTURE_FILE, 'r') as f:
        gesture_data = json.load(f)
else:
    gesture_data = {}

def base64_to_image(base64_str):
    """Convert base64 string to OpenCV image"""
    img_data = base64.b64decode(base64_str)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def image_to_base64(image):
    """Convert OpenCV image to base64 string"""
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

def recognition_worker(country):
    """Background worker for continuous gesture recognition"""
    global last_recognized_gesture, recognition_active
    
    # Initialize MediaPipe hands and pose
    mp_hands = gesture_recognition.mp_hands
    mp_pose = gesture_recognition.mp_pose
    mp_drawing = gesture_recognition.mp_drawing
    
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
        model_complexity=1
    )
    
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    
    # Variables for tracking
    previous_hand_landmarks = None
    previous_pose_landmarks = None
    stable_frames = 0
    gesture_found = False
    recognition_cooldown = 0
    
    # Required number of stable frames to confirm a gesture
    REQUIRED_STABLE_FRAMES = 3
    
    # Movement thresholds
    STABILITY_THRESHOLD = 0.005
    
    # Use webcam
    cap = cv2.VideoCapture(0)
    recognition_active = True
    
    try:
        while not stop_recognition.is_set():
            ret, frame = cap.read()
            if not ret:
                continue
                
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results_hands = hands.process(rgb)
            results_pose = pose.process(rgb)

            current = {}
            current_features = {}
            
            # Process hands
            if results_hands.multi_hand_landmarks:
                hands_landmarks_raw = [gesture_recognition.capture_hand_landmarks(hand) 
                                   for hand in results_hands.multi_hand_landmarks]
                hands_features = [gesture_recognition.capture_hand_features(hand) 
                               for hand in results_hands.multi_hand_landmarks]
                
                if all(len(hand) >= 10 for hand in hands_landmarks_raw):
                    current["hands_raw"] = hands_landmarks_raw
                    current_features["hands_features"] = hands_features
                    
                    # Calculate hand movement
                    if previous_hand_landmarks and len(previous_hand_landmarks) == len(hands_landmarks_raw):
                        hand_movement = sum(gesture_recognition.calculate_movement(hands_landmarks_raw[i], previous_hand_landmarks[i]) 
                                       for i in range(len(hands_landmarks_raw))) / len(hands_landmarks_raw)
                        
                        # Check if stable
                        if hand_movement < STABILITY_THRESHOLD:
                            pass  # Hand is stable
                        else:
                            # Reset stability counter if movement is detected
                            stable_frames = 0
                            
                    previous_hand_landmarks = hands_landmarks_raw
            
            # Process pose
            if results_pose.pose_landmarks:
                pose_landmarks_raw = gesture_recognition.capture_pose_landmarks(results_pose.pose_landmarks)
                if len(pose_landmarks_raw) >= 20:
                    current["pose_raw"] = pose_landmarks_raw
                    
                    # Calculate pose movement
                    if previous_pose_landmarks:
                        pose_movement = gesture_recognition.calculate_movement(pose_landmarks_raw, previous_pose_landmarks)
                        
                        # Check if stable
                        if pose_movement < STABILITY_THRESHOLD:
                            pass  # Pose is stable
                        else:
                            # Reset stability counter if movement is detected
                            stable_frames = 0
                            
                    previous_pose_landmarks = pose_landmarks_raw

            # Process recognition logic only if we have enough data
            if current and recognition_cooldown == 0:
                found_gesture = None
                max_confidence = 0
                
                # Check all gestures for this country
                for gesture_name, saved_data in gesture_data.get(country, {}).items():
                    # For each saved variant of the gesture
                    for variant_idx, variant in enumerate(saved_data.get("raw_data", [])):
                        confidence = 0
                        matches = 0
                        
                        # Check hands
                        if "hands_features" in current_features and "hands_features" in variant:
                            if len(current_features["hands_features"]) == len(variant["hands_features"]):
                                hand_confidence = 0
                                
                                # Compare each hand using the features
                                for i in range(len(current_features["hands_features"])):
                                    similarity = gesture_recognition.compare_gesture_features(
                                        current_features["hands_features"][i], 
                                        variant["hands_features"][i]
                                    )
                                    hand_confidence += similarity
                                    
                                # Average hand confidence
                                hand_confidence /= len(current_features["hands_features"])
                                
                                if hand_confidence > 0.6:  # Only count as match if confidence is high enough
                                    matches += 1
                                    confidence += hand_confidence * 0.7  # Hands are more important
                        
                        # Check pose (using raw comparison)
                        if "pose_raw" in current and "pose_raw" in variant:
                            if gesture_recognition.compare_gesture_raw(current["pose_raw"], variant["pose_raw"]):
                                matches += 1
                                confidence += 0.3  # Pose is less important
                        
                        # Update if this is the best match so far
                        if matches > 0 and confidence > max_confidence:
                            found_gesture = gesture_name
                            max_confidence = confidence
                
                # If we found a potential gesture match
                if found_gesture:
                    # Check if it's the same as last time for stability
                    if found_gesture == last_recognized_gesture:
                        stable_frames += 1
                    else:
                        stable_frames = 0
                        
                    last_recognized_gesture = found_gesture
                    
                    # Only report gesture after it's been stable for some frames
                    if stable_frames >= REQUIRED_STABLE_FRAMES:
                        if not gesture_found or found_gesture != gesture_found:
                            print(f"✅ Recognized Gesture: {found_gesture} (Confidence: {max_confidence:.2f})")
                            gesture_found = found_gesture
                            recognition_cooldown = 15
                else:
                    # Reset stability if no gesture is found
                    stable_frames = 0
                    last_recognized_gesture = None  # Reset last recognized gesture
            
            # Update cooldown
            if recognition_cooldown > 0:
                recognition_cooldown -= 1
                
            # Sleep to reduce CPU usage
            time.sleep(0.03)
    
    finally:
        recognition_active = False
        cap.release()

@app.route('/api/gestures', methods=['GET'])
def get_gestures():
    """Get all saved gestures"""
    return jsonify(gesture_data)

@app.route('/api/gestures/<country>', methods=['GET'])
def get_country_gestures(country):
    """Get gestures for a specific country"""
    if country in gesture_data:
        return jsonify(gesture_data[country])
    else:
        return jsonify({"error": f"No gestures found for country '{country}'"}), 404

@app.route('/api/gestures/<country>/<gesture>', methods=['GET'])
def get_specific_gesture(country, gesture):
    """Get a specific gesture"""
    if country in gesture_data and gesture in gesture_data[country]:
        return jsonify(gesture_data[country][gesture])
    else:
        return jsonify({"error": f"Gesture '{gesture}' not found for country '{country}'"}), 404

@app.route('/api/gestures/<country>/<gesture>', methods=['DELETE'])
def delete_gesture(country, gesture):
    """Delete a specific gesture"""
    global gesture_data
    
    if country in gesture_data and gesture in gesture_data[country]:
        del gesture_data[country][gesture]
        
        # If no gestures left for country, remove the country entry
        if not gesture_data[country]:
            del gesture_data[country]
            
        # Save the updated data
        with open(GESTURE_FILE, 'w') as f:
            json.dump(gesture_data, f, indent=2)
            
        return jsonify({"message": f"Deleted gesture '{gesture}' for country '{country}'"})
    else:
        return jsonify({"error": f"Gesture '{gesture}' not found for country '{country}'"}), 404

@app.route('/api/gestures/<country>', methods=['DELETE'])
def delete_country_gestures(country):
    """Delete all gestures for a country"""
    global gesture_data
    
    if country in gesture_data:
        del gesture_data[country]
        
        # Save the updated data
        with open(GESTURE_FILE, 'w') as f:
            json.dump(gesture_data, f, indent=2)
            
        return jsonify({"message": f"Deleted all gestures for country '{country}'"})
    else:
        return jsonify({"error": f"No gestures found for country '{country}'"}), 404

@app.route('/api/gestures/<country>/<gesture>', methods=['POST'])
def add_gesture(country, gesture):
    """Add a new gesture from video frames"""
    global gesture_data
    
    if not request.json or 'frames' not in request.json:
        return jsonify({"error": "No frames provided"}), 400
        
    frames = request.json['frames']
    if not frames:
        return jsonify({"error": "Empty frames list"}), 400
    
    # Process the frames to extract gestures
    gesture_captures = []
    capture_count = 0
    
    # Initialize MediaPipe hands and pose
    mp_hands = gesture_recognition.mp_hands
    mp_pose = gesture_recognition.mp_pose
    
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
        model_complexity=1
    )
    
    pose = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    
    for frame_base64 in frames:
        # Convert base64 to image
        try:
            frame = base64_to_image(frame_base64)
            if frame is None:
                continue
                
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process hands and pose
            results_hands = hands.process(rgb)
            results_pose = pose.process(rgb)
            
            capture_data = {}
            
            if results_hands.multi_hand_landmarks:
                # Store both raw landmarks and orientation-invariant features
                capture_data["hands_raw"] = [gesture_recognition.capture_hand_landmarks(hand) 
                                         for hand in results_hands.multi_hand_landmarks]
                capture_data["hands_features"] = [gesture_recognition.capture_hand_features(hand) 
                                              for hand in results_hands.multi_hand_landmarks]
            
            if results_pose.pose_landmarks:
                capture_data["pose_raw"] = gesture_recognition.capture_pose_landmarks(results_pose.pose_landmarks)
            
            if capture_data:
                gesture_captures.append(capture_data)
                capture_count += 1
                
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
    
    # Save the gesture data if we have captures
    if gesture_captures:
        # Combine all captures into a single gesture data
        landmarks = {
            "raw_data": gesture_captures,
            "capture_count": capture_count
        }
        
        # Store the gesture under the country
        gesture_data.setdefault(country, {})[gesture] = landmarks
        
        # Save to file
        with open(GESTURE_FILE, 'w') as f:
            json.dump(gesture_data, f, indent=2)
            
        return jsonify({
            "message": f"Saved {capture_count} variants of '{gesture}' for {country}",
            "capture_count": capture_count
        })
    else:
        return jsonify({"error": "No valid gestures captured"}), 400

@app.route('/api/recognize/start/<country>', methods=['POST'])
def start_recognition(country):
    """Start gesture recognition for a country"""
    global recognition_thread, stop_recognition, current_recognition_country, recognition_active
    
    # Check if country exists
    if country not in gesture_data:
        return jsonify({"error": f"No gestures found for country '{country}'"}), 404
    
    # If already running, stop it first
    if recognition_thread and recognition_thread.is_alive():
        stop_recognition.set()
        recognition_thread.join(timeout=2.0)
        stop_recognition.clear()
    
    # Start new recognition thread
    current_recognition_country = country
    recognition_thread = threading.Thread(target=recognition_worker, args=(country,))
    recognition_thread.daemon = True
    recognition_thread.start()
    
    return jsonify({"message": f"Started recognition for country '{country}'"})

@app.route('/api/recognize/stop', methods=['POST'])
def stop_recognition_endpoint():
    """Stop gesture recognition"""
    global recognition_thread, stop_recognition, current_recognition_country, recognition_active
    
    if recognition_thread and recognition_thread.is_alive():
        stop_recognition.set()
        recognition_thread.join(timeout=2.0)
        stop_recognition.clear()
        current_recognition_country = None
        return jsonify({"message": "Recognition stopped"})
    else:
        return jsonify({"message": "No recognition was running"})

@app.route('/api/recognize/status', methods=['GET'])
def recognition_status():
    """Get current recognition status"""
    global recognition_thread, current_recognition_country, last_recognized_gesture, recognition_active
    
    is_running = recognition_thread is not None and recognition_thread.is_alive() and recognition_active
    
    return jsonify({
        "running": is_running,
        "country": current_recognition_country,
        "last_recognized_gesture": last_recognized_gesture
    })

@app.route('/api/recognize/last', methods=['GET'])
def get_last_gesture():
    """Get the last recognized gesture"""
    global last_recognized_gesture
    
    if last_recognized_gesture:
        return jsonify({"gesture": last_recognized_gesture})
    else:
        return jsonify({"gesture": None})
        
@app.route('/api/capture/frame', methods=['POST'])
def capture_frame():
    """Capture a single frame from webcam and analyze it"""
    if not request.json or 'frame' not in request.json:
        return jsonify({"error": "No frame provided"}), 400
        
    frame_base64 = request.json['frame']
    country = request.json.get('country')
    
    if not country or country not in gesture_data:
        return jsonify({"error": "Invalid or missing country parameter"}), 400
    
    try:
        # Convert base64 to image
        frame = base64_to_image(frame_base64)
        if frame is None:
            return jsonify({"error": "Invalid frame data"}), 400
            
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Initialize MediaPipe hands and pose
        mp_hands = gesture_recognition.mp_hands
        mp_pose = gesture_recognition.mp_pose
        
        hands = mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        
        pose = mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Process hands and pose
        results_hands = hands.process(rgb)
        results_pose = pose.process(rgb)
        
        current = {}
        current_features = {}
        
        # Process hands
        if results_hands.multi_hand_landmarks:
            hands_landmarks_raw = [gesture_recognition.capture_hand_landmarks(hand) 
                               for hand in results_hands.multi_hand_landmarks]
            hands_features = [gesture_recognition.capture_hand_features(hand) 
                           for hand in results_hands.multi_hand_landmarks]
            
            if all(len(hand) >= 10 for hand in hands_landmarks_raw):
                current["hands_raw"] = hands_landmarks_raw
                current_features["hands_features"] = hands_features
        
        # Process pose
        if results_pose.pose_landmarks:
            pose_landmarks_raw = gesture_recognition.capture_pose_landmarks(results_pose.pose_landmarks)
            if len(pose_landmarks_raw) >= 20:
                current["pose_raw"] = pose_landmarks_raw
        
        # Process recognition
        found_gesture = None
        max_confidence = 0
        
        # Check all gestures for this country
        for gesture_name, saved_data in gesture_data.get(country, {}).items():
            # For each saved variant of the gesture
            for variant_idx, variant in enumerate(saved_data.get("raw_data", [])):
                confidence = 0
                matches = 0
                
                # Check hands
                if "hands_features" in current_features and "hands_features" in variant:
                    if len(current_features["hands_features"]) == len(variant["hands_features"]):
                        hand_confidence = 0
                        
                        # Compare each hand using the features
                        for i in range(len(current_features["hands_features"])):
                            similarity = gesture_recognition.compare_gesture_features(
                                current_features["hands_features"][i], 
                                variant["hands_features"][i]
                            )
                            hand_confidence += similarity
                            
                        # Average hand confidence
                        hand_confidence /= len(current_features["hands_features"])
                        
                        if hand_confidence > 0.6:  # Only count as match if confidence is high enough
                            matches += 1
                            confidence += hand_confidence * 0.7  # Hands are more important
                
                # Check pose (using raw comparison)
                if "pose_raw" in current and "pose_raw" in variant:
                    if gesture_recognition.compare_gesture_raw(current["pose_raw"], variant["pose_raw"]):
                        matches += 1
                        confidence += 0.3  # Pose is less important
                
                # Update if this is the best match so far
                if matches > 0 and confidence > max_confidence:
                    found_gesture = gesture_name
                    max_confidence = confidence
        
        return jsonify({
            "gesture": found_gesture,
            "confidence": max_confidence if found_gesture else 0,
            "has_hands": bool(results_hands.multi_hand_landmarks),
            "has_pose": bool(results_pose.pose_landmarks)
        })
        
    except Exception as e:
        return jsonify({"error": f"Error processing frame: {str(e)}"}), 500

@app.route('/api/webcam', methods=['GET'])
def get_webcam_frame():
    """Get current webcam frame with landmarks drawn"""
    # Capture from webcam
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return jsonify({"error": "Failed to capture frame from webcam"}), 500
    
    frame = cv2.flip(frame, 1)  # Mirror image
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Initialize MediaPipe hands and pose
    mp_hands = gesture_recognition.mp_hands
    mp_pose = gesture_recognition.mp_pose
    mp_drawing = gesture_recognition.mp_drawing
    mp_drawing_styles = gesture_recognition.mp_drawing_styles
    
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
        model_complexity=1
    )
    
    pose = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    
    # Process hands and pose
    results_hands = hands.process(rgb)
    results_pose = pose.process(rgb)
    
    # Draw landmarks on frame
    if results_hands.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(results_hands.multi_hand_landmarks):
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
    
    if results_pose.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results_pose.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing_styles.get_default_pose_landmarks_style()
        )
    
    # Convert to base64
    frame_base64 = image_to_base64(frame)
    
    return jsonify({
        "frame": frame_base64,
        "has_hands": bool(results_hands.multi_hand_landmarks),
        "has_pose": bool(results_pose.pose_landmarks)
    })

if __name__ == '__main__':
    app.run(debug=True)