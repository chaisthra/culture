import cv2
import mediapipe as mp
import json
import os
import numpy as np
import time
import math

GESTURE_FILE = 'gestures.json'

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

if os.path.exists(GESTURE_FILE):
    with open(GESTURE_FILE, 'r') as f:
        gesture_data = json.load(f)
else:
    gesture_data = {}

def capture_hand_landmarks(hand_landmarks):
    """Capture raw hand landmarks"""
    return [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]

def capture_hand_orientation(hand_landmarks):
    """Capture orientation-invariant representation of hand landmarks"""
    # Get wrist and middle finger MCP positions
    wrist = np.array([hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y, hand_landmarks.landmark[0].z])
    middle_mcp = np.array([hand_landmarks.landmark[9].x, hand_landmarks.landmark[9].y, hand_landmarks.landmark[9].z])
    
    # Create a vector from wrist to middle MCP
    vector = middle_mcp - wrist
    vector_length = np.linalg.norm(vector)
    
    # Normalize vector to get orientation
    if vector_length > 0:
        orientation = vector / vector_length
    else:
        orientation = np.array([0, 0, 0])
    
    return orientation.tolist()

def capture_hand_features(hand_landmarks):
    """Extract features that are invariant to rotation and scale"""
    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
    
    # Center the hand at origin (relative to wrist)
    wrist = landmarks[0]
    centered = landmarks - wrist
    
    # Compute distances between fingers and wrist
    fingertips = [4, 8, 12, 16, 20]  # Thumb, index, middle, ring, pinky
    distances = [np.linalg.norm(centered[tip]) for tip in fingertips]
    
    # Compute angles between fingers
    angles = []
    for i in range(len(fingertips)-1):
        v1 = centered[fingertips[i]] 
        v2 = centered[fingertips[i+1]]
        dot = np.dot(v1, v2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm > 0:
            angle = np.arccos(np.clip(dot/norm, -1.0, 1.0))
            angles.append(angle)
        else:
            angles.append(0)
    
    # Compute relative finger positions (normalized by hand size)
    hand_size = max(distances) if distances else 1
    if hand_size > 0:
        normalized_fingertips = [centered[tip]/hand_size for tip in fingertips]
        relative_positions = []
        for i in range(len(fingertips)):
            for j in range(i+1, len(fingertips)):
                dist = np.linalg.norm(normalized_fingertips[i] - normalized_fingertips[j])
                relative_positions.append(dist)
    else:
        relative_positions = [0] * 10
    
    # Return all features
    return {
        "distances": distances,
        "angles": angles,
        "relative_positions": relative_positions
    }

def capture_pose_landmarks(pose_landmarks):
    """Capture raw pose landmarks"""
    return [[lm.x, lm.y, lm.z] for lm in pose_landmarks.landmark]

def normalize_landmarks(landmarks):
    """Normalize landmarks by scaling to unit cube"""
    arr = np.array(landmarks)
    arr -= np.min(arr, axis=0)
    max_vals = np.max(arr, axis=0)
    max_vals[max_vals == 0] = 1
    arr /= max_vals
    return arr.tolist()

def compare_gesture_raw(new, saved, threshold=0.04, min_movement=0.01):
    """Compare gestures using raw landmarks"""
    if not saved or len(new) != len(saved):
        return False
    
    total_diff = sum(
        ((new[i][0] - saved[i][0])**2 + (new[i][1] - saved[i][1])**2 + (new[i][2] - saved[i][2])**2)
        for i in range(len(new))
    )
    avg_diff = total_diff / len(new)
    
    # Check if there's enough movement to consider it a gesture
    if avg_diff < min_movement:
        return False
    
    return avg_diff < threshold

def compare_gesture_features(new_features, saved_features, threshold=0.15):
    """Compare gestures using orientation-invariant features"""
    if not saved_features or not new_features:
        return 0.0
    
    # Compare distances between fingertips and wrist
    distance_diff = sum(
        (new_features["distances"][i] - saved_features["distances"][i])**2 
        for i in range(len(new_features["distances"]))
    ) / len(new_features["distances"])
    
    # Compare angles between fingers
    angle_diff = sum(
        (new_features["angles"][i] - saved_features["angles"][i])**2 
        for i in range(len(new_features["angles"]))
    ) / len(new_features["angles"]) if new_features["angles"] else 1.0
    
    # Compare relative positions
    position_diff = sum(
        (new_features["relative_positions"][i] - saved_features["relative_positions"][i])**2 
        for i in range(len(new_features["relative_positions"]))
    ) / len(new_features["relative_positions"]) if new_features["relative_positions"] else 1.0
    
    # Weighted average of all differences
    total_diff = 0.3 * distance_diff + 0.3 * angle_diff + 0.4 * position_diff
    
    # Convert to similarity score (1 is perfect match, 0 is no match)
    similarity = max(0, 1 - total_diff/threshold)
    
    return similarity

def calculate_movement(current_landmarks, previous_landmarks):
    """Calculate how much movement occurred between frames"""
    if not previous_landmarks or not current_landmarks:
        return 0
    
    if len(current_landmarks) != len(previous_landmarks):
        return 0
        
    total_diff = sum(
        ((current_landmarks[i][0] - previous_landmarks[i][0])**2 + 
         (current_landmarks[i][1] - previous_landmarks[i][1])**2 + 
         (current_landmarks[i][2] - previous_landmarks[i][2])**2)
        for i in range(len(current_landmarks))
    )
    
    return total_diff / len(current_landmarks)

def should_exit_window(window_name):
    """Check if a window should be closed"""
    return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1

def save_gesture_data():
    """Save the gesture data to the JSON file"""
    with open(GESTURE_FILE, 'w') as f:
        json.dump(gesture_data, f, indent=2)
    print(f"✅ Gesture data saved to {GESTURE_FILE}")

def delete_gesture():
    """Delete a gesture or all gestures for a country"""
    global gesture_data
    
    # List all countries with gestures
    if not gesture_data:
        print("No gestures data available.")
        return
        
    print("\nCountries with gestures:")
    for i, country in enumerate(gesture_data.keys(), 1):
        print(f"{i}. {country} ({len(gesture_data[country])} gestures)")
    
    country_choice = input("\nEnter country name or number (or 'q' to cancel): ")
    
    if country_choice.lower() == 'q':
        return
        
    # Convert number to country name if needed
    if country_choice.isdigit():
        idx = int(country_choice)
        if 1 <= idx <= len(gesture_data):
            country_choice = list(gesture_data.keys())[idx-1]
        else:
            print("Invalid country number.")
            return
    
    # Check if country exists
    if country_choice not in gesture_data:
        print(f"Country '{country_choice}' not found.")
        return
        
    country = country_choice
    
    # Ask if user wants to delete all gestures or specific one
    print(f"\nGestures for {country}:")
    for i, gesture in enumerate(gesture_data[country].keys(), 1):
        print(f"{i}. {gesture}")
    
    print("\nOptions:")
    print("Enter gesture name or number to delete specific gesture")
    print("Enter 'all' to delete all gestures for this country")
    print("Enter 'q' to cancel")
    
    gesture_choice = input("Your choice: ")
    
    if gesture_choice.lower() == 'q':
        return
        
    if gesture_choice.lower() == 'all':
        confirm = input(f"Are you sure you want to delete ALL gestures for {country}? (y/n): ")
        if confirm.lower() == 'y':
            del gesture_data[country]
            save_gesture_data()
            print(f"✅ Deleted all gestures for {country}")
        return
        
    # Convert number to gesture name if needed
    if gesture_choice.isdigit():
        idx = int(gesture_choice)
        if 1 <= idx <= len(gesture_data[country]):
            gesture_choice = list(gesture_data[country].keys())[idx-1]
        else:
            print("Invalid gesture number.")
            return
    
    # Check if gesture exists
    if gesture_choice not in gesture_data[country]:
        print(f"Gesture '{gesture_choice}' not found for {country}.")
        return
        
    # Delete the gesture
    del gesture_data[country][gesture_choice]
    
    # If no gestures left for country, remove the country entry
    if not gesture_data[country]:
        del gesture_data[country]
        print(f"✅ Deleted gesture '{gesture_choice}' for {country} (country removed as it has no gestures)")
    else:
        print(f"✅ Deleted gesture '{gesture_choice}' for {country}")
    
    # Save the updated data
    save_gesture_data()

def main():
    cap = cv2.VideoCapture(0)
    
    # Increased detection confidence to improve accuracy
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
        model_complexity=1  # 0=Lite, 1=Full
    )
    
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,  # 0=Lite, 1=Full, 2=Heavy
        enable_segmentation=False,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )

    while True:
        print("\nOptions:")
        print("1. Show gestures")
        print("2. Capture new gesture")
        print("3. Recognize gesture") 
        print("4. Delete gesture")
        print("5. Quit")
        
        choice = input("Choose an option: ")

        if choice == '1':
            if not gesture_data:
                print("No gestures saved yet.")
            else:
                print("\nSaved Gestures:")
                for country, gestures in gesture_data.items():
                    print(f"\n{country}:")
                    for gesture in gestures:
                        print(f"  - {gesture}")

        elif choice == '2':
            country = input("Enter country name: ")
            gesture = input("Enter gesture name: ")
            
            print("\nCapture gesture instructions:")
            print("- Position yourself clearly in front of camera")
            print("- Press 'c' to capture multiple variants of the gesture")
            print("- Press 's' to save all captures and finish")
            print("- Press 'q' to cancel")
            print("\nPerform the gesture and try different orientations/positions")
            
            # Initialize capture data
            gesture_captures = []
            capture_count = 0
            last_capture_time = 0
            capture_delay = 1  # seconds between captures
            showing_countdown = False
            countdown_start = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame = cv2.flip(frame, 1)  # Mirror image
                display_frame = frame.copy()
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process hands and pose
                results_hands = hands.process(rgb)
                results_pose = pose.process(rgb)
                
                # Draw skeleton on frame
                if results_hands.multi_hand_landmarks:
                    for hand_idx, hand_landmarks in enumerate(results_hands.multi_hand_landmarks):
                        mp_drawing.draw_landmarks(
                            display_frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style()
                        )
                        
                        # Add orientation indicator
                        if hand_landmarks.landmark:
                            wrist = hand_landmarks.landmark[0]
                            middle = hand_landmarks.landmark[9]
                            start_point = (int(wrist.x * frame.shape[1]), int(wrist.y * frame.shape[0]))
                            end_point = (int(middle.x * frame.shape[1]), int(middle.y * frame.shape[0]))
                            cv2.line(display_frame, start_point, end_point, (0, 255, 255), 2)
                            
                            # Add text label for the hand
                            handedness = results_hands.multi_handedness[hand_idx].classification[0].label
                            cv2.putText(display_frame, f"{handedness} Hand", 
                                  (start_point[0], start_point[1] - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                if results_pose.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        display_frame,
                        results_pose.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        mp_drawing_styles.get_default_pose_landmarks_style()
                    )
                
                # Show help text
                cv2.putText(display_frame, "Press 'c' to capture this pose", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Captures: {capture_count}", (10, 60), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Show countdown for auto-capture
                current_time = time.time()
                if showing_countdown:
                    remaining = 3 - int(current_time - countdown_start)
                    if remaining > 0:
                        cv2.putText(display_frame, f"Capturing in {remaining}...", 
                                  (frame.shape[1]//2 - 100, frame.shape[0]//2),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
                    else:
                        showing_countdown = False
                        # Capture the pose
                        capture_data = {}
                        
                        if results_hands.multi_hand_landmarks:
                            # Store both raw landmarks and orientation-invariant features
                            capture_data["hands_raw"] = [capture_hand_landmarks(hand) 
                                                      for hand in results_hands.multi_hand_landmarks]
                            capture_data["hands_features"] = [capture_hand_features(hand) 
                                                          for hand in results_hands.multi_hand_landmarks]
                        
                        if results_pose.pose_landmarks:
                            capture_data["pose_raw"] = capture_pose_landmarks(results_pose.pose_landmarks)
                        
                        if capture_data:
                            gesture_captures.append(capture_data)
                            capture_count += 1
                            last_capture_time = current_time
                            # Add a visual feedback of the capture
                            cv2.putText(display_frame, "Captured!", 
                                      (frame.shape[1]//2 - 70, frame.shape[0]//2),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                
                cv2.imshow("Capture Gesture", display_frame)
                key = cv2.waitKey(5) & 0xFF
                
                # Process key inputs
                if key == ord('c') and not showing_countdown:
                    # Start countdown
                    showing_countdown = True
                    countdown_start = current_time
                
                elif key == ord('s'):
                    # Save the gesture data if we have captures
                    if gesture_captures:
                        # Combine all captures into a single gesture data
                        landmarks = {
                            "raw_data": gesture_captures,
                            "capture_count": capture_count
                        }
                        
                        # Store the gesture under the country
                        gesture_data.setdefault(country, {})[gesture] = landmarks
                        save_gesture_data()
                        print(f"[+] Saved {capture_count} variants of '{gesture}' for {country}")
                        break
                    else:
                        print("No gestures captured. Try again.")
                
                elif key == ord('q') or should_exit_window("Capture Gesture"):
                    break

            cv2.destroyWindow("Capture Gesture")

        elif choice == '3':
            country = input("Enter country: ")
            print("Show gesture to camera. Press 'q' to quit recognition.")
            
            # Check if country exists in the gesture data
            if country not in gesture_data:
                print(f"No gestures found for country '{country}'")
                continue
                
            # Variables for tracking
            last_recognized_gesture = None
            gesture_found = False
            recognition_cooldown = 0
            previous_hand_landmarks = None
            previous_pose_landmarks = None
            stable_frames = 0  # Count frames where the gesture is stable
            
            # Required number of stable frames to confirm a gesture
            REQUIRED_STABLE_FRAMES = 3
            
            # Movement thresholds
            MIN_MOVEMENT_THRESHOLD = 0.005  # Minimum movement to register intentional gesture
            STABILITY_THRESHOLD = 0.005     # Maximum movement to consider pose "stable"

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                results_hands = hands.process(rgb)
                results_pose = pose.process(rgb)

                current = {}
                current_features = {}
                
                # Process hands
                if results_hands.multi_hand_landmarks:
                    hands_landmarks_raw = [capture_hand_landmarks(hand) 
                                       for hand in results_hands.multi_hand_landmarks]
                    hands_features = [capture_hand_features(hand) 
                                   for hand in results_hands.multi_hand_landmarks]
                    
                    if all(len(hand) >= 10 for hand in hands_landmarks_raw):
                        current["hands_raw"] = hands_landmarks_raw
                        current_features["hands_features"] = hands_features
                        
                        # Calculate hand movement
                        if previous_hand_landmarks and len(previous_hand_landmarks) == len(hands_landmarks_raw):
                            hand_movement = sum(calculate_movement(hands_landmarks_raw[i], previous_hand_landmarks[i]) 
                                           for i in range(len(hands_landmarks_raw))) / len(hands_landmarks_raw)
                            
                            # Display movement amount for debugging
                            cv2.putText(frame, f"Hand Movement: {hand_movement:.4f}", (10, 30), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                            
                            # Check if stable
                            if hand_movement < STABILITY_THRESHOLD:
                                cv2.putText(frame, "Hand Stable", (10, 50), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                            else:
                                # Reset stability counter if movement is detected
                                stable_frames = 0
                                
                        previous_hand_landmarks = hands_landmarks_raw
                
                # Process pose
                if results_pose.pose_landmarks:
                    pose_landmarks_raw = capture_pose_landmarks(results_pose.pose_landmarks)
                    if len(pose_landmarks_raw) >= 20:
                        current["pose_raw"] = pose_landmarks_raw
                        
                        # Calculate pose movement
                        if previous_pose_landmarks:
                            pose_movement = calculate_movement(pose_landmarks_raw, previous_pose_landmarks)
                            
                            # Display movement amount for debugging
                            cv2.putText(frame, f"Pose Movement: {pose_movement:.4f}", (10, 70), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                            
                            # Check if stable
                            if pose_movement < STABILITY_THRESHOLD:
                                cv2.putText(frame, "Pose Stable", (10, 90), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                            else:
                                # Reset stability counter if movement is detected
                                stable_frames = 0
                                
                        previous_pose_landmarks = pose_landmarks_raw

                # Draw landmarks
                if results_hands.multi_hand_landmarks:
                    for hand_idx, hand in enumerate(results_hands.multi_hand_landmarks):
                        mp_drawing.draw_landmarks(
                            frame, 
                            hand, 
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style()
                        )
                        
                        # Add handedness label
                        if results_hands.multi_handedness:
                            handedness = results_hands.multi_handedness[hand_idx].classification[0].label
                            wrist_pos = hand.landmark[0]
                            cv2.putText(frame, handedness, 
                                      (int(wrist_pos.x * frame.shape[1]), int(wrist_pos.y * frame.shape[0]) - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                if results_pose.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, 
                        results_pose.pose_landmarks, 
                        mp_pose.POSE_CONNECTIONS,
                        mp_drawing_styles.get_default_pose_landmarks_style()
                    )

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
                                    hands_match = True
                                    hand_confidence = 0
                                    
                                    # Compare each hand using the features
                                    for i in range(len(current_features["hands_features"])):
                                        similarity = compare_gesture_features(
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
                                if compare_gesture_raw(current["pose_raw"], variant["pose_raw"]):
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
                                recognition_cooldown = 15  # Wait 15 frames before recognizing again
                                
                            # Display the recognized gesture on screen
                            cv2.putText(frame, f"Gesture: {found_gesture}", (10, 120), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            cv2.putText(frame, f"Confidence: {max_confidence:.2f}", (10, 150), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    else:
                        # Reset stability if no gesture is found
                        stable_frames = 0
                        if gesture_found:
                            cv2.putText(frame, "No gesture detected", (10, 120), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            gesture_found = False
                
                # Update cooldown
                if recognition_cooldown > 0:
                    recognition_cooldown -= 1
                
                # Display stability info
                cv2.putText(frame, f"Stability: {stable_frames}/{REQUIRED_STABLE_FRAMES}", (10, 180), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                cv2.imshow("Recognize Gesture", frame)
                key = cv2.waitKey(5) & 0xFF
                if key == ord('q') or should_exit_window("Recognize Gesture"):
                    break

            cv2.destroyWindow("Recognize Gesture")

        elif choice == '4':
            delete_gesture()

        elif choice == '5':
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()