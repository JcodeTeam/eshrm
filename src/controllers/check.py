import cv2
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

def detect_blink(video_path, blink_threshold=0.22, blink_frames=2):
    cap = cv2.VideoCapture(video_path)
    blink_count = 0
    consecutive_blink_frames = 0
    
    with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True) as face_mesh:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                
                left_eye = [landmarks[i] for i in [33, 160, 158, 133, 153, 144]]
                right_eye = [landmarks[i] for i in [263, 387, 385, 362, 380, 373]]
                
                def eye_aspect_ratio(eye):
                    p1 = np.array([eye[1].x, eye[1].y])
                    p2 = np.array([eye[5].x, eye[5].y])
                    p3 = np.array([eye[2].x, eye[2].y])
                    p4 = np.array([eye[4].x, eye[4].y])
                    p0 = np.array([eye[0].x, eye[0].y])
                    p5 = np.array([eye[3].x, eye[3].y])
                    return (np.linalg.norm(p1 - p2) + np.linalg.norm(p3 - p4)) / (2.0 * np.linalg.norm(p0 - p5))
                
                left_ear = eye_aspect_ratio(left_eye)
                right_ear = eye_aspect_ratio(right_eye)
                ear = (left_ear + right_ear) / 2.0
                
                if ear < blink_threshold:
                    consecutive_blink_frames += 1
                else:
                    if consecutive_blink_frames >= blink_frames:
                        blink_count += 1
                    consecutive_blink_frames = 0
        
        cap.release()
    
    return blink_count > 0
