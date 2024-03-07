import cv2 as cv
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

cap = cv.VideoCapture(0)
with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame")
            continue

        image = cv.cvtColor(image, cv.COLOR_BGR2RGB) if mp_pose.get_keypoint_tracking_enabled() else image

        results = pose.process(image)

        image = cv.cvtColor(image, cv.COLOR_RGB2BGR) if mp_pose.get_keypoint_tracking_enabled() else image

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                connection_drawing_spec=mp_drawing_styles.get_default_pose_connections_style()
            )

        cv.imshow('MediaPipe Pose Estimation Program', cv.flip(image, 1))
        if cv.waitKey(5) & 0xFF == ord('d'):
            break

cap.release()