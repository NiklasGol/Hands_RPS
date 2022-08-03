import cv2
import numpy as np
import mediapipe as mp
import json

# path (change path name for new sample)
# data_path = '/home/niklas/Ablage/Own_projects/Hand_RPS/'
data_path = '/home/niklas/Ablage/Own_projects/Hand_RPS/2/'

# select name according to upcomig recording
# data_name = 'rock_pose_rh.json'
# data_name = 'rock_pose_lh.json'
# data_name = 'paper_pose_rh.json'
# data_name = 'paper_pose_lh.json'
# data_name = 'scissors_pose_rh.json'
# data_name = 'scissors_pose_lh.json'

# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
save_hand_coordinates = []

run = True
while run:
    # Read each frame from the webcam
    _, frame = cap.read()
    x , y, c = frame.shape
    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Get hand landmark prediction
    result = hands.process(framergb)
    className = ''
    # post process the result
    if result.multi_hand_landmarks:
      landmarks = []
      # save_hand_coordinates.append()
      for handslms in result.multi_hand_landmarks:
          for lm in handslms.landmark:
              landmarks.append([lm.x, lm.y, lm.z])
          # Drawing landmarks on frames
          save_hand_coordinates.append(landmarks)
          mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
    # Show the final output
    cv2.imshow("Output", frame)
    if cv2.waitKey(1) == ord('q'):
        print("End")
        break
    if len(save_hand_coordinates) == 1000:
        run = False
    # time.sleep(1)

# release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()

with open(data_path+data_name, 'w') as f:
    json.dump(save_hand_coordinates, f, indent=2)
