import cv2
import numpy as np
import mediapipe as mp
import time
import pickle
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
import torch
import random

from pose_network import Network

def recognize_pose(network, points, class_map):
    # use network to recognize the hand pose
    dist_matrix = cdist(points, points)
    dist_non_zero = dist_matrix[dist_matrix != 0].flatten()
    dist_normlized = dist_non_zero/np.linalg.norm(dist_non_zero)
    dist_normlized = np.expand_dims(dist_normlized, axis=0)
    input = pca_reload.transform(dist_normlized)
    input = torch.from_numpy(input).type(torch.FloatTensor)
    output = model(input.to(device))
    pose_index = np.argmax(output.detach().numpy())
    pose = class_map[pose_index]
    return pose, pose_index

def find_majority(k):
    myMap = {}
    maximum = ( '', 0 ) # (occurring element, occurrences)
    for n in k:
        if n in myMap:
            myMap[n] += 1
        else:
            myMap[n] = 1
        # Keep track of maximum on the go
        if myMap[n] > maximum[1]:
            maximum = (n,myMap[n])
    return maximum[0]

def rps_outcome(pose_computer, pose_player):
    """ 0: Rock, 1: Paper, 2:Scissor to enter the win table, 0: draw,
        1: computer wins, 2: player wins. """
    win_table = np.asarray([[0, 2, 1],[1, 0, 2],[2, 1, 0]])
    outcomes = {0: "Drawn", 1: "Computer wins", 2: "You win"}
    outcome = outcomes[win_table[pose_computer, pose_player]]
    return outcome

if __name__ == '__main__':
    save_path = '/home/niklas/Ablage/Own_projects/Hand_RPS/trained_model/'

    # network output to pose
    class_map = {0: 'Rock', 1: 'Paper', 2: 'Scissors'}

    # initialize mediapipe
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    mpDraw = mp.solutions.drawing_utils

    # load model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Network()
    model.load_state_dict(torch.load(save_path+'trained_model.pth'))
    model.eval()

    # load pca used from training
    pca_reload = pickle.load(open(save_path+'pca_params.pkl','rb'))

    # select webcam
    cap = cv2.VideoCapture(0)
    started = False
    round_done = False
    pose_record = []
    pose_num_record = []

    while True:
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
          for handslms in result.multi_hand_landmarks:
              for lm in handslms.landmark:
                  # save points
                  landmarks.append([lm.x, lm.y, lm.z])
              points = np.asarray(landmarks)
              pose, pose_num = recognize_pose(model, points, class_map)
              # Drawing landmarks on frames
              mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
        else:
            pose = 'No hand detected'
            pose_num = 4
        if started:
            delta_time = time.time() - start_time
            if delta_time > 4 and choose:
                computer_pose_num = random.choice([0,1,2])
                choose = False
            if delta_time < 2:
                text = "Get ready!"
                text_size = 1
            elif delta_time < 3:
                text = "ROCK!"
                text_size = 1
            elif delta_time < 4:
                text = "PAPER!"
                text_size = 1
                choose = True
            elif delta_time < 5:
                text = "SCISSORS!"
                text_size = 1
            elif delta_time < 5.5:
                # pause to allow for pose settle
                text = ""
                pose_record.append(pose)
                pose_num_record.append(pose_num)
            elif delta_time < 7.5 and not round_done:
                pose_final = find_majority(pose_record)
                pose_num_final = find_majority(pose_num_record)
                if pose_final == "No hand detected":
                    pose_visible = False
                else:
                    pose_visible = True
                text = "You choose: " + pose_final
                text_size = 1.0
                # reset everything
                round_done = True
                pose_record = []
                pose_num_record = []
            elif delta_time < 7.5:
                text = "You choose: " + pose_final
                text_size = 1.0
            elif delta_time < 9.5:
                text = "Computer choose: " + class_map[computer_pose_num]
                text_size = 1.0
            elif delta_time < 11.5:
                if pose_visible:
                    text = "Winner: " + rps_outcome(computer_pose_num, pose_num_final)
                    text_size = 1.0
                else:
                    text = "False game!"
                    text_size = 1.0
            else:
                started = False
        elif round_done:
            text = "Press s to restart or q to end the program."
            text_size = 0.7
        else:
            text = "Press s to start the rock, paper, scissors game!"
            text_size = 0.7
        key_press = cv2.waitKey(1)
        if key_press == ord('s'):
            started = True
            round_done = False
            start_time = time.time()
        elif key_press == ord('q'):
            print("End")
            break
        # inserting text on video
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_extend, _ = cv2.getTextSize(text, font, np.double(text_size), 2)
        text_w, text_h = text_extend
        cv2.rectangle(frame, (50,60), (50 + text_w, 60 - (15+text_h)), (100,100,100), -1)
        cv2.putText(frame, text, (50, 50), font, text_size, (63, 208, 244),\
                    2, cv2.LINE_4)
        # Show the final output
        cv2.imshow("Output", frame)

    # release the webcam and destroy all active windows
    cap.release()
    cv2.destroyAllWindows()
