import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append("./eye_movement")
import cv2
import mediapipe as mp
import pyautogui
from gaze_tracking import GazeTracking
import dlib
import imutils
import argparse
from imutils import face_utils
from gaze_tracking.mouth import mouth_aspect_ratio
import time


def parse_args():
    parser = argparse.ArgumentParser(description="Mouth control with eyes and mouth")

    parser.add_argument("--model_path", default="./gaze_tracking/trained_models/shape_predictor_68_face_landmarks.dat",
                        help="path of pretrained model for shape predictor")
    parser.add_argument("--MAR_THRESH", type=float, default=0.79,
                        help="thresh of MAR")
    parser.add_argument("--mStart", type=int, default=49,
                        help="start index of the facial landmarks for the mouth")
    parser.add_argument("--mEnd", type=int, default=68,
                        help="end index of the facial landmarks for the mouth")
    parser.add_argument("--yawn_weight", type=float, default=1.,
                        help="yawn weight")
    parser.add_argument("--open_too_long_weight", type=float, default=3.,
                        help="eye open too long weight")        
    parser.add_argument("--open_too_long_time_weight", type=float, default=0.1,
                        help="eye open too long time weight")
    parser.add_argument("--close_too_long_weight", type=float, default=2.,
                        help="eye close too long weight")             
    parser.add_argument("--close_count_weight", type=float, default=0.1,
                        help="eye close count weight") 
    parser.add_argument("--weight_bias", type=float, default=0.5,
                        help="eye close count weight")

    parser.add_argument("--whole_eeg_weight", type=float, default=1.,
                        help="eeg weight")
    parser.add_argument("--whole_eye_weight", type=float, default=1.,
                        help="eye movement weight")

    return parser.parse_args()


def eye_init(outcall):
    args = parse_args()
    gaze = GazeTracking()
    webcam = cv2.VideoCapture(0)

    if outcall:
        args.model_path = "./eye_movement/gaze_tracking/trained_models/shape_predictor_68_face_landmarks.dat"

    face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
    screen_w, screen_h = pyautogui.size()
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args.model_path)

    click_flag = 0
    close_count = 0
    Mouse_flag = False
    click_time = time.time()

    return args, gaze, webcam, face_mesh, screen_w, screen_h, detector, predictor, click_flag, close_count, Mouse_flag, click_time


def get_general_landmarks(landmark_points, frame, frame_w, frame_h, rects, gray, args, gaze, webcam, face_mesh, screen_w, screen_h, detector, predictor, click_flag, close_count, Mouse_flag, click_time):
    landmarks = landmark_points[0].landmark
    yawn_flag = 0
    for id, landmark in enumerate(landmarks[474:478]):
        x = int(landmark.x * frame_w)
        y = int(landmark.y * frame_h)
        cv2.circle(frame, (x, y), 3, (0, 255, 0))

        if id == 1:
            screen_x = screen_w * landmark.x
            screen_y = screen_h * landmark.y
            # pyautogui.moveTo(screen_x, screen_y)

    left = [landmarks[145], landmarks[159]]
    for landmark in left:
        x = int(landmark.x * frame_w)
        y = int(landmark.y * frame_h)
        cv2.circle(frame, (x, y), 3, (0, 255, 255))

    print(f"rects = {rects}")
    # get mouth
    if len(rects) > 0:
        # cv2.putText(frame, "Detect mouth, you can draw", (15, 85), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 1)

        shape = predictor(gray, rects[0])
        shape = face_utils.shape_to_np(shape)

        mouth = shape[args.mStart:args.mEnd]
        mouthMAR = mouth_aspect_ratio(mouth)
        mar = mouthMAR
        # compute the convex hull for the mouth, then visualize the mouth
        mouthHull = cv2.convexHull(mouth)
        # cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
        cv2.putText(frame, "MAR: {:.2f}".format(mar), (650, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # Draw text if mouth is open
        if mar > args.MAR_THRESH:
            yawn_flag = 1
            cv2.putText(frame, "Yawn!", (300, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if Mouse_flag:
                Mouse_flag = False
                pyautogui.sleep(0.5)
                # pyautogui.mouseUp()
            else:
                Mouse_flag = True
                pyautogui.sleep(0.5)
                # pyautogui.mouseDown()

    return yawn_flag, Mouse_flag


def get_data(dataset_path, count):
    # Return real-time-updated Output.mat path in `dataset_path`
    # Return Boolean Flag `update_eeg_weight`, if Output.mat updates, Flag = True; else Flag = Flase
    data_path = f"{dataset_path}default.mat"
    if os.path.exists(f"{dataset_path}dataOut{count}.mat"):
        data_path = f"{dataset_path}dataOut{count}.mat"
        count += 1
        
    return data_path, count


def get_eye_weight(yawn_flag, open_too_long_flag, open_too_long_time, close_too_long_flag, close_count, args):
    eye_weight =    args.weight_bias + yawn_flag * args.yawn_weight + \
                    open_too_long_flag * args.open_too_long_weight + open_too_long_time * args.open_too_long_time_weight +\
                    close_too_long_flag * args.close_too_long_weight + close_too_long_flag * close_count * args.close_count_weight 

    return eye_weight


def get_eeg_weight(inference_func, dataset_path, model, count):
    data_path, count = get_data(dataset_path, count)
    eeg_weight =    inference_func(data_path, model)

    return eeg_weight, count

def get_all_weight(eye_weight, eeg_weight, args):
    whole_weight = eye_weight * args.whole_eye_weight + eeg_weight * args.whole_eeg_weight + args.weight_bias
    
    return whole_weight


def eye_movement_process(inference_func, dataset_path, model, eeg_weight=None, update_eeg_weight=None, outcall=False):
    args, gaze, webcam, face_mesh, screen_w, screen_h, detector, predictor, click_flag, close_count, Mouse_flag, click_time = eye_init(outcall)
    count = 1
    while True:
        open_too_long_flag = 0
        close_too_long_flag = 0
        yawn_flag = 0
        open_too_long_time = 0
        text = ""
        _, frame = webcam.read()
        frame = cv2.flip(frame, 1)
        gaze.refresh(frame)

        frame = gaze.annotated_frame()
        frame = imutils.resize(frame, width=800, height=520)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        size = gray.shape
        # detect faces in the grayscale frame
        rects = detector(gray, 0)
        output = face_mesh.process(frame)
        landmark_points = output.multi_face_landmarks
        frame_h, frame_w, _ = frame.shape

        left_pupil = gaze.pupil_left_coords()
        right_pupil = gaze.pupil_right_coords()

        if landmark_points:
            yawn_flag, Mouse_flag = get_general_landmarks(landmark_points, frame, frame_w, frame_h, rects, gray, args, gaze, webcam, face_mesh, screen_w, screen_h, detector, predictor, click_flag, close_count, Mouse_flag, click_time)


        if left_pupil != None and right_pupil != None:
            close_count = 0

            cv2.putText(frame, "Normal Operation", (15, 50), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 1)

            if gaze.is_blinking():
                text = "Blinking"
                click_flag = click_flag ^ 1
                click_time = time.time()
                # pyautogui.click()
                # pyautogui.sleep(0.5)

            if time.time() - click_time > 10:
                open_too_long_flag = 1
                open_too_long_time = time.time() - click_time
                cv2.putText(frame, "Eyes Open too long!", (200, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

                    
        else:
            cv2.putText(frame, "Eyes Closed!", (0, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            close_count += 1
            Mouse_message = "Not Painting"
            if close_count > 10:
                close_too_long_flag = 1
                cv2.putText(frame, "Eyes Closed too long!", (40, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                click_time = time.time()
        
        if Mouse_flag:
            Mouse_message = "Painting"
        else:
            Mouse_message = "Not Painting" 

        eye_weight = get_eye_weight(yawn_flag, open_too_long_flag, open_too_long_time, close_too_long_flag, close_count, args)
        eeg_weight, count = get_eeg_weight(inference_func, dataset_path, model, count)
        whole_weight = get_all_weight(eye_weight, eeg_weight, args)

        cv2.putText(frame, text, (90, 100), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)
        cv2.putText(frame, f"{Mouse_message}", (90, 160), cv2.FONT_HERSHEY_DUPLEX, 1.6, (127,0,224), 1)
        cv2.putText(frame, f"{close_count}", (90, 220), cv2.FONT_HERSHEY_DUPLEX, 1.6, (127,0,224), 1)
        cv2.putText(frame, f"Watch Time: {(time.time() - click_time):.3f} s", (20, 280), cv2.FONT_HERSHEY_DUPLEX, 1.6, (127,0,224), 1)
        cv2.putText(frame, f"Count: {count}", (20, 360), cv2.FONT_HERSHEY_DUPLEX, 1.6, (127,0,224), 1)

        cv2.putText(frame, f"EEG: {eeg_weight:.3f}", (500, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Eye: {eye_weight:.3f}", (360, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Whole: {whole_weight:.3f}", (200, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # cv2.line(frame, left_pupil, right_pupil, (0,0,255), 1, 8)
        cv2.imshow('Eye Controlled Mouse', frame)
        cv2.waitKey(1)


if __name__ == "__main__":
    eye_movement_process()