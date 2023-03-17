import cv2
import mediapipe as mp
import pyautogui
from gaze_tracking import GazeTracking
import dlib
import imutils
import argparse
from imutils import face_utils
from gaze_tracking.mouth import mouth_aspect_ratio


def parse_args():
    parser = argparse.ArgumentParser(description="Mouth control with eyes and mouth")

    parser.add_argument("--model_path", default="./gaze_tracking/trained_models/shape_predictor_68_face_landmarks.dat",
                        help="path of pretrained model for shape predictor")
    parser.add_argument("--MAR_THRESH", type=float, default=0.79,
                        help="thresh of MAR")

    return parser.parse_args()


args = parse_args()
gaze = GazeTracking()
webcam = cv2.VideoCapture(0)

face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args.model_path)

click_flag = 0
close_count = 0
Mouse_flag = False
# grab the indexes of the facial landmarks for the mouth
(mStart, mEnd) = (49, 68)

if __name__ == "__main__":
    while True:
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
            landmarks = landmark_points[0].landmark
            for id, landmark in enumerate(landmarks[474:478]):
                x = int(landmark.x * frame_w)
                y = int(landmark.y * frame_h)
                cv2.circle(frame, (x, y), 3, (0, 255, 0))

                if id == 1:
                    screen_x = screen_w * landmark.x
                    screen_y = screen_h * landmark.y
                    pyautogui.moveTo(screen_x, screen_y)

            left = [landmarks[145], landmarks[159]]
            for landmark in left:
                x = int(landmark.x * frame_w)
                y = int(landmark.y * frame_h)
                cv2.circle(frame, (x, y), 3, (0, 255, 255))

            print(f"rects = {rects}")
            # get mouth
            if len(rects) > 0:
                cv2.putText(frame, "Detect mouth, you can draw", (15, 85), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 1)

                shape = predictor(gray, rects[0])
                shape = face_utils.shape_to_np(shape)

                mouth = shape[mStart:mEnd]
                mouthMAR = mouth_aspect_ratio(mouth)
                mar = mouthMAR
                # compute the convex hull for the mouth, then visualize the mouth
                mouthHull = cv2.convexHull(mouth)
                # cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
                cv2.putText(frame, "MAR: {:.2f}".format(mar), (650, 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # Draw text if mouth is open
                if mar > args.MAR_THRESH:
                    cv2.putText(frame, "Mouse state changed!", (800, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if Mouse_flag:
                        Mouse_flag = False
                        pyautogui.sleep(0.5)
                        pyautogui.mouseUp()
                    else:
                        Mouse_flag = True
                        pyautogui.sleep(0.5)
                        pyautogui.mouseDown()

        if left_pupil != None and right_pupil != None:
            close_count = 0

            cv2.putText(frame, "Detect pupils, you can click", (15, 50), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 1)

            if gaze.is_blinking():
                text = "Blinking"
                click_flag = click_flag ^ 1
                pyautogui.click()
                pyautogui.sleep(0.5)
                    
        else:
            close_count += 1
            Mouse_message = "Not Painting"
        
        if Mouse_flag:
            Mouse_message = "Painting"
        else:
            Mouse_message = "Not Painting" 

        cv2.putText(frame, text, (90, 100), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)
        cv2.putText(frame, f"{Mouse_message}", (90, 160), cv2.FONT_HERSHEY_DUPLEX, 1.6, (127,0,224), 1)
        # cv2.line(frame, left_pupil, right_pupil, (0,0,255), 1, 8)
        cv2.imshow('Eye Controlled Mouse', frame)
        cv2.waitKey(1)
