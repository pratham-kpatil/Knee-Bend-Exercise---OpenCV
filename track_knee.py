import cv2
import mediapipe as mp
import numpy as np
import time


class Knee_Exercise():
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_pose = mp.solutions.pose

    def calculate_angle(self,a,b,c):
        '''
            Function to calculate the angle between 3 points
            input: 2D co-ordinates of three points
            output : angle in betweeen those points
        '''
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle >180.0:
            angle = 360-angle
            
        return round(angle,2)


    def count_repitations(self,video_path):
        counter = 0 
        rep_time = 0
        start_time = time.time()
        knee_angle = 400
        position = None
        fm_timer = 0
        min_knee_angle = 400
        prev_knee_y = 0
        skip = 0

        # open a file to write the exercise stats in txt file
        file = open('exercise_report.txt', 'w')

        # read video
        cap = cv2.VideoCapture(video_path)

        # create a object to write frames 
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter('ProcessedVideo.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, (width,height))

        # start pose estimation
        with self.mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7) as pose:
            while cap.isOpened():
                success, image=cap.read()
                if not success:
                    print("Empty video")
                    break
                
                image = cv2.cvtColor(cv2.flip(image,1), cv2.COLOR_BGR2RGB)
                result = pose.process(image)

                imlist = []

                # if landmarks are available plot them
                if result.pose_landmarks:
                    self.mp_drawing.draw_landmarks(
                        image, result.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)   
                
                try:
                    landmarks = result.pose_landmarks.landmark
                    left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].z
                    right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].z

                    # take the co-ordinates of the leg closer to camera
                    if right_hip<left_hip:
                        hip = [landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                        knee = [landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                        ankle = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                    else:
                        hip = [landmarks[self.self.mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
                        knee = [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                        ankle = [landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                    # calulcate knee angle
                    knee_angle = self.calculate_angle(hip, knee, ankle)
                    
                    # update the postion of the knee according to the angle and also count the repetitions
                    if knee_angle > 169 and position=="up":
                            position = "down"

                            # reset repetition start time
                            start_time = 0

                            # check if the position was held was for minimum 8 sec
                            if rep_time>= 8:
                                counter += 1
                                file.writelines(["Repetition number : {} \n".format(counter),
                                                "Minimum knee angle : {} degree\n".format(min_knee_angle),
                                                "Repetition Time : {} seconds\n \n".format(rep_time)]
                                        )
                                
                                # reset minimum knee angle
                                min_knee_angle = 1e9

                            else:
                                fm_timer = 40

                    if knee_angle <= 140:
                            min_knee_angle = min(min_knee_angle, knee_angle)
                            position="up"

                    if position=="down" and knee_angle>169:
                        start_time = time.time()

                    # update the time for ongoing repetition
                    rep_time = (time.time() - start_time)


                except Exception as e:
                    pass

                image = cv2.flip(image,1)
                
                # add the required information to image
                cv2.putText(image, "Knee angle : {} degree".format(str(knee_angle)), 
                                (100,100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2, cv2.LINE_AA
                            )

                cv2.putText(image, "Repetitions : {}".format(str(counter)),  
                                (100,130),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2, cv2.LINE_AA
                            )
                
                cv2.putText(image, "Rep time : {} seconds".format(str(int(rep_time))),  
                                (100,160),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2, cv2.LINE_AA
                            )
                
                # display feedback keep your knee bent on the screen
                if fm_timer != 0:
                            cv2.putText(image, "Keep your knee bent",  
                                (100,200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA
                            )
                            fm_timer -= 1
                
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # cv2.imshow('Knee Exercise',image)
                writer.write(image)
                key = cv2.waitKey(1)
                if key==ord('q'):
                    break

        file.close()
        cap.release()


# create a object of Knee_Exercise class
knee_exercise_obj = Knee_Exercise()

# call the knee_exercise function by passing the video path as argument
knee_exercise_obj.count_repitations("KneeBendVideo.mp4")