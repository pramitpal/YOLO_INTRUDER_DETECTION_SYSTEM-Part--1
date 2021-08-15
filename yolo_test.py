
import os
import cv2
import numpy as np
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

##################################################
testing_video_file='walking.avi' ## test video file

line_pos=400 ####GREEN LINE ON SCREEN BELOW WHICH IF PERSON IS PRESENT THEN MAIL IS SENT
mail_flag=True ## if set to TRUE mail is sent to the given mail address
save_video_on_alert=False ## set to TRUE if want to save video file
save_file='intruder.avi' ## save video file name

#The mail addresses and password
sender_address = 'exampleSender@gmail.com'      ########## change the sender email address
sender_pass = 'password123'                     ########## change the password
receiver_address = 'exampleReceiver@gmail.com'  ########## chane the receiver email address

#version of yolo to be used
yolo_version=4 ## which version of yolo to be used (2,3,4) #### ONLY YOLO 4 TINY MODEL IS PRESENT IN THIS REPO##################
use_tiny=True ## use TRUE for fast processing-- use TINY MODEL
##################################################
###################################### Contents of the sent email (CHANGE IF YOU WISH) #####################
subject='Intruder Alert'
mail_content = '''
Hello,
There is some intruder in the private premises.
Thank You
'''
##################################################

def send_mail(sender,receiver,passw,subject,body):
    message = MIMEMultipart()
    message['From'] = sender
    message['To'] = receiver
    message['Subject'] = subject   #The subject line
    # message['body'] = mail_content
    #The body and the attachments for the mail
    message.attach(MIMEText(body, 'plain'))
    #Create SMTP session for sending the mail
    session = smtplib.SMTP('smtp.gmail.com', 587) #use gmail with port
    session.starttls() #enable security
    session.login(sender, passw) #login with mail_id and password
    text = message.as_string()
    session.sendmail(sender, receiver, text)
    session.quit()
    print('Mail Sent')
    
if use_tiny:
    weight_file='yolov' + str(yolo_version) + '-tiny.weights'
    cfg_file='yolov' + str(yolo_version) + '-tiny.cfg'    
else:
    weight_file='yolov' + str(yolo_version) + '.weights'
    cfg_file='yolov' + str(yolo_version) + '.cfg' 
print('model used='+weight_file)
net = cv2.dnn.readNet(weight_file, cfg_file)

classes = []
with open("coco.names", "r") as f:
    classes = f.read().splitlines()

cap = cv2.VideoCapture('walking.avi')
font = cv2.FONT_HERSHEY_PLAIN
colors =(0,255,0)

codec = cv2.VideoWriter_fourcc(*'XVID')
vid_fps =int(cap.get(cv2.CAP_PROP_FPS))
vid_width,vid_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# out = cv2.VideoWriter('results.avi', codec, vid_fps, (vid_width, vid_height))
intruder_counter=0
intruder_flag=False
record_on=False
_, img = cap.read()
height, width, _ = img.shape
prev_center_y=height

if os.path.isfile(save_file):
    os.remove(save_file)

while cap.isOpened():
    
    _, img = cap.read()
    if img.any():
        height, width, _ = img.shape
    if (yolo_version==3 or yolo_version==4) and use_tiny==False:
        h,w=416,416
    else:
        h,w=416,416
    blob = cv2.dnn.blobFromImage(img, 1/255, (h, w), (0,0,0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []
    

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)
    cv2.line(img, (0,line_pos),(width,line_pos),(0,255,0), thickness=2)
    
    if prev_center_y<line_pos and center_y <= height and center_y>= line_pos and intruder_flag==False:
        intruder_flag=True
        
    if intruder_flag==True:
        record_on=True
        if not os.path.isfile(save_file) and save_video_on_alert:
            print("file created")
            out = cv2.VideoWriter(save_file, codec, vid_fps, (vid_width, vid_height))
            
        cv2.putText(img, 'Intruder Alert(RECORD ON) press '+ 'q' +' to stop',(10,50), font, 2, (255,255,255), 2)
        if mail_flag:
            mail_flag=False
            send_mail(sender_address,receiver_address,sender_pass,subject,mail_content)
    
    prev_center_y=center_y
    if len(indexes)>0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i],2))
            color = (0,255,0)
            cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
            # cv2.putText(img, label + " " + confidence, (x, y+20), font, 1, (255,255,255), 1)

    cv2.imshow('Image', img)
    if save_video_on_alert and record_on:
        # print('recording')
        out.write(img)
    key = cv2.waitKey(1)
    if key==27:
        break
    if key == ord('q'):
        print('stop key')
        record_on=False
        if record_on:
            out.release()

    
cap.release()
cv2.destroyAllWindows()
#######################################################################
