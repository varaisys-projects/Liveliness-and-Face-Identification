import random 
import numpy as np
import traceback
import questions
import pandas as pd
import ast
from dotenv import load_dotenv
load_dotenv('.env')
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify
import face_recognition
import base64
from flask import Flask, render_template
from flask_socketio import SocketIO, join_room
import time
import os
import json
import logging
import cv2
import f_liveness_detection
import imutils
from openpyxl import Workbook
from PIL import Image
log_format = '%(asctime)s  %(levelname)5s -[%(pathname)s:%(lineno)d in function %(funcName)s]- %(message)s'
logging.basicConfig(level='INFO', format=log_format)

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")                                               # Allow connections from all origins
from flask_cors import CORS
CORS(app, resources={r"/*": {"origins": "*"}})

path="./data/encodings/encoding.xlsx"
# remote_server_ip=os.getenv("REMOTE_SREVER_IP")
remote_server_port=os.getenv("REMOTE_SERVER_PORT")
tolerance_of_model=float(os.getenv("TOLERANCE_OF_MODEL"))
time_limit=int(os.getenv("STARTING_TIME_LIMIT_FOR_PERSON_TO_BE_AVAILABLE_IN_FRAME"))                              ### time limit 10 seconds                                
frames_required=int(os.getenv("FRAMES_REQUIRED_FOR_PERSON_TO_BE_AVAILABLE_IN_FRAME_TO_LIVELINESS_TEST"))
time_limit_for_blink=int(os.getenv("TIME_LIMIT_FOR_TOTAL_BLINK_DETECTION"))                                        ### time to done each task
time_linit_for_each_action_in_face_movement_function=int(os.getenv("TIME_LIMIT_FOR_EACH_ACTION_IN_FACE_MOVEMENT_FUNCTION"))    ### time to done each task


@app.route('/upload_image',methods=['POST'])
def index():
    try:
        flag=1
        logging.info("inside upload_image")
        if os.path.isfile(path):
            logging.info("excel exists")
            df = pd.read_excel(path)
        else:
            workbook = Workbook()
            workbook.save(path) 
            logging.info("excel created")
            df = pd.DataFrame(columns = ['Name', 'Encoding','Timestamp','Image_Name'])

        data= request.get_json()
        frame = data['image']
        Name = data['name']
        # print(type(frame),str(frame)[:100])
        # print(Name)
        image_data = frame.split(",")[1] if "," in frame else frame
        image_bytes = base64.b64decode(image_data)
            
        im = np.frombuffer(image_bytes, np.uint8)
        im = cv2.imdecode(im, cv2.IMREAD_COLOR)
        im = imutils.resize(im, width=720)        
        name=str(Name)
        timee=str(int(time.time()*1000))
        cv2.imwrite("data/people/"+name+"_"+timee+".jpg",im)                                                                 ### SAVING IMAGE

        gray_image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        equalized_image = cv2.equalizeHist(gray_image)
        im = cv2.imread("data/people/"+name+"_"+timee+".jpg")

        # im = cv2.resize(im, (480,640), interpolation = cv2.INTER_LINEAR)
        encoding = face_recognition.face_encodings(im)                                 ## check if find a person else return error

        image = cv2.imread("data/people/"+name+"_"+timee+".jpg")
        contrast_threshold=50
        sharpness_threshold=50, 
        brightness_threshold=150
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_variance = laplacian.var()
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        brightness = np.sum(hist * np.arange(256)) / np.sum(hist)
        contrast = np.max(hist) - np.min(hist)
        sharpness_quality = 100.0 if laplacian_variance > sharpness_threshold else (laplacian_variance / sharpness_threshold) * 100.0
        brightness_quality = 100.0 if brightness > brightness_threshold else (brightness / brightness_threshold) * 100.0
        contrast_quality = 100.0 if contrast > contrast_threshold else (contrast / 50.0) * 100.0  

        if(len(encoding)==0):
            message="No Person Detected in the image"
            flag=0
        elif(len(encoding)>1):
            message="More than One Person Detected in the image"
            flag=0
        elif(sharpness_quality<100 or brightness_quality<100 or contrast_quality<100):
            flag=0
            message="Quality of image is not good"

        if(flag==0):
            logging.info("Some Error occur in uploading image")
            return {"upload_message":message,"result":"Failed"}

        x=encoding[0].tobytes()

        df.loc[len(df.index)] = [name,list(x),timee,name+"_"+timee+".jpg"]  
        df.to_excel(path,index=False)
        logging.info("excel updated successfully with name as :--  %s",name)
        logging.info("IMAGE RECEIVED")
        return {"upload_message":"Successfully Uploaded, Start Liveliness Detection","result":"Passed"}
    except Exception as e:
        err_message = traceback.format_exc()
        err = err_message.replace('"', '')
        err = err.replace("'", '')
        logging.error(traceback.format_exc())
        return {"upload_message":"Some Error occured while uploading the image","result":"Failed"}


@socketio.on('connect')
def handle_connect():
    user_id = request.sid                                                                 # Unique identifier for each connected client
    room_name = f'user_{user_id}'
    join_room(room_name)
    socketio.emit('message', f'Welcome, you are in room {room_name}', room=user_id)
    logging.info(f'Client {user_id} connected and joined room {room_name}')
    global MAIN_FRAME
    MAIN_FRAME[user_id]=""


@socketio.on('disconnect')
def handle_disconnect():
    user_id = request.sid
    logging.info(f'Client {user_id} disconnected')
    global MAIN_FRAME
    del MAIN_FRAME[user_id]


def face_black(image,flagg=0):
    height,width=image.shape[0],image.shape[1]
    center_coordinates = (width//2,height//2) 
    if(flagg==0):
        axesLength = (int(min(height,width)/3),int(min(height,width)/4))           
    else:
        axesLength = (int(min(height,width)/5),int(min(height,width)/6)) 
    angle = 90
    startAngle = 0
    endAngle = 360
    color = (0,0,0) 
    thickness = 5
    shapes = np.zeros([height,width,3],dtype=np.uint8)
    shapes.fill(255) 
    shapes = cv2.ellipse(shapes, center_coordinates, axesLength, angle, startAngle, endAngle, color, -1) 

    out = image.copy()
    alpha = 0.05
    mask = shapes.astype(bool)
    out[mask] = cv2.addWeighted(image, alpha, shapes, 1 - alpha, 0)[mask]
    return out


@socketio.on('frame_from_client')
def frame_from_client(frame):
    user_id = request.sid
    room_name = f'user_{user_id}'
    image_data = frame.split(",")[1] if "," in frame else frame
    image_bytes = base64.b64decode(image_data)

    global MAIN_FRAME
    user_id = request.sid
    MAIN_FRAME[user_id]=image_bytes

    # im = np.frombuffer(MAIN_FRAME, np.uint8)
    # im = cv2.imdecode(im, cv2.IMREAD_COLOR)
    # im = imutils.resize(im, width=720)
    # cv2.imwrite("zzz"+str(time.time()-start)+".jpg",im)


@socketio.on('camera_started')
def camera_started_function(paramen):
    # print("-----------------------------",type(paramen),paramen,)
    logging.info("inside camera_started_function")
    user_id = request.sid
    room_name = f'user_{user_id}'

    socketio.emit('request_frame', room=user_id)
    time.sleep(0.5)
    st=time.time()
    
    color = (255,255,255)
    rect_color=(253,110,13)
    rect1=(0,30)                                   ##### upper
    rect2=(790,70)
    text_po=(20,60)
    rect11=(0,40)                                    ##### lower
    rect22=(790,80)
    text_po2=(180,70)
    color2=(255,165,0)
    x=time.time()-st
    cou=0
    flag=1
    cou=0
    completed=0
    socketio.emit('sound_instructions', " Please adjust your face inside the oval ", room=user_id)
    while(x<time_limit):
        x=time.time()-st
        socketio.emit('request_frame', room=user_id)

        ret, im = True,MAIN_FRAME[user_id]
        im = np.frombuffer(im, np.uint8)
        im = cv2.imdecode(im, cv2.IMREAD_COLOR)
        im = imutils.resize(im, width=720)
        im=face_black(im)

        im = cv2.flip(im, 1) 
        cv2.rectangle(im, rect1, rect2, rect_color, -1)
        cv2.putText(im,"Please adjust your face inside the oval",text_po,cv2.FONT_HERSHEY_COMPLEX,1,color,2)        


        frame=face_black(im,flagg=1)
        small_frame = cv2.resize(frame, (0, 0), fx=0.250, fy=0.250)
        face_locations = face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)

        result="outside"
        if(len(face_encodings)>0):
            result="inside"
            flag=1
            cou+=1
        else:
            flag=0
            cou=0
        # cv2.rectangle(im, rect11, rect22, rect_color, -1)
        # cv2.putText(im,result+str(cou),text_po2,cv2.FONT_HERSHEY_COMPLEX,1,color2,2)


        frame=im
        ret, buffer = cv2.imencode('.jpg', frame)
        base64_encoded = base64.b64encode(buffer)
        frame = base64_encoded.decode("utf-8")
        frame="data:image/jpeg;base64,"+frame
        socketio.emit('response_from_server', frame, room=user_id)
        if(cou>frames_required):                                                  #### wait for 50 frames where person is inside it
            if(paramen=="blink"):
                blink_eye_function(user_id)
            else:
                face_movement_function(user_id)
            completed=1
            break
    if completed==0:
        socketio.emit('sound_instructions', " Timeout, Please reload ", room=user_id)
        blank_image = np.zeros((540,720,3), np.uint8)
        im = imutils.resize(blank_image, width=720)
        cv2.putText(im,"TimeOut, Please reload..",(120,260),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
        frame=im
        ret, buffer = cv2.imencode('.jpg', frame)
        base64_encoded = base64.b64encode(buffer)
        frame = base64_encoded.decode("utf-8")
        frame="data:image/jpeg;base64,"+frame
        socketio.emit('response_from_server', frame, room=user_id)
        logging.info("TEST COMPLETED result- person not found in frame")
        socketio.emit('test_complete', frame, room=user_id)


def blink_eye_function(user_id):
    logging.info("inside blink_eye_function")
    global MAIN_FRAME
    global start

    start=time.time()

    # time.sleep(1)
    socketio.emit('request_frame', room=user_id)
    time.sleep(0.5)

    logging.info("test has benn started")  

    cou=0
    color = (255,255,255)
    rect_color=(253,110,13)

    rect1=(100,0)                                   ##### how many blinks done
    rect2=(590,40)
    text_po=(170,30)
    

    rect11=(100,40)                                    ##### total blinks needed
    rect22=(590,80)
    text_po2=(120,70)
    color2=(255,165,0)

    COUNTER, TOTAL = 0,0
    counter_ok_questions = 0
    counter_ok_consecutives = 0
    limit_consecutives = 2
    limit_questions = random.randint(1,3)
    counter_try = 0
    limit_try = time_limit_for_blink                                                              ### time to done each task
    persons=[]


    for i_questions in range(0,1):
        socketio.emit('sound_instructions', " blink "+str(limit_questions)+" times", room=user_id)
        flag=1
        index_question = 0
        question = questions.question_bank(index_question)

        socketio.emit('request_frame', room=user_id)
        ret, im = True,MAIN_FRAME[user_id]

        im = np.frombuffer(im, np.uint8)
        im = cv2.imdecode(im, cv2.IMREAD_COLOR)
        cou+=1

        im = imutils.resize(im, width=720)
        im=face_black(im)
        im = cv2.flip(im, 1) 
        
        cv2.rectangle(im, rect1, rect2, rect_color, -1)
        cv2.putText(im,question+" "+str(limit_questions)+" time(s)",text_po,cv2.FONT_HERSHEY_COMPLEX,1,color,2)
        cv2.rectangle(im, rect11, rect22, rect_color, -1)
        cv2.putText(im,"Blink count "+str(TOTAL)+", Please Wait",text_po2,cv2.FONT_HERSHEY_COMPLEX,1,color2,2)

        success, frame = ret,im         # read the cam frame
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            base64_encoded = base64.b64encode(buffer)
            frame = base64_encoded.decode("utf-8")
            frame="data:image/jpeg;base64,"+frame
            socketio.emit('response_from_server', frame, room=user_id)

        if cv2.waitKey(1) &0xFF == ord('q'):
            break 

        for i_try in range(limit_try):
            socketio.emit('request_frame', room=user_id)
            ret, im = True,MAIN_FRAME[user_id]
            im = np.frombuffer(im, np.uint8)
            im = cv2.imdecode(im, cv2.IMREAD_COLOR)
            cou+=1
            im = imutils.resize(im, width=720)
            im=face_black(im)      
            im = cv2.flip(im, 1)  

            cv2.rectangle(im, rect1, rect2,rect_color, -1)
            cv2.putText(im,question+" "+str(limit_questions)+" time(s)",text_po,cv2.FONT_HERSHEY_COMPLEX,1,color,2)  
            cv2.rectangle(im, rect11, rect22, rect_color, -1)
            cv2.putText(im,"Blink count "+str(TOTAL)+", Please Wait",text_po2,cv2.FONT_HERSHEY_COMPLEX,1,color2,2)

            success, frame = ret,im   
            if not success:
                break
            else:
                ret, buffer = cv2.imencode('.jpg', frame)
                base64_encoded = base64.b64encode(buffer)
                frame = base64_encoded.decode("utf-8")
                frame="data:image/jpeg;base64,"+frame
                socketio.emit('response_from_server', frame, room=user_id)

            im = cv2.flip(im, 1)
            TOTAL_0 = TOTAL
            out_model = f_liveness_detection.detect_liveness(im,COUNTER,TOTAL_0)
            TOTAL = out_model['total_blinks']
            COUNTER = out_model['count_blinks_consecutives']
            dif_blink = TOTAL-TOTAL_0

            if TOTAL == limit_questions+100:
                blinks_up = 1
            else:
                blinks_up = 0
            challenge_res = questions.challenge_result(question, out_model,blinks_up)

            socketio.emit('request_frame', room=user_id)
            ret, im = True,MAIN_FRAME[user_id]
            im = np.frombuffer(im, np.uint8)
            im = cv2.imdecode(im, cv2.IMREAD_COLOR)
            cou+=1
            im = imutils.resize(im, width=720)
            im=face_black(im)     

            im = cv2.flip(im, 1) 
            cv2.rectangle(im, rect1, rect2,rect_color, -1)
            cv2.putText(im,question+" "+str(limit_questions)+" time(s)",text_po,cv2.FONT_HERSHEY_COMPLEX,1,color,2)
            cv2.rectangle(im, rect11, rect22, rect_color, -1)
            cv2.putText(im,"Blink count "+str(TOTAL)+", Please Wait",text_po2,cv2.FONT_HERSHEY_COMPLEX,1,color2,2)
        
            success, frame = ret,im                                                                              # read the cam frame
            if not success:
                break
            else:
                ret, buffer = cv2.imencode('.jpg', frame)
                base64_encoded = base64.b64encode(buffer)
                frame = base64_encoded.decode("utf-8")
                frame="data:image/jpeg;base64,"+frame
                socketio.emit('response_from_server', frame, room=user_id)

            if cv2.waitKey(1) &0xFF == ord('q'):
                break

            if challenge_res == "pass":

                socketio.emit('request_frame', room=user_id)
                ret, im = True,MAIN_FRAME[user_id]
                im = np.frombuffer(im, np.uint8)
                im = cv2.imdecode(im, cv2.IMREAD_COLOR)
                cou+=1

                im = imutils.resize(im, width=720)
                im=face_black(im)     
                im = cv2.flip(im, 1) 
                cv2.rectangle(im, rect1, rect2, rect_color, -1)
                cv2.putText(im,question+" "+str(limit_questions)+" time(s)",text_po,cv2.FONT_HERSHEY_COMPLEX,1,color,2)
                cv2.rectangle(im, rect11, rect22, rect_color, -1)
                cv2.putText(im,"Blink count "+str(TOTAL)+", Please Wait",text_po2,cv2.FONT_HERSHEY_COMPLEX,1,color2,2)

                success, frame = ret,im  # read the camera frame
                if not success:
                    break
                else:
                    ret, buffer = cv2.imencode('.jpg', frame)
                    base64_encoded = base64.b64encode(buffer)
                    frame = base64_encoded.decode("utf-8")
                    frame="data:image/jpeg;base64,"+frame
                    socketio.emit('response_from_server', frame, room=user_id)

                if cv2.waitKey(1) &0xFF == ord('q'):
                    break

                counter_ok_consecutives += 1
                if counter_ok_consecutives == limit_consecutives:
                    counter_ok_questions += 1
                    counter_try = 0
                    counter_ok_consecutives = 0
                    break
                else:
                    continue
            elif challenge_res == "fail":
                counter_try += 1
            elif i_try == limit_try-1:
                break

        if counter_ok_questions ==  1:
            while True:
                flag=0
                output= " LIVENESS PASSED "

                socketio.emit('request_frame', room=user_id)
                ret, im = True,MAIN_FRAME[user_id]
                im = np.frombuffer(im, np.uint8)
                im = cv2.imdecode(im, cv2.IMREAD_COLOR)
                im = imutils.resize(im, width=720)
                im=face_black(im)      

                im = cv2.flip(im, 1) 
                cv2.rectangle(im, rect1, rect2, rect_color, -1)
                cv2.putText(im,question+" "+str(limit_questions)+" time(s)",text_po,cv2.FONT_HERSHEY_COMPLEX,1,color,2)
                cv2.rectangle(im, rect11, rect22, rect_color, -1)
                cv2.putText(im,"Blink count "+str(TOTAL)+", Please Wait",text_po2,cv2.FONT_HERSHEY_COMPLEX,1,color2,2)

                success, frame = ret,im              # read the cam frame
                if not success:
                    break
                else:
                    ret, buffer = cv2.imencode('.jpg', frame)
                    base64_encoded = base64.b64encode(buffer)
                    frame = base64_encoded.decode("utf-8")
                    frame="data:image/jpeg;base64,"+frame
                    socketio.emit('response_from_server', frame, room=user_id)

                if(flag==0):
                    break
                if cv2.waitKey(1) &0xFF == ord('q'):
                    break
        elif i_try == limit_try-1:
            while True:
                flag=0
                output=  "LIVENESS FAILED"

                socketio.emit('request_frame', room=user_id)
                ret, im = True,MAIN_FRAME[user_id]
                im = np.frombuffer(im, np.uint8)
                im = cv2.imdecode(im, cv2.IMREAD_COLOR)  
                im=face_black(im)     

                im = cv2.flip(im, 1)  
                im = imutils.resize(im, width=720)
                cv2.rectangle(im, rect1, rect2, rect_color, -1)
                cv2.putText(im,question+" "+str(limit_questions)+" time(s)",text_po,cv2.FONT_HERSHEY_COMPLEX,1,color,2)
                cv2.rectangle(im, rect11, rect22, rect_color, -1)
                cv2.putText(im,"Blink count "+str(TOTAL)+", Please Wait",text_po2,cv2.FONT_HERSHEY_COMPLEX,1,color2,2)

                success, frame = ret,im                          # read the cam frame
                if not success:
                    break
                else:
                    ret, buffer = cv2.imencode('.jpg', frame)
                    base64_encoded = base64.b64encode(buffer)
                    frame = base64_encoded.decode("utf-8")
                    frame="data:image/jpeg;base64,"+frame
                    socketio.emit('response_from_server', frame, room=user_id)

                if(flag==0):
                    break
                if cv2.waitKey(1) &0xFF == ord('q'):
                    break
            if(flag==0):
                break
            break 
        if(flag==0):
            break
        else:
            continue
    
    socketio.emit('request_frame', room=user_id)
    ret, im = True,MAIN_FRAME[user_id]
    im = np.frombuffer(im, np.uint8)
    im = cv2.imdecode(im, cv2.IMREAD_COLOR)  
    cou+=1 
    im=face_black(im)
    success, frame = ret,im  # read the cam frame

    small_frame = cv2.resize(frame, (0, 0), fx=0.250, fy=0.250)
    name = "Unknown"
    face_locations = face_recognition.face_locations(small_frame)
    face_encodings = face_recognition.face_encodings(small_frame, face_locations)
    sol=[]

    try:
        df = pd.read_excel(path)
        logging.info("Excel Found with shape %s",df.shape)
    except:
        df = pd.DataFrame(columns = ['Name', 'Encoding','Timestamp','Image_Name'])
        logging.info("No Excel Found ")

    for face_encoding in face_encodings:
        sol=[]
        final_name="Unknown"
        final_per =0
        for i in range(len(df)):
            name=df['Name'].loc[df.index[i]]
            x=df['Encoding'].loc[df.index[i]]
            x=ast.literal_eval(x)
            x=bytes(x)
            encoding=np.frombuffer(x)
            match = face_recognition.compare_faces([encoding], face_encoding,tolerance=1-tolerance_of_model)
            distance = face_recognition.face_distance([encoding], face_encoding)
            x=100-round(distance[0]*100,1)
            if(match[0]):
                print(name,x)
            if (match[0] and x>final_per):
                final_per =x
                final_name=name
        print(final_name,final_per)    
        sol.append(final_name)     
        logging.info("%s  %s",final_name,final_per)                            
    persons.extend(sol)
    logging.info("%s  %s",sol,persons)

    cv2.destroyAllWindows()
    blank_image = np.zeros((540,720,3), np.uint8)
    im = imutils.resize(blank_image, width=720)

    if(TOTAL==limit_questions):
        output= " LIVENESS PASSED "

    if(len(face_encodings)>1):
        cv2.putText(im,"There seem to be more than one person",(0,260),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
        socketio.emit('sound_instructions', " There seem to be more than one person ", room=user_id)
        # cv2.putText(im,"There seems to be a Technical problem",(0,260),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
        cv2.putText(im,"Kindly Restart",(200,300),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
        socketio.emit('sound_instructions', " Kindly Restart ", room=user_id)
        output="total persons in frame issue"
    elif(output=="LIVENESS FAILED" or len(persons)==0):                                                                                               
        cv2.putText(im,"LIVENESS FAILED",(200,260),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
        socketio.emit('sound_instructions', " Liveness failed ", room=user_id)
    elif(len(list(set(persons)))==1 and persons[0]=="Unknown"):                                                                  
        cv2.putText(im,"Person Unknown",(160,260),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
        socketio.emit('sound_instructions', " Person Unknown ", room=user_id)
    elif(len(list(set(persons)))==1 and persons[0]!="Unknown"):     
        cv2.putText(im,"Welcome "+str(persons[0]),(160,260),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
        socketio.emit('sound_instructions', "Welcome "+str(persons[0]) , room=user_id)
    else:
        cv2.putText(im,"There seems to be a Technical problem",(0,260),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
        socketio.emit('sound_instructions', " There seems to be a Technical problem ", room=user_id)
        # cv2.putText(im,"There seems to be a Technical problem",(0,260),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
        cv2.putText(im,"Kindly Restart",(200,300),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
        socketio.emit('sound_instructions', " Kindly Restart ", room=user_id)
        output="total persons in frame issue"

    logging.info("blink count  %s  ",TOTAL)
    logging.info("%s  %s",persons,output)
    logging.info("limit is  %s , Everything runs successfully and the output is       :---  %s",limit_questions,output)

    frame=im
    ret, buffer = cv2.imencode('.jpg', frame)
    base64_encoded = base64.b64encode(buffer)
    frame = base64_encoded.decode("utf-8")
    frame="data:image/jpeg;base64,"+frame
    socketio.emit('response_from_server', frame, room=user_id)
    cou+=1
    logging.info("TEST COMPLETED   %s",output)
    socketio.emit('test_complete', frame, room=user_id)

def face_movement_function(user_id):
    logging.info("inside face_movement_function")
    global MAIN_FRAME
    global start
    start=time.time()

    # time.sleep(1)
    socketio.emit('request_frame', room=user_id)
    time.sleep(0.5)
    logging.info("test has benn started")

    color = (255,255,255)
    rect_color=(253,110,13)
    rect1=(100,20)                                   ##### how many blinks done
    rect2=(590,60)
    text_po=(170,50)

    # rect11=(100,40)                                    ##### total blinks needed
    # rect22=(590,80)
    # text_po2=(120,70)

    # rect1=(0,460)
    # rect2=(660,520)
    # text_po=(10,500)

    COUNTER, TOTAL = 0,0
    counter_ok_questions = 0
    counter_ok_consecutives = 0
    limit_consecutives = 2
    limit_questions =  random.randint(2,3)
    counter_try = 0
    limit_try = time_linit_for_each_action_in_face_movement_function                         ### time to done each task
    persons=[]
    successs=0
    flag=1

    ret, im = True,MAIN_FRAME[user_id]
    detection_image=im
    for i_questions in range(limit_questions):
        not_detected=0
        detected=0
        turn=0
        if(flag==0):
            break
        flag=0
        index_question = random.randint(1,2)                                      #### check thisssssss
        question = questions.question_bank(index_question)
        if(question=="turn face right"):
            question_real="turn face left"
            socketio.emit('sound_instructions', "turn your face left ", room=user_id)
        else:
            question_real="turn face right"
            socketio.emit('sound_instructions', "turn your face right ", room=user_id)

        

        # ret, im = cam.read()
        socketio.emit('request_frame', room=user_id)
        ret, im = True,MAIN_FRAME[user_id]
        im = np.frombuffer(im, np.uint8)
        im = cv2.imdecode(im, cv2.IMREAD_COLOR)

        im = cv2.flip(im, 1)         
        im = imutils.resize(im, width=720)
        im=face_black(im)
        cv2.rectangle(im, rect1, rect2, rect_color, -1)
        cv2.putText(im,  question_real  ,text_po,cv2.FONT_HERSHEY_COMPLEX,1,color,2)

        success, frame = ret,im  # read the cam frame
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            base64_encoded = base64.b64encode(buffer)
            frame = base64_encoded.decode("utf-8")
            frame="data:image/jpeg;base64,"+frame
            socketio.emit('response_from_server', frame, room=user_id)
        if cv2.waitKey(1) &0xFF == ord('q'):
            break 

        for i_try in range(limit_try):
            socketio.emit('request_frame', room=user_id)
            ret, im = True,MAIN_FRAME[user_id]
            im = np.frombuffer(im, np.uint8)
            im = cv2.imdecode(im, cv2.IMREAD_COLOR)
            im2=im

            small_frame = cv2.resize(im, (0, 0), fx=0.250, fy=0.250)
            face_locations = face_recognition.face_locations(small_frame)
            if(len(face_locations)>0):
                result="detected"
            else:
                not_detected+=1
                detected=0
                result="not detected"

            im = imutils.resize(im, width=720)
            im=face_black(im)

            TOTAL_0 = TOTAL
            out_model = f_liveness_detection.detect_liveness(im,COUNTER,TOTAL_0)
            TOTAL = out_model['total_blinks']
            COUNTER = out_model['count_blinks_consecutives']
            dif_blink = TOTAL-TOTAL_0
            if dif_blink > 0:
                blinks_up = 1
            else:
                blinks_up = 0

            challenge_res = questions.challenge_result(question, out_model,blinks_up)
            socketio.emit('request_frame', room=user_id)
            ret, im = True,MAIN_FRAME[user_id]
            im = np.frombuffer(im, np.uint8)
            im = cv2.imdecode(im, cv2.IMREAD_COLOR)

            im = cv2.flip(im, 1)
            small_frame = cv2.resize(im, (0, 0), fx=0.250, fy=0.250)
            face_locations = face_recognition.face_locations(small_frame)
            if(len(face_locations)>0):
                detected+=1
                result="detected"
            else:
                not_detected+=1
                detected=0
                result="not detected"

            # print(not_detected,detected,challenge_res)
            if(not_detected>0  and detected>0  and challenge_res == "pass"):      
                turn+=1
                not_detected=0
                flag=1
            im = imutils.resize(im, width=720)
            im=face_black(im)
            cv2.rectangle(im, rect1, rect2, rect_color, -1)

            if(flag==1):                                    
                successs+=1
                cv2.putText(im,   question_real + " OK "   ,text_po,cv2.FONT_HERSHEY_COMPLEX,1,color,2)
                socketio.emit('sound_instructions', " Ok ", room=user_id)
            else:
                cv2.putText(im,   question_real   ,text_po,cv2.FONT_HERSHEY_COMPLEX,1,color,2)
            success, frame = ret,im  # read the cam frame
            if not success:
                break
            else:
                ret, buffer = cv2.imencode('.jpg', frame)
                base64_encoded = base64.b64encode(buffer)
                frame = base64_encoded.decode("utf-8")
                frame="data:image/jpeg;base64,"+frame
                socketio.emit('response_from_server', frame, room=user_id)

            if cv2.waitKey(1) &0xFF == ord('q'):
                break 

            if(flag==1):
                for i in range(20):
                    socketio.emit('request_frame', room=user_id)
                    ret, im = True,MAIN_FRAME[user_id]
                    im = np.frombuffer(im, np.uint8)
                    im = cv2.imdecode(im, cv2.IMREAD_COLOR)
                    im = cv2.flip(im, 1)
                    im = imutils.resize(im, width=720)
                    im=face_black(im)
                    cv2.rectangle(im, rect1, rect2, rect_color, -1)
                    cv2.putText(im,   question_real + " OK"   ,text_po,cv2.FONT_HERSHEY_COMPLEX,1,color,2)
                    success, frame = ret,im  # read the cam frame
                    ret, buffer = cv2.imencode('.jpg', frame)
                    base64_encoded = base64.b64encode(buffer)
                    frame = base64_encoded.decode("utf-8")
                    frame="data:image/jpeg;base64,"+frame
                    socketio.emit('response_from_server', frame, room=user_id)
                break


    socketio.emit('request_frame', room=user_id)
    ret, im = True,detection_image
    im = np.frombuffer(im, np.uint8)
    im = cv2.imdecode(im, cv2.IMREAD_COLOR)  
    im=face_black(im)
    success, frame = ret,im  # read the cam frame

    small_frame = cv2.resize(frame, (0, 0), fx=0.250, fy=0.250)
    name = "Unknown"
    face_locations = face_recognition.face_locations(small_frame)
    face_encodings = face_recognition.face_encodings(small_frame, face_locations)
    sol=[]

    try:
        df = pd.read_excel(path)
        logging.info("Excel Found with shape  %s",df.shape)
    except:
        df = pd.DataFrame(columns = ['Name', 'Encoding','Timestamp','Image_Name'])
        logging.info("No Excel Found ")

    for face_encoding in face_encodings:
        sol=[]
        final_name="Unknown"
        final_per =0
        for i in range(len(df)):
            name=df['Name'].loc[df.index[i]]
            x=df['Encoding'].loc[df.index[i]]
            # print(x,type(x))
            x=ast.literal_eval(x)
            x=bytes(x)
            encoding=np.frombuffer(x)
            match = face_recognition.compare_faces([encoding], face_encoding,tolerance=1-tolerance_of_model)
            distance = face_recognition.face_distance([encoding], face_encoding)
            x=100-round(distance[0]*100,1)
            if(match[0]):
                print(name,x)
            if (match[0] and x>final_per):
                final_per =x
                final_name=name
        print(final_name,final_per)    
        sol.append(final_name)
        logging.info("%s  %s",final_name,final_per) 

    persons.extend(sol)                           
    logging.info("%s  %s",sol,persons)
    cv2.destroyAllWindows()
    blank_image = np.zeros((540,720,3), np.uint8)
    im = imutils.resize(blank_image, width=720)

    if(successs==limit_questions):
        output="LIVENESS PASSED"
    else:
        output="LIVENESS FAILED"

    if(len(face_encodings)>1):
        cv2.putText(im,"There seem to be more than one person",(0,260),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
        socketio.emit('sound_instructions', " There seem to be more than one person ", room=user_id)
        # cv2.putText(im,"There seems to be a Technical problem",(0,260),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
        cv2.putText(im,"Kindly Restart",(200,300),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
        socketio.emit('sound_instructions', " Kindly Restart ", room=user_id)
        output="total persons in frame issue"
    elif(output=="LIVENESS FAILED" or len(persons)==0):                                                                                               
        cv2.putText(im,"LIVENESS FAILED",(200,260),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
        socketio.emit('sound_instructions', " Liveness Failed ", room=user_id)
    elif(len(list(set(persons)))==1 and persons[0]=="Unknown"):                                                                  
        cv2.putText(im,"Person Unknown",(160,260),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
        socketio.emit('sound_instructions', " Person Unknown ", room=user_id)
    elif(len(list(set(persons)))==1 and persons[0]!="Unknown"):     
        cv2.putText(im,"Welcome "+str(persons[0]),(160,260),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
        socketio.emit('sound_instructions', "Welcome "+str(persons[0]), room=user_id)
    else:
        cv2.putText(im,"There seems to be a Technical problem",(0,260),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
        socketio.emit('sound_instructions', "There seems to be a Technical problem", room=user_id)
        # cv2.putText(im,"There seem to be more than one person",(0,260),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
        # cv2.putText(im,"There seems to be a Technical problem",(0,260),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
        cv2.putText(im,"Kindly Restart",(200,300),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
        socketio.emit('sound_instructions', "Kindly Restart ", room=user_id)
        output="total persons in frame issue"
    logging.info("%s  %s",persons,output)
    logging.info("limit is  %s  , Everything runs successfully and the output is  :---  %s",limit_questions,output)

    frame=im
    ret, buffer = cv2.imencode('.jpg', frame)
    base64_encoded = base64.b64encode(buffer)
    frame = base64_encoded.decode("utf-8")
    frame="data:image/jpeg;base64,"+frame
    socketio.emit('response_from_server', frame, room=user_id)
    logging.info("TEST COMPLETED  %s",output)
    socketio.emit('test_complete', frame, room=user_id)


if __name__ == '__main__':

    # blank_image = np.zeros((540,720,3), np.uint8)
    # im = imutils.resize(blank_image, width=720)
    # frame=im
    # ret, buffer = cv2.imencode('.jpg', frame)
    # frame = buffer.tobytes()
    # MAIN_FRAME=frame

    MAIN_FRAME={}
    start=time.time()
    logging.info("Server Started in main function")
    socketio.run(app, host="0.0.0.0", port=remote_server_port)
    logging.info("Server ended in main function")
