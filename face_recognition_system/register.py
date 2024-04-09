
import numpy as np
from datetime import datetime
import cv2
import base64
import pandas as pd
import os
import csv
import pandas as pd
import json
from deepface import DeepFace
# import face_recognition
from ultralytics import YOLO
import math

confidence = 0.7

# import tensorflow as tf

# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

def bs64_to_frame(bs64):
        if bs64:
            decoded_bytes = base64.b64decode(bs64)
            decoded_image = np.frombuffer(decoded_bytes, dtype=np.uint8)
            decoded_image = cv2.imdecode(decoded_image, flags=cv2.IMREAD_COLOR)
            return decoded_image
        else:
            return 'Unable to process image'

userImagePath=''
userDetails={}
checkBlick=False

def respons(status,meassge):
    return {
        "status":status,
        "meassge":meassge,
        "image": image_to_base64(userImagePath),
        "userDetails":userDetails
        }

def image_to_base64(image_path):
    if image_path:
        with open(image_path, "rb") as img_file:
            image_data = img_file.read()
            base64_encoded = base64.b64encode(image_data).decode("utf-8")
        return base64_encoded
    else:
        return ""

def register(frame,id,name,dept):
    # global userImagePath
    # global userDetails
    print("Start Time ",datetime.now().strftime('%H-%M-%S'))
    known_ids=[]
    check2=[]
    count=False
    excelDataSet=r"D:\face\face_recognition_system\Employee_details.csv"
    path = r"D:\face\face_recognition_system\dataset"
   
    l1 = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
    if frame is not None:
        # _=DeepFace.build_model("Facenet")
        for i in l1:
            known_ids.append(os.path.splitext(os.path.basename(i))[0])
        _=DeepFace.build_model("Facenet")
        check=DeepFace.find(frame,path,model_name='Facenet',normalization='Facenet')
            
        if check[0]['identity'].values:
            return respons("ok","'your data is already in dd!")

        
        else:
            if id in known_ids:
                return respons("error","your data is already in dd!!!!!")
            else:    
                count=True
                cv2.imwrite(path+f'/{id}.jpg',frame)  
        if not os.path.exists(excelDataSet):
            with open(excelDataSet,'w') as ff:
                csveriting = csv.writer(ff)
                csveriting.writerow(['EmployeeID','Emp_name','Department'])
                csveriting.writerow([id,name,dept])
        else:
            with open(excelDataSet,'a') as ff:
                csvreading=pd.read_csv(excelDataSet)
                csveriting = csv.writer(ff)
                for i in range(len(csvreading)):
                    if csvreading['EmployeeID'].values[i]==id and count==True:
                        return respons("error","your details already in d")
                    else:
                        csveriting.writerow([id,name,dept])
                        break
        df = pd.read_csv(excelDataSet)
        df.to_csv(excelDataSet, index=False)
        # print(df)
    else:
        # return 'Unable to process image'
        return respons("error","'Unable to process image")
 
    return respons("ok","'your details Save")





model = YOLO(r'D:\face\AntiSpoofingDetector\models\n_version_1_30.pt')
classNames = ["fake", "real"]

def viewFakeandReal(img):
    try:
        pt("check start 1")    
        results = model(img, stream=False, verbose=False)
        pt("check start 2")    

        for r in results:
            boxes = r.boxes
            for box in boxes:
                # x1, y1, x2, y2 = box.xyxy[0]
                # x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # w, h = x2 - x1, y2 - y1

                # Extract confidence score and class index
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = box.cls[0]
                # Get class name based on class index
                name = classNames[int(cls)].upper()
                if conf > confidence:
                    if name == "REAL":
                       pt("check start 3", name)    
                       return True
                    else:  
                         pt("check start 4", name)    
                         return False

    except Exception as e:
         print(e)
         return False




def getFace(frame):
    # face_locations=face_recognition.face_locations(frame)
    # print('reach 10',face_locations)
    # print('reach 1')

    try:
            blink=viewFakeandReal(frame)
            pt("check start 5")    
            # return blink
            if blink:
                pt("check start 6")    
                global userImagePath
                global userDetails
                known_ids=[]
                # check2=[]
                # count=False
                excelDataSet=r"D:\face\face_recognition_system\Employee_details.csv"
                path =r"D:\face\face_recognition_system\dataset"


                l1 = [os.path.join(path, f) for f in os.listdir(path) if not f.endswith('.pkl')]
        
                if frame is not None:
                    for i in l1:
                        known_ids.append(os.path.splitext(os.path.basename(i))[0])
                    _=DeepFace.build_model("Facenet")
                    # check=DeepFace.find(frame,path,model_name='Facenet',normalization='Facenet',detector_backend='yolov8',enforce_detection=True,threshold=0.40,silent=True)
                    check=DeepFace.find(frame,path,model_name='Facenet',distance_metric="euclidean",normalization='Facenet',detector_backend='yolov8',threshold=9,silent=True)
                        
                    # print("check[0]['identity'].values", check[0]['identity'].values)
                    if len(check[0]['identity'].values)>0:
                        face_id=os.path.basename(os.path.dirname(check[0]['identity'].values[0]))
                        # face_id=os.path.splitext(os.path.basename(check[0]['identity'].values[0]))[0]
                        print(face_id)
                        
                        userImagePath =check[0]['identity'].values[0]
                        print(userImagePath)
                        Attendance_auto(face_id)
                    
                    
                        emp_df=pd.read_csv(excelDataSet)
                        check3=emp_df[face_id==emp_df['EmployeeID'].astype(str)]
                        userDetails=check3.to_dict(orient='records')[0]
                        return respons("ok","your data is already in dd!")
                        # return "ok"
                    else:
                        return respons("error","not in database@@@")
                    
                else:
                    # return 'Unable to process image'
                    return respons("error","Unable to process image!!!!")
                

                
            
                # l1 = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
                
                # if frame is not None:
                #     pt("check start 7")    
                #     for i in l1:
                #         known_ids.append(os.path.splitext(os.path.basename(i))[0])
                #     _=DeepFace.build_model("Facenet")
                #     # enforce_detection=False,
                #     pt("check start")    
                #     check=DeepFace.find(frame,path,model_name='Facenet',distance_metric="euclidean",normalization='Facenet',detector_backend='yolov8',threshold=9,silent=True)
                #     pt("check",check)    
                #     # print("check[0]['identity'].values", check[0]['identity'].values)
                #     if check[0]['identity'].values:
                #         face_id=os.path.splitext(os.path.basename(check[0]['identity'].values[0]))[0]
                #         pt(face_id)
                        
                #         userImagePath =check[0]['identity'].values[0]
                #         pt(userImagePath)
                    
                       
                #         emp_df=pd.read_csv(excelDataSet)
                #         check3=emp_df[face_id==emp_df['EmployeeID'].astype(str)]
                #         userDetails=check3.to_dict(orient='records')[0]
                #         Attendance_auto(face_id)
                #         return respons("ok","'your data is already in dd!")
                #         # return "ok"
                #     else:
                #         return respons("notin","Your not a ragister.")
                # else:
                #     # return 'Unable to process image'
                #     return respons("error","'Unable to process image!!!!")
           
           
           
           
           
            else:
             return respons("error",'')

    except Exception as e:
        return respons("error",str(e))


def Attendance_in(id):


    excelDataSet=r"D:\face\face_recognition_system\Employee_details.csv"
    path =r"D:\face\face_recognition_system\dataset"
    excelAttendanceData = r"D:\face\face_recognition_system\Attendance.csv"
    known_ids = []
    data = []
    # check2 = []
    face_ids = []
    face_id = 'Unknown'
    l1 = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
    
    # frame = bs64_to_frame(bs64)
    
    for i in l1:
            known_ids.append(os.path.splitext(os.path.basename(i))[0])
            data.append(os.path.splitext(os.path.basename(i))[0])
    # face_id = known_ids
    if id:
        face_id=id
    face_ids.append(face_id)
    emp_df = pd.read_csv(excelDataSet)
    if os.path.exists(excelAttendanceData): 
        csvreading = pd.read_csv(excelAttendanceData)
    else:
         csvreading=pd.DataFrame()
    print(face_ids)
    for face_id in face_ids:
        print(face_id)
        if face_id in known_ids and face_id in data:
            check3 = emp_df[emp_df['EmployeeID'] == face_id]
            
            if not check3.empty:
                today_date = datetime.now().strftime('%d-%m-%Y')
                if csvreading.empty:
                    existing_entry=pd.DataFrame()
                else:
                    existing_entry = csvreading[(csvreading['EmployeeID'] == face_id) & (csvreading['Date'] == today_date)]
                
                if existing_entry.empty:
                    new_df = check3.copy()
                    intime = datetime.now().strftime('%H:%M:%S')
                    new_df['Date'] = today_date
                    new_df['Intime'] = intime
                    new_df['Out_time'] = np.nan
                    
                    if not os.path.exists(excelAttendanceData):
                        new_df.to_csv(excelAttendanceData, mode='a', index=False)
                    else:
                        new_df.to_csv(excelAttendanceData, mode='a', header=False, index=False)
                    
                    # return f"Attendance marked for {face_id} at {intime}"
                    print( f"Attendance marked for {face_id} at {intime}")
                else:
                    # pass
                    print( f"Attendance already marked for {face_id} today")
        else:
            
            print("Employee ID not recognized") 

def Attendance_out(id):
    excelDataSet=r"D:\face\face_recognition_system\Employee_details.csv"
    path =r"D:\face\face_recognition_system\dataset"
    excelAttendanceData = r"D:\face\face_recognition_system\Attendance.csv"
    known_ids = []
    data = []
    check2 = []
    face_ids = []
    face_id = 'Unknown'
    l1 = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
    for i in l1:
        known_ids.append(os.path.splitext(os.path.basename(i))[0])
        data.append(os.path.splitext(os.path.basename(i))[0])
        
    if id:
        face_id = id
    face_ids.append(face_id)
        
        # face_ids.append(face_id)
            
    emp_df = pd.read_csv(excelDataSet)
    # csvreading = pd.read_csv(excelAttendanceData)
    print(face_ids)
    for face_id in face_ids:
        print(face_id)
        if face_id in known_ids and face_id in data:
            
            
            check3 = emp_df[emp_df['EmployeeID'] == face_id]
            data.remove(face_id)
            # emp_df = pd.read_csv("Employee_details.csv")
            # check = emp_df[face_ids[0] == emp_df['EmployeeID'].astype(str)]
            if check3 is not None:
                new_df = pd.DataFrame(check3)
                out_time = datetime.now().strftime('%H:%M:%S')
                new_df['Out_time'] = [out_time]
                existing_attendance_df = pd.read_csv(excelAttendanceData)
                existing_attendance_df.loc[(existing_attendance_df['EmployeeID'] == new_df['EmployeeID'].values[0]) &
                                            (existing_attendance_df['Date'] == datetime.now().strftime('%d-%m-%Y')), 'Out_time'] = out_time
                existing_attendance_df.to_csv(excelAttendanceData, index=False)
            # else:
            #     print(f"Attendance already marked for {face_id}!")
        else:
            return f"{face_id} is not in the database!"

def Attendance_auto(id):
    excelDataSet=r"D:\face\face_recognition_system\Employee_details.csv"
    path =r"D:\face\face_recognition_system\dataset"
    excelAttendanceData = r"D:\face\face_recognition_system\Attendance.csv"
    known_ids = []
    data = []
    face_ids = []
    face_id = 'Unknown'
    l1 = [os.path.join(path, f) for f in os.listdir(path) if not f.endswith('.pkl')]
    for i in l1:
        known_ids.append(os.path.splitext(os.path.basename(i))[0])
        data.append(os.path.splitext(os.path.basename(i))[0])
    if id:
        face_id = id
    face_ids.append(face_id)
    emp_df = pd.read_csv(excelDataSet)
    if os.path.exists(excelAttendanceData): 
        csvreading = pd.read_csv(excelAttendanceData)
    else:
         csvreading=pd.DataFrame()
    print(face_ids)
    for face_id in face_ids:
        print(face_id)
        if face_id in known_ids and face_id in data:
            check3 = emp_df[emp_df['EmployeeID'] == face_id]
            
            if not check3.empty:
                today_date = datetime.now().strftime('%d-%m-%Y')
                if csvreading.empty:
                    existing_entry=pd.DataFrame()
                else:
                    existing_entry = csvreading[(csvreading['EmployeeID'] == face_id) & (csvreading['Date'] == today_date)]
                
                if existing_entry.empty:
                    new_df = check3.copy()
                    intime = datetime.now().strftime('%H:%M:%S')
                    new_df['Date'] = today_date
                    new_df['Intime'] = intime
                    new_df['Out_time'] = np.nan
                    
                    if not os.path.exists(excelAttendanceData):
                        new_df.to_csv(excelAttendanceData, mode='a', index=False)
                    else:
                        new_df.to_csv(excelAttendanceData, mode='a', header=False, index=False)
                    
                    print(f"Attendance marked for {face_id} at {intime}")
                elif not existing_entry.empty:
                    new_df = pd.DataFrame(check3)
                    out_time = datetime.now().strftime('%H:%M:%S')
                    new_df['Out_time'] = [out_time]
                    existing_attendance_df = pd.read_csv(excelAttendanceData)
                    existing_attendance_df.loc[(existing_attendance_df['EmployeeID'] == new_df['EmployeeID'].values[0]) &
                                                (existing_attendance_df['Date'] == datetime.now().strftime('%d-%m-%Y')), 'Out_time'] = out_time
                    existing_attendance_df.to_csv(excelAttendanceData, index=False)
                else:
                    print(f"Attendance already marked for {face_id} today")
        else:
            print("Employee ID not recognized")








def pt(value,data=""):
    pass
    # print(value,data)
# cap=cv2.VideoCapture(0)
# while True:
#     ret,frame=cap.read()
#     if ret:
#         cv2.imshow('frame', frame)
#         # face_locations=face_recognition.face_locations(frame)
#         # print('reach 9',face_locations)
#         print(getFace(frame))
        
        # if cv2.waitKey(1) & 0xFF == ord('q'): 
        #     break
# # "E1129","Md","mobile"
# cap.release()
# print("end Time ",datetime.now().strftime('%H-%M-%S'))
# cv2.destroyAllWindows()
# def register(faec_base64,id,name,dept):
