from ultralytics import YOLO
import math

confidence = 0.8

# Initialize video capture and set resolution
# cap = cv2.VideoCapture(0)
# cap.set(3, 1280)
# cap.set(4, 720)

model = YOLO(r'D:\face\AntiSpoofingDetector\models\n_version_1_30.pt')
classNames = ["fake", "real"]

def viewFakeandReal(img):
    try:
        results = model(img, stream=False, verbose=False)

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
                       return str(name)
                        
                    else:  
                         return str(name)

    except Exception as e:
        return str(e)


# viewFakeandReal()
