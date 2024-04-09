import cv2
import mediapipe as mp
import time


class FaceDetector():
    def __init__(self,minDetectionCon=0.5):
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(minDetectionCon)

    def findFaces(self,img,draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)  
        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                    int(bboxC.width * iw), int(bboxC.height * ih)
                bboxs.append([id,bbox,detection.score])
                if draw:
                    img = self.fancyDraw(img,bbox)

                    cv2.putText(img, f'{int(detection.score[0] * 100)}%', (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2,
                            (255, 0, 255), 2)
        return img,bboxs
    
    def fancyDraw(self,img,bbox,l=30,t=5):
        x,y,w,h = bbox
        x1,y1=x+w,y+h

        cv2.rectangle(img, bbox, (255, 0, 255), 1)
        # top left x,y
        cv2.line(img, (int(x), int(y)), (int(x+l), int(y)), (255,0,255), int(t))  
        cv2.line(img, (int(x), int(y)), (int(x), int(y+l)), (255,0,255), int(t))
        # top right x1,y
        cv2.line(img, (int(x1), int(y)), (int(x1-l), int(y)), (255,0,255), int(t))  
        cv2.line(img, (int(x1), int(y)), (int(x1), int(y+l)), (255,0,255), int(t))
        # bottom left x,y1
        cv2.line(img, (int(x), int(y1)), (int(x+l), int(y1)), (255,0,255), int(t))  
        cv2.line(img, (int(x), int(y1)), (int(x), int(y1-l)), (255,0,255), int(t))
        # bottom right x1,y1
        cv2.line(img, (int(x1), int(y1)), (int(x1-l), int(y1)), (255,0,255), int(t))  
        cv2.line(img, (int(x1), int(y1)), (int(x1), int(y1-l)), (255,0,255), int(t))
        return img






def main():
    cap = cv2.VideoCapture("Videos/1.mp4")
    pTime = 0
    detector = FaceDetector()
    while True:
        success, img = cap.read()
        img,bboxs = detector.findFaces(img)
        print(bboxs)

        if not success:
            break

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
        cv2.imshow("Image", img)
        cv2.waitKey(1)

        # cap.release()
        # cv2.destroyAllWindows()



if __name__ == "__main__" :
    main()