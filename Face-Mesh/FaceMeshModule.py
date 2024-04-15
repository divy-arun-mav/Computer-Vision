import cv2
import mediapipe as mp
import time

class FaceMeshDetector():
    def __init__(self, staticMode=False, maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        # Ensure that the parameters passed here match the expected types
        self.faceMesh = self.mpFaceMesh.FaceMesh(max_num_faces=self.maxFaces, min_detection_confidence=self.minDetectionCon, min_tracking_confidence=self.minTrackCon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)


    def findFaceMesh(self,img,draw=True):
        self.imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces=[]
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw: 
                    self.mpDraw.draw_landmarks(img,faceLms,self.mpFaceMesh.FACEMESH_TESSELATION,self.drawSpec,self.drawSpec)

                face=[]
                for id,lm in enumerate(faceLms.landmark):
                    print(lm)
                    ih,iw,ic = img.shape
                    x,y =int(lm.x*iw),int(lm.y*ih)
                    # cv2.putText(img,str(id),(x,y),cv2.FONT_HERSHEY_PLAIN,0.7,(0,255,0),1)
                    print(id,x,y)
                    face.append([x,y])

                faces.append(face)
        return img,faces


def main():

    cap = cv2.VideoCapture("Videos/1.mp4")
    pTime = 0
    detector = FaceMeshDetector()

    frame_skip = 2 # Skip every 2 frames
    frame_count = 0

    while True:
        success, img = cap.read()
        frame_count += 1
        if frame_count % frame_skip == 0: # Process only every nth frame
            img, faces = detector.findFaceMesh(img, True)
            if len(faces) != 0:
                print(len(faces))
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(img, f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
            cv2.imshow("Image", img)
        cv2.waitKey(1)




if __name__ == "__main__":
    main()