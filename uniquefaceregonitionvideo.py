import face_recognition
import os
import cv2
import pickle
import time
knownfacedir="knownvface"
#unknownfacedir="unknownface"
video=cv2.VideoCapture("4.mp4")
#video=cv2.VideoCapture(0)
tolerance=0.51
framethickness=3
fontthickness=2
MODEL="cnn"
print("Loading known faces")
knownfaces=[]
knownnames=[]

for name in os.listdir(knownfacedir):
    for filename in os.listdir(f"{knownfacedir}/{name}"):
        #image=face_recognition.load_image_file(f"{knownfacedir}/{name}/{filename}")
        #encoding=face_recognition.face_encodings(image)[0]
        encoding=pickle.loads(open(f"{name}/{filename}","rb"))
        knownfaces.append(encoding)
        knownnames.append(int(name))
        print("1over")
if(len(knownnames)>0):
    nextid=max(knownnames)+1
else:
    nextid=0
print("Processign unknown faces")
while True:
    #print(filename)
    print("in for")
    #image=face_recognition.load_image_file(f"{unknownfacedir}/{filename}")
    ret, image=video.read()
    print("Processign unknown loading")
    locations=face_recognition.face_locations(image,model=MODEL)
    print("Processign unknown location")
    encodings=face_recognition.face_encodings(image,locations)
    print("encoding over")
    #image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    
    for face_encoding, face_location in zip(encodings,locations):
        result=face_recognition.compare_faces(knownfaces,face_encoding,tolerance)
        print("finding match")
        match=None 
        if True in result:
            match=knownnames[result.index(True)]
            print(f"Match found: {match}")
            
            topleft=(face_location[3],face_location[0])
            bottomright=(face_location[1],face_location[2])
            color=[0,255,0]
            cv2.rectangle(image,topleft,bottomright,color,framethickness)
            
            topleft=(face_location[3],face_location[2])
            bottomright=(face_location[1],face_location[2]+22)
            cv2.rectangle(image,topleft,bottomright,color,cv2.FILLED)
            cv2.putText(image,match,(face_location[3]+10,face_location[2]+15),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,0),fontthickness)
       
        else:
            print("Match NOT found")
            match=str(nextid)
            nextid+=1
            knownnames.append(match)
            knownfaces.append(face_encoding)
            os.mkdir(f"{knownfacedir}/{match}")
            pickle.dump(face_encoding,open(f"{knownfacedir}/{match}/{match}-{int(time.time())}.pkl","wb"))
            topleft=(face_location[3],face_location[0])
            bottomright=(face_location[1],face_location[2])
            color=[0,255,0]
            cv2.rectangle(image,topleft,bottomright,color,framethickness)
            
            topleft=(face_location[3],face_location[2])
            bottomright=(face_location[1],face_location[2]+22)
            cv2.rectangle(image,topleft,bottomright,color,cv2.FILLED)
            cv2.putText(image,match,(face_location[3]+10,face_location[2]+15),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,0),fontthickness)
    cv2.imshow("",image)
    if cv2.waitKey(1)&0xFF==ord("q"):
        break

    #cv2.waitKey(10000)
    #cv2.destroyWindow(filename)
            
    #video.release()
# Destroy all the windows
    #cv2.destroyAllWindows()
    
            
        
    
        
        

