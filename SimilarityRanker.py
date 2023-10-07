import os
import multiprocessing
import cv2
import numpy as np
from deepface import DeepFace
import shutil

def cropFaces(inputFolder, outputFolder):
    
    print("Crop function called...")
    for filename in os.listdir(inputFolder):
        cap2 = cv2.VideoCapture(inputFolder+"/"+filename)
        exit_flag=False
        while True:
            # Capture a frame from the video stream
            ret, frame = cap2.read()
            frame1 = frame.copy()
            
            if not ret:
                # If frame is not captured successfully, break the loop
                break

            # Convert the frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces in the grayscale image using the Haar Cascade classifier
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            # Loop through each face detected in the frame
            for (x, y, w, h) in faces:
                # Draw a rectangle around the face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # Calculate the percentage of the face that is visible
                total_area = w * h
                visible_area = 0

                # Loop through each pixel in the face region
                for i in range(x, x+w):
                    for j in range(y, y+h):
                        # If the pixel is part of the face region, increment the visible area
                        if i >= x and i < x+w and j >= y and j < y+h:
                            visible_area += 1

                # Calculate the confidence score for full facial detection
                confidence_score = visible_area / total_area * 100
                
                if confidence_score >= 95.0:
                    cv2.imwrite(outputFolder+"/"+filename.split('.')[0]+'.jpg', frame1)
                    #print(filename, confidence_score)
                    exit_flag = True
                    break
            if exit_flag==True:
                break
        # Release the video capture device and close all windows
        cap2.release()
    return 1


def rankSimilar(file, outputFolder, rankFolder):
    file = outputFolder+"/"+file
    rankList = []
    
    for file2 in os.listdir(outputFolder):
        print(file.split('/')[1], file2)
        try:
            result = DeepFace.verify(file, outputFolder+"/"+file2)
        except:
            print("Face not detected properly for: ", file.split('/')[1], file2)
            
        print("Similarity score: ", result["distance"], "\n")
        rankList.append((file2, result["distance"]))
        #print(result, "\n")
    
    #print(rankList, "\n")
    #print(sorted(rankList, key=lambda x: x[1], reverse=True))
    rankList = sorted(rankList, key=lambda x: x[1])
    print(rankList, "\n")
    
    rank = 1
    for (x,y) in rankList:
        source_path = outputFolder+"/"+x
        destination_path = rankFolder
        newfilename = destination_path+"/"+str(rank)+"_"+x
        
        shutil.copy(source_path, newfilename)
        rank+=1

def inputVideo(filename, inputFolder, outputFolder, file, rankFolder):
    filename = inputFolder+"/"+filename
    cap = cv2.VideoCapture(filename)
    
    cropFaces(inputFolder, outputFolder)
    #rankSimilar(file, outputFolder, rankFolder)
    cntr = 0
    cntrFlag = 0
    cntrFlag2 = 0
    
    while True:
        # Capture a frame from the video stream
        ret, frame = cap.read()
        frame1 = frame.copy()

        if not ret:
            # If frame is not captured successfully, break the loop
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale image using the Haar Cascade classifier
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Loop through each face detected in the frame
        for (x, y, w, h) in faces:
            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Calculate the percentage of the face that is visible
            total_area = w * h
            visible_area = 0

            # Loop through each pixel in the face region
            for i in range(x, x+w):
                for j in range(y, y+h):
                    # If the pixel is part of the face region, increment the visible area
                    if i >= x and i < x+w and j >= y and j < y+h:
                        visible_area += 1

            # Calculate the confidence score for full facial detection
            confidence_score = visible_area / total_area * 100
            
            if confidence_score == 100.0:
                cv2.imwrite(outputFolder+"/"+filename.split('.')[0]+'.jpg', frame1)

                if cntr<=3 and cntrFlag==0:
                    cntrFlag=1

                # if cntr<=3 and cntrFlag2==0:
                #     cntrFlag2=1
                #print(filename)
                #with concurrent.futures.ThreadPoolExecutor() as executor:
                    #results = [executor.submit(cropFaces, inputFolder, outputFolder)]
                    #output = [r.result() for r in results]
                    #cropFaces(inputFolder, outputFolder)
                
            # Display the confidence score on the frame
            cv2.putText(frame, f"Confidence Score: {confidence_score:.2f}%", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow('Frame', frame)
        
        cntr+=1
        if cntrFlag==1:
            cntrFlag=0
            #cropFaces(inputFolder, outputFolder)
            rankSimilar(file, outputFolder, rankFolder)

        # if cntrFlag2==1:
        #     cntrFlag2=0
        #     rankSimilar(file, outputFolder, rankFolder)

        # Press 'q' to quit
        # if cv2.waitKey(1) == ord('q'):
        #     break
        
        endkey = cv2.waitKey(1)
        if endkey == ord('q') or endkey == 27: # press q or Esc to exit
            break

    # Release the video capture device and close all windows
    cap.release()
    cv2.destroyAllWindows()

# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

if __name__ == '__main__':
    #p = multiprocessing.Process(target=cropFaces)
    #p.start()
    
    filename = "Recording17.mp4"
    file = "Recording17.jpg"
    
    inputFolder = "Eloclips"
    outputFolder = "CropFaces"
    rankFolder = "RankSimilarity"
    
    p = multiprocessing.Process(target=cropFaces, args=(inputFolder, outputFolder))
    p.start()
    p1 = multiprocessing.Process(target=rankSimilar, args=(file, outputFolder, rankFolder))
    p1.start()
    inputVideo(filename, inputFolder, outputFolder, file, rankFolder)
    p.join()
    p1.join()