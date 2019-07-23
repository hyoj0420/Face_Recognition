import cv2
import numpy as np
from PIL import Image
import os
# Path for face image database
PATH= r"/home/pi/Project/Trainingimage/"
recognizer = cv2.face.LBPHFaceRecognizer_create()

# function to get the images and label data
def getImagesAndLabels(PATH):
    file_list = [file for file in os.listdir(PATH) if not (file.endswith(".jpg") or file.endswith(".yml"))]
      
    faceSamples=[]
    ids = []
    for filePath in file_list :
        img_list = [file for file in os.listdir(PATH+filePath)]
        id = int(filePath.split("_")[1])
        
        for imagePath in img_list:
            PIL_img = Image.open(PATH+filePath+"/"+imagePath)
            img_numpy = np.array(PIL_img,'uint8')
            faceSamples.append(img_numpy)
            ids.append(id)
    return faceSamples,ids
    
print ("Training faces. It will take a few seconds. Wait ...")
faces,ids = getImagesAndLabels(PATH)
recognizer.train(faces, np.array(ids))
# Save the model into trainer/trainer.yml
recognizer.write(r'/home/pi/Project/Trainingimage/trainer.yml') # recognizer.save() worked on Mac, but not on Pi
# Print the numer of faces trained and end program
print("{0} faces trained. Exiting Program".format(len(np.unique(ids))))
