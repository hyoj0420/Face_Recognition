import os
import cv2
import face_recognition
import numpy as np
from sklearn.cluster import DBSCAN

def encode(PATH) :
    file_list = os.listdir(PATH)
    #디렉토리 내 이미지 파일 목록 추출
    img_list = [file for file in file_list if file.endswith(".jpg")]
    img_list.sort()
    faces = []

    #이미지 파일에 대하여 데이터 encoding
    for i in  img_list :
        img = cv2.imread(PATH+i)
        boxes = face_recognition.face_locations(img, model="hog")
        encodings = face_recognition.face_encodings(img, boxes)
        face = [{"imgName" : i, "box" : box, "encoding" : encoding} for (box, encoding) in zip(boxes, encodings)]
        faces.extend(face)

    return faces

def cluster(PATH, faces) :
    feature = [face["encoding"] for face in faces]
    #DBSCAN 통해 분류
    clt = DBSCAN(metric = "euclidean")
    clt.fit(feature)

    #분류된 얼굴에 대한 label
    label_ids = np.unique(clt.labels_)

    for label_id in label_ids :
        #label_id에 대한 디렉토리 생성
        dir_name = "SB_%d" % label_id
        os.mkdir(PATH+dir_name)

        indexes = np.where(clt.labels_ == label_id)[0]
        #label_id에 해당하는 인덱스를 디렉토리로 옮긴다
        for i in indexes:
            imageName= faces[i]["imgName"]
            os.rename(PATH+imageName, PATH+"SB_"+str(label_id)+"/"+imageName)

PATH = r"/home/pi/Project/Trainingimage/"
faces = encode(PATH)
cluster(PATH, faces)
print("Finish!")
