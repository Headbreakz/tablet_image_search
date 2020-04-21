from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
from matplotlib import pyplot as plt
import io
from google.cloud import vision
from PIL import Image
import pandas as pd
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


from flask import Flask, escape, request
app = Flask(__name__)


# @app.route('/tablet_search',methods=['POST'])    
# def tablet_search():
#     body = request.json
    
#     img_url = body['action']['detailParaims']['url']['origin']
    
#     urllib.request.urlretrieve(img_url,'./image.jpg')
    
#     image = cv2.imread('./image.jpg')
#     # 분류를 위한 이미지 전처리를 수행합니다
#     image = cv2.resize(image, (280, 160))
#     image = image.astype("float") / 255.0
#     image = img_to_array(image)
#     image = np.expand_dims(image, axis=0)

#     # 학습된 네트워크와 `MultiLabelBinarizer`를 로드합니다
# #     print("[INFO] loading network...")
#     model = load_model("model.h5")
#     mlb = pickle.loads(open("labelbin", "rb").read())

#     # 이미지에 대한 분류를 수행한 후, 
#     # 확률이 가장 높은 두 개의 클래스 라벨을 찾습니다
# #     print("[INFO] classifying image...")
#     proba = model.predict(image)[0]
#     idxs = np.argsort(proba)[::-1][:2]
    
#     first_acc=mlb.classes_[idxs[0]]
#     second_acc=mlb.classes_[idxs[1]]
    
#     # 구글API
#     os.environ["GOOGLE_APPLICATION_CREDENTIALS"] ='C:/test/crucial-lyceum-274205-eadd5d760574.json'

#     client = vision.ImageAnnotatorClient()

#     with io.open(path, 'rb') as image_file:
#         content = image_file.read()

#     image = vision.types.Image(content=content)

#     response = client.text_detection(image=image)
#     texts = response.text_annotations

#     search_word=texts[1].description
    
    
#     df1=pd.read_excel('공공데이터개방_낱알식별목록_re.xlsx')
    
#     check1 = df1[df1['의약품제형'].str.contains(first_acc) | df1['색상앞'].str.contains(first_acc)]  
#     check2 = check1[check1['표시앞'].isin([search_word]) | check1['표시뒤'].isin([search_word])]
#     tablet_name =check2['품목명'].values
#     tablet_function=check2['분류명'].values
    
#     if not tablet_name:
#         check1 = df1[df1['의약품제형'].str.contains(second_acc) | df1['색상앞'].str.contains(second_acc)]  
#         check2 = check1[check1['표시앞'].isin([search_word]) | check1['표시뒤'].isin([search_word])]
#         tablet_name =check2['품목명'].values
#         tablet_function=check2['분류명'].values
        
#     return {
#         "version": "2.0",
#         "template": {
#             "outputs": [
#                 {
#                     "listCard": {
#                         "header": {
#                             "title": "카카오 i 디벨로퍼스를 소개합니다",
#                             "imageUrl": "http://k.kakaocdn.net/dn/xsBdT/btqqIzbK4Hc/F39JI8XNVDMP9jPvoVdxl1/2x1.jpg"
#                          },
#                         "items": [
#                             {
#                           "title": tablet_name[0],
#                           "description": tablet_funcion[0],
#                           "imageUrl": "http://k.kakaocdn.net/dn/APR96/btqqH7zLanY/kD5mIPX7TdD2NAxgP29cC0/1x1.jpg",
#                           "link": {
#                             "web": "https://namu.wiki/w/%EB%9D%BC%EC%9D%B4%EC%96%B8(%EC%B9%B4%EC%B9%B4%EC%98%A4%ED%94%84%EB%A0%8C%EC%A6%88)"
#                           }
#                         },
#                             {
#                               "title": tablet_name[1],
#                               "description":  tablet_funcion[1],
#                               "imageUrl": "http://k.kakaocdn.net/dn/N4Epz/btqqHCfF5II/a3kMRckYml1NLPEo7nqTmK/1x1.jpg",
#                               "link": {
#                                 "web": "https://namu.wiki/w/%EB%AC%B4%EC%A7%80(%EC%B9%B4%EC%B9%B4%EC%98%A4%ED%94%84%EB%A0%8C%EC%A6%88)"
#                               }
#                             },
#                             {
#                               "title":  tablet_name[2],
#                               "description": tablet_funcion[2] ,
#                               "imageUrl": "http://k.kakaocdn.net/dn/bE8AKO/btqqFHI6vDQ/mWZGNbLIOlTv3oVF1gzXKK/1x1.jpg",
#                               "link": {
#                                 "web": "https://namu.wiki/w/%EC%96%B4%ED%94%BC%EC%B9%98"
#                               }
#                             }
#                           ],
#                           "buttons": [
#                             {
#                               "label": "처음으로",
                                
#                               "action": "block",
#                               "blockId": "시작화면"
#                             }
#                           ]
#                         }
#                       }
#                     ]
#                   }
#                 }

    
    



if __name__ == '__main__':    
    app.run(host='127.0.0.1', port=5000, debug=True)

    
