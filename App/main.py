#!/usr/bin/env python
# coding: utf-8

# In[ ]:




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
import json
from flask import jsonify
import urllib
from urllib.request import Request, urlopen
import json



from flask import Flask, escape, request
app = Flask(__name__)


@app.route('/tablet', methods=['POST'])
def tablet():
    req=request.get_json(force = True)
    print(json.dumps(req, indent=4,ensure_ascii=False))
       
    res ={'fulfillmentText':"ggggggggg"}
    
    return jsonify(res)


@app.route('/sample', methods=['POST'])
def sample():
    return {
    "version": "2.0",
    "template": {
        "outputs": [
            {
                "simpleText": {
                    "text": "aaaaa"
                }
            }
        ]
    }
}


@app.route('/tablet_search', methods=['POST'])
def tablet_search():
    
    body = request.json
    img_url = body['action']['detailParams']['photo']['origin']
    img_url = img_url.replace(')','(').replace(',','(').split('(')[1]
    urllib.request.urlretrieve(img_url, './image.jpg')
    
    img = cv2.imread('./image.jpg')
#     분류를 위한 이미지 전처리를 수행합니다
    image = cv2.resize(img, (280, 160))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    

     # 학습된 네트워크와 `MultiLabelBinarizer`를 로드합니다
    # print("[INFO] loading network...")
    model = load_model("model.h5")
    mlb = pickle.loads(open("labelbin", "rb").read())

    # 이미지에 대한 분류를 수행한 후, 
    # 확률이 가장 높은 네 개의 클래스 라벨을 찾습니다
    #print("[INFO] classifying image...")
    proba = model.predict(image)[0]
    idxs = np.argsort(proba)[::-1][:4]
    
  
    
    tablet_shape=[]
    tablet_color=[]

    tablet_shape_labels=['a hemicyclea semicircle','circle','diamond','ellipse','hexagon','octagon','pentagon',
    'rectangle','tetragon','triangle']
    tablet_color_labels=['black','blue','blue, light','blue, transparency','bluish green','bluish green, transparency','brown',
    ' brown, transparency','dark blue','dark blue','transparency','gray','green','green, transparency',
    'orange','orange, transparency','pink','pink, deep','pink, light','pink, transparency','purple',
    'purple, transparency','red','red, transparency','transparency','white','white, blue','white, brown',
    'white, green','white, red','white, transparency','white, yellow','wine','wine, transparency','yellow',
    'yellow, transparency','yellowish green','yellowish green, transparency']
    
    for idx in range(4) :
        if mlb.classes_[idxs[idx]] in tablet_shape_labels :
            tablet_shape.append(mlb.classes_[idxs[idx]])

        elif mlb.classes_[idxs[idx]] in tablet_color_labels :
            tablet_color.append(mlb.classes_[idxs[idx]])
          
    print(tablet_shape[0])
    print(tablet_color[0])
    
# #     구글API
#     os.environ["GOOGLE_APPLICATION_CREDENTIALS"] ='./crucial-lyceum-274205-eadd5d760574.json'
# #     print(os.environ['GOOGLE_APPLICATION_CREDENTIALS'])

#     client = vision.ImageAnnotatorClient()
    

#     with io.open('./image.jpg', 'rb') as image_file:
#         content = image_file.read()

#     image = vision.types.Image(content=content)

#     response = client.text_detection(image=image)
#     texts = response.text_annotations

#     search_word=texts[1].description
    
    
#     df1=pd.read_excel("test.xlsx")
    
  
    
#     check1 = df1[(df1['의약품제형']== tablet_shape[0]) & (df1['색상앞']==tablet_color[0])] 
#     check2 = check1[(check1['표시앞']==search_word) | (check1['표시뒤']==search_word)]
    
#     check2.to_json('test.json', orient='table')

#     with open('test.json') as json_file:
#         json_data = json.load(json_file)
    
#     tablet_name = json_data["data"][0]["품목명"]
#     tablet_function = json_data["data"][0]["분류명"]
#     print(tablet_name)
    res = {
   "version": "2.0",
   "template": {
      "outputs": [
         {
            "basicCard": {
               "description": "이 알약은 "+str(tablet_shape[0])+","+str(tablet_color[0])+"(으)로 판단됩니다",
               "buttons": [
                  {
                     "action": "block",
                     "label": "결과 보기",
                     "blockId":"5e9fe75f1f5af500010671f6",  
                     "messageText": "이 알약은 "+str(tablet_shape[0])+","+str(tablet_color[0])+"(으)로 판단됩니다",
                      "extra": {
                        "tablet_shape": "a",
                        "a": str(tablet_shape[0]) ,
                        "tablet_color": "b",
                        "b": str(tablet_color[0]) ,
                                              
                      }
                }
             ]
            }
         }
      ]
   }
}
    
    return jsonify(res)
    

@app.route('/tablet_result', methods=['POST'])
def tablet_result():
    body = request.json
    tablet_color = body['action']['clientExtra']['b']
    print(tablet_color)
    tablet_shape = body['action']['clientExtra']['a']
    print(tablet_shape)

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] ='./crucial-lyceum-274205-eadd5d760574.json'
#     print(os.environ['GOOGLE_APPLICATION_CREDENTIALS'])

    client = vision.ImageAnnotatorClient()
    

    with io.open('./image.jpg', 'rb') as image_file:
        content = image_file.read()

    image = vision.types.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations

    search_word=texts[1].description
    
    
    
    df1=pd.read_csv('./공공데이터개방_낱알식별목록_re.csv',encoding = 'cp949')
    
  
    
    check1 = df1[(df1['의약품제형']== tablet_shape) & (df1['색상앞']==tablet_color)] 
    check2 = check1[(check1['표시앞']==search_word) | (check1['표시뒤']==search_word)]
    
    check2.to_json('test.json', orient='table')

    with open('test.json') as json_file:
        json_data = json.load(json_file)
    
#     tablet_name = json_data["data"][0]["품목명"]
#     tablet_function = json_data["data"][0]["분류명"]
#     print(tablet_name)
    if len(check2) == 0 :
        res = {
              "version": "2.0",
              "template": {
                "outputs": [
                  {
                    "basicCard": {
                      "title": "알약 결과",
                      "description": "총"+str(len(check1))+"개의 알약을 찾았습니다. 더욱 자세한 사진을 사용하여 다시 하기를 추천합니다.",
                      "thumbnail": {
                        "imageUrl": "http://k.kakaocdn.net/dn/gVmuK/btqDBr6UaLe/IqwiJyjbktWowLNmpX5gd1/img_xl.jpg"
                      },          
                      "buttons": [
                        {
                          "action": "block",
                          "label": "다시 하기",
                          "blockId": "5e9e5bd463df8a0001ce07a9"
                        }            
                      ]
                    }
                  }
                ]
              }
            }
        

    elif len(check2) > 3 :
        res = {
              "version": "2.0",
              "template": {
                "outputs": [
                  {
                    "basicCard": {
                      "title": "알약 결과",
                      "description": "총"+str(len(check2))+"개의 알약을 찾았습니다. 더욱 자세한 사진을 사용하여 다시 하기를 추천합니다.",
                      "thumbnail": {
                        "imageUrl": "http://k.kakaocdn.net/dn/gVmuK/btqDBr6UaLe/IqwiJyjbktWowLNmpX5gd1/img_xl.jpg"
                      },          
                      "buttons": [
                        {
                          "action": "block",
                          "label": "다시 하기",
                          "blockId": "5e9e5bd463df8a0001ce07a9"
                        }            
                      ]
                    }
                  }
                ]
              }
            }
        

    elif len(check2) == 3 :
        res = {
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                        "listCard": {
                            "header": {
                                "title": "다음과 같은 알약으로 판단됩니다.",
                                "imageUrl": "http://k.kakaocdn.net/dn/w9sXr/btqDFcuEEC8/REi0gyPtobK4O32pu4awy1/img_l.jpg"
                             },
                            "items": [
                                {
                              "title": json_data["data"][0]["품목명"],
                              "description": json_data["data"][0]["분류명"],
                              "imageUrl": "http://k.kakaocdn.net/dn/APR96/btqqH7zLanY/kD5mIPX7TdD2NAxgP29cC0/1x1.jpg",
                              "link": {
                                "web": "https://search.daum.net/search?w=tot&DA=YZR&t__nil_searchbox=btn&sug=&sugo=&q="+json_data["data"][0]["품목명"]
                              }
                            },
                                {
                                  "title": json_data["data"][1]["품목명"],
                              "description": json_data["data"][1]["분류명"],
                                  "imageUrl": "http://k.kakaocdn.net/dn/N4Epz/btqqHCfF5II/a3kMRckYml1NLPEo7nqTmK/1x1.jpg",
                                  "link": {
                                    "web": "https://search.daum.net/search?w=tot&DA=YZR&t__nil_searchbox=btn&sug=&sugo=&q="+json_data["data"][1]["품목명"]
                                  }
                                },
                                {
                                  "title": json_data["data"][2]["품목명"],
                              "description": json_data["data"][2]["분류명"],
                                  "imageUrl": "http://k.kakaocdn.net/dn/bE8AKO/btqqFHI6vDQ/mWZGNbLIOlTv3oVF1gzXKK/1x1.jpg",
                                  "link": {
                                    "web": "https://search.daum.net/search?w=tot&DA=YZR&t__nil_searchbox=btn&sug=&sugo=&q="+json_data["data"][2]["품목명"]
                                  }
                                }
                              ],
                              "buttons": [
                                {
                                  "label": "처음으로",                                
                                  "action": "block",
                                  "blockId": "5e9cf354ebeedd00011a0c68"
                                }
                              ]
                            }
                          }
                        ]
                      }
                    }
        
    
    elif len(check2) == 2 :
        res = {
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                        "listCard": {
                            "header": {
                                "title": "다음과 같은 알약으로 판단됩니다.",
                                "imageUrl": "http://k.kakaocdn.net/dn/w9sXr/btqDFcuEEC8/REi0gyPtobK4O32pu4awy1/img_l.jpg"
                             },
                            "items": [
                                {
                              "title": json_data["data"][0]["품목명"],
                              "description": json_data["data"][0]["분류명"],
                              "imageUrl": "http://k.kakaocdn.net/dn/APR96/btqqH7zLanY/kD5mIPX7TdD2NAxgP29cC0/1x1.jpg",
                              "link": {
                                "web": "https://search.daum.net/search?w=tot&DA=YZR&t__nil_searchbox=btn&sug=&sugo=&q="+json_data["data"][0]["품목명"]
                              }
                            },
                                {
                                  "title": json_data["data"][1]["품목명"],
                              "description": json_data["data"][1]["분류명"],
                                  "imageUrl": "http://k.kakaocdn.net/dn/N4Epz/btqqHCfF5II/a3kMRckYml1NLPEo7nqTmK/1x1.jpg",
                                  "link": {
                                    "web": "https://search.daum.net/search?w=tot&DA=YZR&t__nil_searchbox=btn&sug=&sugo=&q="+json_data["data"][1]["품목명"]
                                  }
                                }
                              ],
                              "buttons": [
                                {
                                  "label": "처음으로",                                
                                  "action": "block",
                                  "blockId": "5e9cf354ebeedd00011a0c68"
                                }
                              ]
                            }
                          }
                        ]
                      }
                    }
        
    elif len(check2) == 1 :
        res = {
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                        "listCard": {
                            "header": {
                                "title": "다음과 같은 알약으로 판단됩니다.",
                                "imageUrl": "http://k.kakaocdn.net/dn/w9sXr/btqDFcuEEC8/REi0gyPtobK4O32pu4awy1/img_l.jpg"
                             },
                            "items": [
                                {
                                  "title": json_data["data"][0]["품목명"],
                                  "description": json_data["data"][0]["분류명"],
                                  "imageUrl": "http://k.kakaocdn.net/dn/APR96/btqqH7zLanY/kD5mIPX7TdD2NAxgP29cC0/1x1.jpg",
                                  "link": {
                                    "web": "https://search.daum.net/search?w=tot&DA=YZR&t__nil_searchbox=btn&sug=&sugo=&q="+json_data["data"][0]["품목명"]
                                  }
                                }                                
                              ],
                              "buttons": [
                                {
                                  "label": "처음으로",                                
                                  "action": "block",
                                  "blockId": "5e9cf354ebeedd00011a0c68"
                                }
                              ]
                            }
                          }
                        ]
                      }
                    }
    return jsonify(res)
    



    
    



if __name__ == '__main__':    
    app.run(host='127.0.0.1', port=5000, debug=True)

    
