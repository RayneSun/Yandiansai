import os
import json

def get_img_path(img_path):
    file_path=[]
    for file in os.listdir(img_path):
        file_path.append(os.path.join(img_path+'/', file))
    return file_path

def get_label(label_path):# ./label.json
    labels={}
    with open(label_path,'r',encoding='UTF-8') as f:
        js=json.load(f)
    for i in js:
        x,y,name=i['Data']['svgArr'][0]['data'][0]['x'],i['Data']['svgArr'][0]['data'][0]['y'],i['imageName']
        labels[name]=[x,y]

    return labels

def get_all(img_path,label_path):
    file_path=get_img_path(img_path)
    labels=get_label(label_path)
    label=[]
    for file in file_path:
        label.append(labels[file.split('/')[2]])
    return file_path,label
