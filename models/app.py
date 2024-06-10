import glob
import streamlit as st
from PIL import ImageDraw
from PIL import Image
import torch
import sys

import cv2
import os
import time
from ultralytics import YOLO
from datetime import datetime
import subprocess
import shutil
from glob import glob



def imageInput(model,src):

    if src == 'upload your own data':
        image_file = st.file_uploader(
            'upload an image',['.png','.jpg']
        )


    

def main():
    # global variables
    global model, confidence, cfg_model_path

    st.title("PMITO Detecttion")
    
    

    model_src = st.sidebar.selectbox('select model weight fille (recommend yolov8) ',['yolov8','yolov5+resnet50','yolov5+mobilenet','yolov9'])
    image_file = st.file_uploader(
            'upload an image',['png','jpg']
        )

    

    if model_src == 'yolov8':
        st.sidebar.title("choose model and setting ")
        confidence = float(st.sidebar.slider(
        "select Model Confidence",20,100,30
        ))/100
        model_path = 'hotaq/ai_builder/main/models/Yolov8.pt'
        model = YOLO(model_path)
        model_info_yolov8 = model.info()
        
        st.sidebar.subheader("Model Info")
        st.sidebar.text("Model Type: YOLOv8")
        st.sidebar.text(f"Number of classes: {model_info_yolov8[0]}")
        st.sidebar.text(f"Model Size: {model_info_yolov8[1]}")
        st.sidebar.text(f"Image Size: {model_info_yolov8[2]}")
        
    #elif model_src == 'yolov5':
        #st.sidebar.title("choose model and setting ")
        #confidence = float(st.sidebar.slider(
        #"select Model Confidence",25,100,30
        #))/100
        #threshold =  float(st.sidebar.slider(
        #"select Model Threshold",0,100,20
        #))/100
        #model_path = 'C:/Users/ADMINS/Downloads/AI_builder-main/AI_builder-main/best.pt'
        #model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
        #model_info = model
        
        st.sidebar.subheader("Model info")
        st.sidebar.text('Model Type: YOLOv5')
        st.sidebar.text(f"Number of classes: 206 layers")
        st.sidebar.text(f"Model Size: 12319756")
        st.sidebar.text(f"Image Size: 0")
        #st.sidebar.text(model_info)
        
    elif model_src == 'yolov5+resnet50':
        st.sidebar.title("choose model and setting ")
        confidence = float(st.sidebar.slider(
        "select Model Confidence",25,100,30
        ))/100
        threshold =  float(st.sidebar.slider(
        "select Model Threshold",0,100,20
        ))/100
        model_path = ' C:/Users/ADMINS/Downloads/AI_builder-main/AI_builder-main/models/Yolov5+resnet.pt'
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        model_info = model
        st.sidebar.subheader("Model Info")
        st.sidebar.text("Model Type: YOLOv5+resnet50")
        st.sidebar.text(model_info)
        
    elif model_src == 'yolov5+mobilenet':
        st.sidebar.title("choose model and setting ")
        confidence = float(st.sidebar.slider(
        "select Model Confidence",25,100,30
        ))/100
        threshold =  float(st.sidebar.slider(
        "select Model Threshold",0,100,20
        ))/100
        model_path = 'C:/Users/ADMINS/Downloads/AI_builder-main/AI_builder-main/models/Yolov5+mobilenet.pt'
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        model_info = model
        st.sidebar.subheader("Model Info")
        st.sidebar.text("Model Type: YOLOv5+resnet50")
        st.sidebar.text(model_info)
    elif model_src == 'yolov9':
        st.sidebar.title("choose model and setting ")
        confidence = float(st.sidebar.slider(
        "select Model Confidence",25,100,30
        ))/100
        threshold =  float(st.sidebar.slider(
        "select Model Threshold",0,100,20
        ))/100
        st.sidebar.subheader("Model Info")
        st.sidebar.text("Model Type: Yolov9")
        st.sidebar.text(f"Number of classes: 588 layers")
        st.sidebar.text(f"Model Size: 32557504")
        st.sidebar.text(f"Image Size: 640")

    submit = st.button("Predict!")
    col1 ,col2 = st.columns(2)
    if image_file is not None:
        img = Image.open(image_file)
        with col1:
            st.image(img, caption='Uploaded Image',
            use_column_width='always')
        ts = datetime.timestamp(datetime.now())
        imgpath = os.path.join('data/uploads/',str(ts)+image_file.name)
        outputpath = os.path.join(
                'data/outputs', os.path.basename(imgpath))
        with open(imgpath, mode="wb") as f:
                f.write(image_file.getbuffer())
        with col2:
            if image_file is not None and submit:
                with st.spinner(text='Predicting...'):
                    
                    #if model_src == 'yolov5':
                       # model_path = 'C:/Users/ADMINS/Downloads/AI_builder-main/AI_builder-main/best.pt'
                        #command_yolov5 #f'python C:/Users/ADMINS/Downloads/AI_builder-main/AI_builder-main/yolov5-master/detect.py --weight {model_path} --img 640 --conf {confidence} --source C:/Users/ADMINS/Downloads/AI_builder-main/AI_builder-main/{imgpath} '
                        #files_folder1 = glob('/home/hootoo/Downloads/Code/Ai_builder/main/data/uploads/*.jpg')
                        #for file_path in files_folder1:
                         #   os.remove(file_path)
                        
                        
                    if model_src == 'yolov8':
                        model_path = 'C:/Users/ADMINS/Downloads/AI_builder-main/AI_builder-main/models/Yolov8.pt'
                        model = YOLO(model_path)
                        res = model(imgpath, conf=confidence )
                        boxes = res[0].boxes
                        res_plotted = res[0].plot()[:, :, ::-1]
                        files_folder1 = glob('/home/hootoo/Downloads/Code/Ai_builder/main/data/uploads/*.jpg')
                        
                        for file_path in files_folder1:
                            os.remove(file_path)
                        
                        st.image(res_plotted, caption='Detected Image',
                         use_column_width='always')
                        try:
                            with st.expander("Detection Results"):
                                for box in boxes:
                                    st.write(box.data)

                        except Exception as ex:
                            st.write(ex)
                            st.write("No image is uploaded yet!")
                    if  model_src == 'yolov5+resnet50':
                        model_path = 'C:/Users/ADMINS/Downloads/AI_builder-main/AI_builder-main/models/Yolov5+renet.pt'
                        #command1 = 'cd /home/hootoo/Downloads/Code/Ai_builder/main/flexible-yolov5/'
                        command2 = f'python C:/Users/ADMINS/Downloads/AI_builder-main/AI_builder-main/scripts/detector.py  --weights C:/Users/ADMINS/Downloads/AI_builder-main/AI_builder-main/models/Yolov5+resnet.pt --imgs_root C:/Users/ADMINS/Downloads/AI_builder-main/AI_builder-main/data/uploads   --save_dir  C:/Users/ADMINS/Downloads/AI_builder-main/AI_builder-main/save_resnet --img_size  640  --conf_thresh {confidence} --iou_thresh {threshold}'
                        
                        #os.system(command1)
                      
                        
                        
                        os.system(command2)
                        processed_imgs_dir_resnet50 = 'C:/Users/ADMINS/Downloads/AI_builder-main/AI_builder-main/save_resnet'
                        processed_imgs_resnet50 = os.listdir(processed_imgs_dir_resnet50)
                        if processed_imgs_resnet50:
                            img_name = processed_imgs_resnet50[0]
                            processed_img_path = os.path.join(processed_imgs_dir_resnet50, img_name)
                            st.image(processed_img_path, caption=img_name)
                        files_folder1 = glob('C:/Users/ADMINS/Downloads/AI_builder-main/AI_builder-main/data/uploads/*.jpeg')
                        filese_folder2 = glob('C:/Users/ADMINS/Downloads/AI_builder-main/AI_builder-main/save_resnet/*.jpeg')
                        for file_path in files_folder1:
                            os.remove(file_path)
                        for file_path in filese_folder2:
                            os.remove(file_path)
                       
                    if model_src == 'yolov5+mobilenet':
                        model_path = 'C:/Users/ADMINS/Downloads/AI_builder-main/AI_builder-main/models/Yolov5+mobilenet.pt'
                        command2 = f'python C:/Users/ADMINS/Downloads/AI_builder-main/AI_builder-main/scripts/detector.py  --weights {model_path} --imgs_root C:/Users/ADMINS/Downloads/AI_builder-main/AI_builder-main/data/uploads   --save_dir C:/Users/ADMINS/Downloads/AI_builder-main/AI_builder-main/save_mobile  --img_size  640  --conf_thresh {confidence} --iou_thresh {threshold}'

                        os.system(command2)
                        processed_img_dirs_mobilenet = 'C:/Users/ADMINS/Downloads/AI_builder-main/AI_builder-main/save_mobile'
                        processed_imgs_mobilenet = os.listdir(processed_img_dirs_mobilenet)
                        if processed_imgs_mobilenet:
                            img_name = processed_imgs_mobilenet[0]
                            processed_img_path = os.path.join(processed_img_dirs_mobilenet, img_name)
                            st.image(processed_img_path, caption=img_name)
       
                        files_folder1 = glob('C:/Users/ADMINS/Downloads/AI_builder-main/AI_builder-main/data/uploads/*.jpeg')
                        files_folder2 = glob('C:/Users/ADMINS/Downloads/AI_builder-main/AI_builder-main/save_mobile/*.jpeg')
                        for file_path in files_folder1:
                            os.remove(file_path)
                        for file_path in files_folder2:
                            os.remove(file_path)
                    if model_src == 'yolov9':
                        #imgpath_yolov9 = glob('C:/Users/ADMINS/Downloads/AI_builder-main/AI_builder-main/data/uploads/*.jpeg')
                        
                        model_path_yolov9 = 'C:/Users/ADMINS/Downloads/AI_builder-main/AI_builder-main/models/yolov9.pt'
                        commnad_yolov9 = f'python C:/Users/ADMINS/Downloads/AI_builder-main/AI_builder-main/yolov9-main/detect_dual.py --source C:/Users/ADMINS/Downloads/AI_builder-main/AI_builder-main/{imgpath} --img 640  --weights {model_path_yolov9} --project detect_yolov9 --conf-thres {confidence} --iou-thres {threshold} '
                        subprocess.run(commnad_yolov9,shell=True)
                        image_yolov9_dir = glob('C:/Users/ADMINS/Downloads/AI_builder-main/AI_builder-main/detect_yolov9/exp/*.jpeg')
                        st.image(image_yolov9_dir, caption='detection-image')
                        files_folder1 = glob('C:/Users/ADMINS/Downloads/AI_builder-main/AI_builder-main/data/uploads/*.jpeg')
                        for file_path in files_folder1:
                            os.remove(file_path)
                        
               

                
       







if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass
