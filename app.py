import glob
import streamlit as st
from PIL import ImageDraw
from PIL import Image
import torch
import cv2
import os
import time
from ultralytics import YOLO
from datetime import datetime

def imageInput(model,src):

    if src == 'upload your own data':
        image_file = st.file_uploader(
            'upload an image',['.png','.jpg']
        )


    

def main():
    # global variables
    global model, confidence, cfg_model_path

    st.title("PMITO Detecttion")

    

    model_src = st.sidebar.selectbox('select model weight fille (recommend yolov8) ',['yolov5','yolov8','yolov5+resnet50','yolov5+mobilenet','yolov9'])
    image_file = st.file_uploader(
            'upload an image',['png','jpg']
        )

    

    if model_src == 'yolov8':
        st.sidebar.title("choose model and setting ")
        confidence = float(st.sidebar.slider(
        "select Model Confidence",25,100,20
        ))/100
        model_path = '/home/hootoo/Downloads/Code/Ai_builder/main/Yolov8/best.pt'
        model = YOLO(model_path)
        model_info = model.info()
        
        st.sidebar.subheader("Model Info")
        st.sidebar.text("Model Type: YOLOv8")
        st.sidebar.text(f"Number of classes: {model_info[0]}")
        st.sidebar.text(f"Model Size: {model_info[1]}")
        st.sidebar.text(f"Image Size: {model_info[2]}")
    elif model_src == 'yolov5':
        st.sidebar.title("choose model and setting ")
        confidence = float(st.sidebar.slider(
        "select Model Confidence",25,100,30
        ))/100
        threshold =  float(st.sidebar.slider(
        "select Model Threshold",0,100,20
        ))/100
        model_path = '/home/hootoo/Downloads/Code/Ai_builder/main/Yolov5/best.pt'
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        model_info = model
        
        st.sidebar.subheader("Model info")
        st.sidebar.text('Model Type: YOLOv5')
        st.sidebar.text(f"Number of classes: 206 layers")
        st.sidebar.text(f"Model Size: 12319756")
        st.sidebar.text(f"Image Size: 0")
        
    elif model_src == 'yolov5+resnet50':
        st.sidebar.title("choose model and setting ")
        confidence = float(st.sidebar.slider(
        "select Model Confidence",25,100,20
        ))/100


    submit = st.button("Predict!")
    col1 ,col2 = st.columns(2)
    if image_file is not None:
        img = Image.open(image_file)
        with col1:
            st.image(img, caption='Uploaded Image',
            use_column_width='always')
        ts = datetime.timestamp(datetime.now())
        imgpath = os.path.join('data/uploads',str(ts)+image_file.name)
        outputpath = os.path.join(
                'data/outputs', os.path.basename(imgpath))
        with open(imgpath, mode="wb") as f:
                f.write(image_file.getbuffer())
        with col2:
            if image_file is not None and submit:
                with st.spinner(text='Predicting...'):
                    if model_src == 'yolov5':
                        model_path = '/home/hootoo/Downloads/Code/Ai_builder/main/Yolov5/best.pt'
                        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
                        model.conf = confidence
                        model.iou = threshold
                        results = model(imgpath)
                        os.remove('/home/hootoo/Downloads/Code/Ai_builder/main/'+imgpath)
                        
                        st.image(results.render()[0], caption='Detected Image', use_column_width='always')
                    if model_src == 'yolov8':
                        model_path = '/home/hootoo/Downloads/Code/Ai_builder/main/Yolov8/best.pt'
                        model = YOLO(model_path)
                        res = model(imgpath, conf=confidence)
                        boxes = res[0].boxes
                        res_plotted = res[0].plot()[:, :, ::-1]
                        os.remove('/home/hootoo/Downloads/Code/Ai_builder/main/'+imgpath)
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
                        model_path = '/home/hootoo/Downloads/Code/Ai_builder/main/resnet50+yolov5/Run_25_2/weights/best.pt'
                        #command1 = 'cd /home/hootoo/Downloads/Code/Ai_builder/main/flexible-yolov5/'
                        command2 = f'python /home/hootoo/Downloads/Code/Ai_builder/main/detector_yolov5_backbone.py  --weights /home/hootoo/Downloads/Code/Ai_builder/main/resnet50+yolov5/Run_25_2/weights/best.pt --imgs_root /home/hootoo/Downloads/Code/Ai_builder/main/data/uploads   --save_dir  /home/hootoo/Downloads/Code/Ai_builder/main/detect_yolov+resnet --img_size  640  --conf_thresh {confidence} --iou_thresh 0.4'
                        
                        #os.system(command1)
                      
                        
                        
                        os.system(command2)

                        processed_imgs_dir = '/home/hootoo/Downloads/Code/Ai_builder/main/detect_yolov+resnet'

                        for img_name in os.listdir(processed_imgs_dir):
                            processed_img_path = os.path.join(processed_imgs_dir, img_name)
                            st.image(processed_img_path, caption=img_name)
        # Optionally, remove the original image
                            os.remove(os.path.join('/home/hootoo/Downloads/Code/Ai_builder/main/data/uploads', img_name))
                       
                        

                        
       







if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass