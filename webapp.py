import argparse
from PIL import Image
import cv2
import numpy as np
import tensorflow as tf
from re import DEBUG, sub
from flask import Flask, render_template, request, redirect, send_file, url_for, Response
from werkzeug.utils import secure_filename, send_from_directory
import os
import time
import os
import numpy as np
import supervision as sv
from subprocess import Popen
from ultralytics import YOLO
import cv2
import os
import subprocess
HOME = os.getcwd()

app = Flask(__name__)
check=f"{HOME}/runs/detect/result.mp4"
# if(os.path.exists(check)):
#     os.remove(check)
@app.route("/")
def hello_world():
    return render_template('index.html')
@app.route("/", methods=["GET", "POST"])
def predict_img():
    if request.method == "POST":
        if 'file' in request.files:
            f = request.files['file']
            
            basepath = os.path.dirname(__file__)
            # print("bspath=", basepath)
            filepath = os.path.join(basepath,'uploads',f.filename)
            print("upload folder is ", filepath)
            f.save(filepath)
            global imgpath
            predict_img.imgpath = f.filename
            # print("printing predict_img :::::: ", predict_img)

            file_extension = f.filename.rsplit('.', 1)[1].lower()     
            if file_extension == 'mp4':
                
                video_path= filepath
                print(video_path)
                subprocess.Popen(f"cd {HOME}", shell=True)
                MARKET_SQUARE_VIDEO_PATH = video_path
                print(MARKET_SQUARE_VIDEO_PATH)
                model = YOLO('yolov8s.pt')
                colors = sv.ColorPalette.default()

                rel=os.path.relpath(MARKET_SQUARE_VIDEO_PATH, start = os.curdir)
                vcap = cv2.VideoCapture(rel)
                width=height=0
                if vcap.isOpened(): 
                    width  = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))   # float `width`
                    height = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float `height`
                
                print('width, height:', width, height)

                polygons = [
                    np.array([
                        [0,0],
                        [0, height],
                        [width, height],
                        [width, 0]
                    ], np.int32),
                
                ]
                video_info = sv.VideoInfo.from_video_path(MARKET_SQUARE_VIDEO_PATH)

                zones = [
                    sv.PolygonZone(
                        polygon=polygon, 
                        frame_resolution_wh=video_info.resolution_wh
                    )
                    for polygon
                    in polygons
                ]
                zone_annotators = [
                    sv.PolygonZoneAnnotator(
                        zone=zone, 
                        color=colors.by_idx(index), 
                        thickness=6,
                        text_thickness=8,
                        text_scale=4
                    )
                    for index, zone
                    in enumerate(zones)
                ]
                box_annotators = [
                    sv.BoxAnnotator(
                        color=colors.by_idx(index), 
                        thickness=4, 
                        text_thickness=4, 
                        text_scale=2
                        )
                    for index
                    in range(len(polygons))
                ]
                def process_frame(frame: np.ndarray, i) -> np.ndarray:
                    # detect
                    results = model(frame, imgsz=1280)[0]
                    detections = sv.Detections.from_yolov8(results)
                    detections = detections[detections.class_id == 0]
                    # annotate
                    for zone, zone_annotator, box_annotator in zip(zones, zone_annotators, box_annotators):
                        mask = zone.trigger(detections=detections)
                        detections_filtered = detections[mask]
                        frame = box_annotator.annotate(scene=frame, detections=detections_filtered, skip_label=True)
                        frame = zone_annotator.annotate(scene=frame)
                    return frame

                sv.process_video(source_path=MARKET_SQUARE_VIDEO_PATH, target_path=f"{HOME}/runs/detect/result.mp4", callback=process_frame)                                

            else:
                print("Invalid video format")

    folder_path = 'runs/detect'
    # subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]    
    # latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))    
    image_path = folder_path+'/result.mp4'
    if os.path.exists(image_path):
        return render_template('indexnew.html', image_path=image_path)
    else:
        return render_template('index.html', image_path=image_path)
    # return render_template('index.html', image_path=image_path)
    

# function to get the frames from video (output video)

def get_frame():
    folder_path = 'runs/detect/result.mp4'
    image_path=folder_path    
    video = cv2.VideoCapture(image_path)  # detected video path
    while True:
        success, image = video.read()
        if not success:
            break
        ret, jpeg = cv2.imencode('.jpg', image)   
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')   
        time.sleep(1)  #control the frame rate to display one frame every 100 milliseconds: 


# function to display the detected objects video on html page
@app.route("/video_feed")
def video_feed():
    return Response(get_frame(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# The display function is used to serve the image or video from the folder_path directory.
@app.route('/<path:filename>')
def display(filename):
    folder_path = 'runs/detect/result.mp4'
    image_path=folder_path    
    directory = folder_path
    print("printing directory: ",directory)  
    filename = predict_img.imgpath
    file_extension = filename.rsplit('.', 1)[1].lower()
    #print("printing file extension from display function : ",file_extension)
    environ = request.environ
    if file_extension == 'jpg':      
        return send_from_directory(directory,filename,environ)

    elif file_extension == 'mp4':
        return render_template('index.html')

    else:
        return "Invalid file format"
@app.route('/download')
def download():
    path = 'runs/detect/result.mp4'
    return send_file(path, as_attachment=True)
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Flask app exposing yolov5 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()
    app.jinja_env.auto_reload = True
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(debug=True, host="0.0.0.0", port=args.port)  # debug=True causes Restarting with stat

