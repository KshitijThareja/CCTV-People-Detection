import os
import numpy as np
import supervision as sv
import subprocess
from ultralytics import YOLO
import cv2
HOME = os.getcwd()
import matplotlib.pyplot as plt
subprocess.Popen(f"cd {HOME}", shell=True)
MARKET_SQUARE_VIDEO_PATH = f"{HOME}/market-square.mp4"
print(MARKET_SQUARE_VIDEO_PATH)
model = YOLO('yolov8s.pt')
colors = sv.ColorPalette.default()
# polygon = np.array([
#     [1725, 1550],
#     [2725, 1550],
#     [3500, 2160],
#     [1250, 2160]
# ])
# video_info = sv.VideoInfo.from_video_path(MARKET_SQUARE_VIDEO_PATH)
# zone = sv.PolygonZone(polygon=polygon, frame_resolution_wh=video_info.resolution_wh)

# # initiate annotators
# box_annotator = sv.BoxAnnotator(thickness=4,color=sv.Color.blue(), text_thickness=4, text_scale=2)
# zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.blue(), thickness=6, text_thickness=6, text_scale=4)
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
    # zone.trigger(detections=detections)

    # annotate
    for zone, zone_annotator, box_annotator in zip(zones, zone_annotators, box_annotators):
        mask = zone.trigger(detections=detections)
        detections_filtered = detections[mask]
        frame = box_annotator.annotate(scene=frame, detections=detections_filtered, skip_label=True)
        frame = zone_annotator.annotate(scene=frame)
    cv2.imshow("Image", frame)
    cv2.waitKey(1)
    return frame

sv.process_video(source_path=MARKET_SQUARE_VIDEO_PATH, target_path=f"{HOME}/mall-result.mp4", callback=process_frame)