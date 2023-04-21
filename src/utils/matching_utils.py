from shapely.geometry import Polygon
import os
import json
def get_iou(bbox, pts):
    polygon1 = Polygon(bbox)
    polygon2 = Polygon(pts)
    intersect = polygon1.intersection(polygon2).area
    return intersect / polygon1.area
def check_inside(bboxes, outzone):
    # Outzone 
    points = outzone
    _count = 0
    for box in bboxes:
        x, y, w, h = box
        for point in points:
            _iou = get_iou([(x,y), (x+w,y), (x+w, y+h), (x,y+h)], point)
            if _iou > 0.9: 
                _count += 1
                break
    return _count / len(bboxes)

def calculate_err(x, y):
    return (abs(x-y)/(y+1e-8)) * 100

def load_roi_mask():
    out_bbox = {}
    out_zone = {}
    for file in os.listdir("datasets/ROI"):
        if "out_bbox" in file:
            camera = file.replace("out_bbox_", "").replace(".json", "")
            out_bbox[camera] = json.load(open(os.path.join("datasets/ROI", file)))
        if "out_zone_tracklet" in file:
            camera = file.replace("out_zone_tracklet_", "").replace(".json", "")
            out_zone[camera] = json.load(open(os.path.join("datasets/ROI", file)))
    return out_bbox, out_zone