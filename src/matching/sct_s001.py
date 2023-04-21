import json
from src.matching.matching.single_s001 import Single_camera_matching


from src.utils.utils import load_defaults

if __name__ == '__main__':
    cameras_S001 = ['c001','c002', 'c003', 'c005', 'c006', 'c007']
    # cameras_S001 = ['c003']
    for cam in cameras_S001:
        single_camera_matching = Single_camera_matching(cam)
        single_camera_matching.run()           
        