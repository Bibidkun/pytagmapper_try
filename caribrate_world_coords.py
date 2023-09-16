import json
import cv2

import numpy as np
import math

def get_camera_pose(json_file):
    with open(json_file) as f:
        data = json.load(f)
        
    return np.array(data["0"]) 
    
def caribrate_pose(pose_data):
    """
    args: pose_data: 4x4 matrix of camera pose
    """
    rot = pose_data[:3, :3]
    tvec = pose_data[:3, 3:4]
    print(rot)
    print(tvec)
    # ベクトルの定義を変更する
    R_raw = rot.T
    t_raw = -R_raw @ tvec
    print("R_raw: ", R_raw)
    print("t_raw: ", t_raw)


if __name__ == "__main__":
    pose = get_camera_pose("camera1/viewpoints.json")
    caribrate_pose(pose)