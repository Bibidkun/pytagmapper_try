from collections import OrderedDict
from hack_sys_path import *
import argparse
import math
import matplotlib.pyplot as plt
import numpy as np

from pytagmapper.data import *
from pytagmapper.geometry import *
from pytagmapper.project import *
from pytagmapper.map_builder import solvePnPWrapper

def preparate_data():
    pass

def estimate_camera_view_points(tags_world_coords, tags_img_coords, cam_params):
    _camera_matrix = cam_params['camera_matrix']
    _distortion = cam_params['distortion_coefficients']
    tx_camera_world = solvePnPWrapper(tags_world_coords, tags_img_coords, _camera_matrix, _distortion)
    camera_view_point = SE3_inv(tx_camera_world)
    return camera_view_point

def main():

    parser = argparse.ArgumentParser(description='Plot a tag map.')
    parser.add_argument('map_dir', type=str, help='map directory')
    args = parser.parse_args()

    map_data = load_map(args.map_dir)
    tag_side_lengths = map_data['tag_side_lengths']

    map_type = map_data['map_type']
    map_data['tag_locations'] = OrderedDict(sorted(map_data['tag_locations'].items()))
    view_points_data = load_viewpoints(args.map_dir)
    # 2dのポイント座標，カメラ行列を取得
    scene_data = load_data(args.map_dir)
    # 3dのポイント座標を取得
    tag_corners_world = {}
    # 2 and 2.5d maps
    if map_type == '2.5d' or map_type == '2d':
        for tag_id, pose_world_tag in map_data['tag_locations'].items():
            if tag_id in tag_side_lengths:
                tag_side_length = tag_side_lengths[tag_id]
            else:
                tag_side_length = tag_side_lengths["default"]

            tag_corners_2d = get_corners_mat2d(tag_side_length)
                
            if map_type == '2.5d':
                xyt_world_tag = pose_world_tag[:3]
            else:
                xyt_world_tag = pose_world_tag

            tx_world_tag = xyt_to_SE2(np.array([xyt_world_tag]).T)
            world_corners = tx_world_tag @ tag_corners_2d
            for i in range(4):
                x1 = world_corners[0,i]
                x2 = world_corners[0,(i+1)%4]
                y1 = world_corners[1,i]
                y2 = world_corners[1,(i+1)%4]
                line = plt.Line2D((x1,x2), (y1,y2), lw=1.5)
                plt.gca().add_line(line)

            center = np.sum(world_corners, axis=1)/4
            plt.text(center[0], center[1], str(tag_id))

            if map_type == '2.5d':
                z = "{:#.4g}".format(pose_world_tag[3])
                plt.text(center[0], center[1] - tag_side_length/2, f"z={z}")
    elif map_type == '3d':
        for tag_id, pose_world_tag in map_data['tag_locations'].items():
            if tag_id in tag_side_lengths:
                tag_side_length = tag_side_lengths[tag_id]
            else:
                tag_side_length = tag_side_lengths["default"]
            tag_corners_3d = get_corners_mat(tag_side_length)                
            
            tx_world_tag = np.array(pose_world_tag)
            world_corners = tx_world_tag @ tag_corners_3d
            for i in range(4):
                x1 = world_corners[0,i]
                x2 = world_corners[0,(i+1)%4]
                y1 = world_corners[1,i]
                y2 = world_corners[1,(i+1)%4]
                line = plt.Line2D((x1,x2), (y1,y2), lw=1.5)
                plt.gca().add_line(line)
            # print(world_corners.copy()[:2,:].T)
            tag_corners_world[tag_id] = world_corners.copy()[:2,:].T # 3dの座標

            center = np.sum(world_corners, axis=1)/4
            plt.text(center[0], center[1], str(tag_id))

            z = "{:#.4g}".format(tx_world_tag[2,3])
            plt.text(center[0], center[1] - tag_side_length/2, f"z={z}")
        # for cam_id, cam_pos in view_points_data.items():
        #     pos = cam_pos[:3, 3]
        #     plt.plot(pos[0], pos[1], 'bo')
        #     plt.text(pos[0], pos[1]-5, str(cam_id))
        #     z = "{:#.4g}".format(pos[2])
        #     plt.text(pos[0] + 0.5, pos[1] - 0.5, f"z={z}")
    else:
        raise RuntimeError("Unsupported map type", map_type)
    
    tags_world_coords = np.vstack(tag_corners_world.values()) # id順に並べた3dの座標
    for cam_view in scene_data['viewpoints'].values():
        cam_id, tags = cam_view["cam_id"], OrderedDict(sorted(cam_view["tags"].items()))
        array = np.vstack([np.array(tag_corner).reshape(4,2) for tag_corner in tags.values()])
        
        camera_world_coords  = estimate_camera_view_points(tags_world_coords, np.vstack(tags.values()), scene_data['camera_matrix'][int(cam_id)])
        print(f"cam_id: {cam_id}")
        print(camera_world_coords)
        
    plt.axis('scaled')
    plt.show()

if __name__ == "__main__":
    main()
    
