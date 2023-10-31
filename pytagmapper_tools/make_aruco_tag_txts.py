from hack_sys_path import *

import argparse
from pytagmapper.data import *
import cv2

def main():
    parser = argparse.ArgumentParser(description='Write tags.txt into a directory of images using opencv aruco tag detector.')
    parser.add_argument('image_dir', type=str, help='directory of image_{id}.png images')
    parser.add_argument('--show-detections', '-s', action='store_true', default=False, help='use cv2.imshow to show detected tags')
    args = parser.parse_args()
    
    image_paths = get_image_paths(args.image_dir)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    
    aruco_params = cv2.aruco.DetectorParameters()

    # BGR format
    aruco_side_colors = [(0, 0, 255),
                         (0, 255, 0),
                         (255, 0, 0),
                         (0, 255, 255)]

    for file_id, image_path_and_id in image_paths.items():
        image_path, cam_id = image_path_and_id[0], image_path_and_id[1]
        image = cv2.imread(image_path)
        aruco_corners, aruco_ids, aruco_rejected = \
            cv2.aruco.detectMarkers(image, aruco_dict, parameters=aruco_params)
        
        camera_matrix = [[1.27368947e+03, 0, 9.64119740e+02],
            [0.00000000e+00, 1.27512097e+03, 5.28344898e+02],
            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
        revecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(aruco_corners, 0.60, camera_matrix,None)

        with open(os.path.join(args.image_dir, f"tags_{cam_id}_{file_id}.txt"), "w") as f:
            for tag_idx, tag_id in enumerate(aruco_ids):
                tag_id = tag_id[0]
                acorners = aruco_corners[tag_idx][0]
                f.write(f"{tag_id}\n")
                f.write(f"{acorners[0][0]} {acorners[0][1]}\n")
                f.write(f"{acorners[1][0]} {acorners[1][1]}\n")
                f.write(f"{acorners[2][0]} {acorners[2][1]}\n")
                f.write(f"{acorners[3][0]} {acorners[3][1]}\n")

                if args.show_detections:
                    for i in range(4):
                        start = (int(acorners[i][0]), int(acorners[i][1]))
                        end = (int(acorners[(i+1)%4][0]), int(acorners[(i+1)%4][1]))
                        color = aruco_side_colors[i]
                        cv2.line(image, start, end, color, thickness=2)

                    center = np.sum(acorners, axis=0)/4
                    cv2.putText(image, str(tag_id), (int(center[0]),int(center[1])), 0, 2, (0, 255, 255), thickness=5)

        if args.show_detections:
            scale = 900 / image.shape[1]
            resize_shape = (int(image.shape[1] * scale),
                            int(image.shape[0] * scale))
            resized = cv2.resize(image, resize_shape)
            cv2.imshow(image_path, resized)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
