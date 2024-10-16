# License: Apache 2.0. See LICENSE file in root directory.
# Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#####################################################
##              Align Depth to Color               ##
#####################################################

import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os
from datetime import datetime
import sys

def record_a_demo_fp_only(scene_name:str = "", depth_max_dist_meters=2, seq:list[str]=['gripper']):

    # gripper_camera_id = "242322072561"
    # side_camera_id = "146322076186"

    # Create a pipeline
    pipeline = rs.pipeline()
    

    config = rs.config()
    # config.enable_device(gripper_camera_id)


    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    profile = pipeline.start(config)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: ", depth_scale)

    # We will be removing the background of objects more than
    #  clipping_distance_in_meters meters away
    clipping_distance_in_meters = depth_max_dist_meters  # meters
    clipping_distance = clipping_distance_in_meters / depth_scale

    align_to = rs.stream.color
    align = rs.align(align_to)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    now = datetime.now()
    t = now.strftime("%m.%d.%H.%M")

    if scene_name != "":
        if os.path.exists(os.path.join(script_dir, "out", scene_name)):
            print(f"record folder of name '{scene_name}' exists. using default name of '{t}' ...")
        else:
            t = scene_name
        

    script_dir = os.path.dirname(os.path.abspath(__file__))
    subfolder_depth = os.path.join(script_dir, "out", t, seq[0], "depth")
    subfolder_rgb = os.path.join(script_dir, "out", t, seq[0], "rgb")


    os.makedirs(subfolder_depth, exist_ok=True)
    os.makedirs(subfolder_rgb, exist_ok=True)


    RecordStream = False
    framename = 0
    time_stamp_list = []

    cv2.namedWindow("Align Example", cv2.WINDOW_AUTOSIZE)

    # Streaming loop
    try:
        while True:
            # Get frameset of color and depth
            frames = pipeline.wait_for_frames()
            # frames.get_depth_frame() is a 640x360 depth image

            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # Get aligned frames
            aligned_depth_frame = (
                aligned_frames.get_depth_frame()
            )  # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()

            # Get instrinsics from aligned_depth_frame
            intrinsics = aligned_depth_frame.profile.as_video_stream_profile().intrinsics

            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())


            cv2.imshow("Align Example", color_image)

            key = cv2.waitKey(1)

            # Start saving the frames if space is pressed once until it is pressed again

            if key & 0xFF == ord(" "):
                if not RecordStream:
                    time.sleep(0.2)
                    RecordStream = True

                    with open(os.path.join(script_dir, "out", t, seq[0], "cam_K.txt"), "w") as f:
                        f.write(f"{intrinsics.fx} {0.0} {intrinsics.ppx}\n")
                        f.write(f"{0.0} {intrinsics.fy} {intrinsics.ppy}\n")
                        f.write(f"{0.0} {0.0} {1.0}\n")

                    print("Recording started")
                else:
                    print("Recording stopped")
                    cv2.destroyAllWindows()
                    break


            if RecordStream:

                # Define the path to the image file within the subfolder
                image_path_depth = os.path.join(subfolder_depth, f"{framename:06d}.png")
                image_path_rgb = os.path.join(subfolder_rgb, f"{framename:06d}.png")

                framename += 1
                time_stamp_list.append(color_frame.get_timestamp()) # ms

                cv2.imwrite(image_path_depth, depth_image)
                cv2.imwrite(image_path_rgb, color_image)

            # Press esc or 'q' to close the image window
            if key & 0xFF == ord("q") or key == 27:

                cv2.destroyAllWindows()

                break
    finally:
        pipeline.stop()
        if RecordStream:
            print("saving timestamps ...")
            with open(os.path.join(script_dir, "out", t, "timestamp.txt"), "w") as f:
                for time_i in time_stamp_list:
                    f.write(str(time_i))
                    f.write("\n")
            
            print("finish saving the timestamps")

            for object in seq[1:]:
                object_path_i = os.path.join(script_dir, "out", t, object)
                os.makedirs(object_path_i, exist_ok=True)

                os.symlink(src="../"+seq[0]+"/rgb", dst=object_path_i+"/rgb")
                os.symlink(src="../"+seq[0]+"/depth", dst=object_path_i+"/depth")
                os.symlink(src="../"+seq[0]+"/cam_K.txt", dst=object_path_i+"/cam_K.txt")


if __name__ == "__main__":
    # record a single scene
    try:
        scene_name = sys.argv[1]
    except:
        scene_name = ""
    record_a_demo_fp_only(scene_name=scene_name, seq=['gripper'])

    # record multiple scenes
    # for i in range(1, 11):
    #     print(f"start to record scene {i} ...")
    #     scene_name = str(i)
    #     record_a_demo_fp_only(scene_name=scene_name, seq=['gripper', 'cup'])