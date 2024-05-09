# umi related codes

Note: the files in this repository is assembled from multiple places in a bigger directory. I'm still testing whether the dependency relationships are complete.

Codes for umi-related works:  
- Gopro trajectory extraction
- Foundation-pose estimation of object's pose
- Load trajectories into sapien simulator
- Aruco marker detection

## Gopro trajectory extraction
Extract the Gopro's trajectory using umi. Save the trajectory as transform matrix (4*4 matrix T)  
The 

### Prerequisites
- Place (fix) a custom aruco marker (along side with the marker umi asked to be used) on the table.
- Scan the scene following the umi's guide for [scene construction](https://swanky-sphere-ad1.notion.site/UMI-Data-Collection-Instruction-4db1a1f0f2aa4a2e84d9742720428b4c)
- Remove the marker that umi used. Don't remove the custom marker because slam is unstable if the custom marker is added to the scene after creating the surrounding environment.
- Record your task using Gopro. Make sure the marker is fully visible in the Gopro record's first frame.
- Follow the umi's slam pipeline to generate the camera_trajectory.csv for this scene (this file does not exist if the slam fails).

### Usage
First, extract the first frame  

```python
python gopro_trajectory_extraction/extract_first_frame.py [path-to-.mp4]
```

Second, calculate the T[marker/camera_origin]. camera_origin is the camera's position at the first frame  
Remember to change the type of marker if you are not using one of the first 50 7*7 markers.  

```python
python gopro_trajectory_extraction/get_reference_T.py [path-to-gp_first_frame.png]
```

Finally, compute the T matrix and save them in a .npy file. Array is saved with a shape of [N1 4 4] where N is the number of frames.  
You need to align the realsense's video and Gopro's video because it's hard to let them start to record simultaneously. A simple stop watch shown on a mobile phone could help you to get this done. *first_frame_index* is the index of the new first frame in Gopro's original record. 

```python
python gopro_trajectory_extraction/gopro_marker_as_origin.py [path-to-camera_trajectory.csv] [first_frame_index] [path-to-T_matrix.npy]
```

You will get the gopro_T.npy as the final output. Matrices saved in this have the custom marker as the origin.  

## Foundation-pose trajectory extraction
Get the estimated pose from foundationpose and change the origin to the custom marker.  

### Prerequisites
- Place the realsense camera on the side of the scene and make sure the custom marker could be fully viewed at the first frame.  
- Get a .obj model for the object to be tracked. The unit of the mesh must be meter. You can use sudoai to generate the mesh and use blender to resize it. A ruler is needed to accurately match the real size.

### Usage
First, record the scene as a series of rgb/depth pictures.  
Press Space to start recording after the program is launched. Press Space again to stop recording.

```python
python foundation_pose_trajectory_extraction/realsense_align_rgb_depth.py
```

Second, generate a black-white mask of the target object at the first frame. The mask should also be a picture with the same resolution to the rgb' first frame.  

Third, organize the rgb, depth, masks, mesh file in a folder with the following format:  
\- scene_folder  
\-- depth  
\-- masks  
\-- mesh  
\-- rgb  
\-- cam_K.txt  
cam_K.txt is generated in the first part.  
Refer to foundation-pose's demo data for specific data arrangement.  

Fourth, run the foundation-pose.  
cd to foundation-pose's directory.  
```python
python run_demo.py --mesh_file [path-to-.obj] --test_scene_dir [path-to-scene_folder]
```

A folder called *ob_in_cam* should be generated under scene_folder.

Fifth, run get_reference_T.py to generate the T[marker/camera]  
```python
python foundation_pose_trajectory_extraction/get_reference_T.py [path-to-first-rgb-image]
```

Sixth, align the foundation-pose's image with gopro manually. delete the extra frames in ob_in_cam to let the first frame of foundation-pose aligned with the first frame of gopro.  

Finally, save all the T[marker/object] to a .npy file. Array in this file has a shape of [N2 4 4]
```python
python foundation_pose_trajectory_extraction/foundationpose_marker_as_origin.py [path-to-scene_folder] [path-to-T_matrix.npy]
```

## Automatic trajectory extraction
Use main.py to automatic the process of trajectory extraction/calibration for both GoPro and Foundation-Pose.  
Outputs are the gp_T.npy and fp_T.npy saved in the same directory of main.py.  
Both files contain the gripper/object's trajectory with the marker as the reference frame, and could be used directly for sapien simulation.  

## Prerequisites
- Run the Foundation-Pose to get object's T matrices saved in ./ob_in_cam
- Run the UMI to get the GoPro's trajectory saved in camera_trajectory.csv
- Get the raw record (raw_video.mp4) of GoPro in this scene. Usually it is located in the same directory with camera_trajectory.csv
- Get the path to foundation-pose directory of this scene. It contains sub-directories of ./rgb ./depth ./ob_in_cam etc.

## Usage
```python
python main.py [path-to-camera_trajectory.csv] [path-to-raw_video.mp4] [path-to-foundation-pose-directory]
```


## Load to sapien
Load and show the generated tajectory matrix T into sapien.  

### Prerequisites
- Calculate the matrix T of object' trajectory or gopro's trajectory.
- setup the sapien environment

### Usage
Use real2sim_from_T.py to show the trajectory of object.  
The object is represented by a red box for now.  

```python
python load_to_sapien/real2sim_from_T.py [path-to-trajectory-T-npy-file]
```

Use real2sim_from_T_gripper.py to show the gripper's trajectory with Xarm's urdf mesh.  
Xarm's urdf files are attached in ./xarm_urdf  

```python
python load_to_sapien/real2sim_from_T_gripper.py [path-to-trajectory-T-npy-file]
```

Use real2sim_both.py to show the gripper's and object's trajectories.  

```python
python load_to_sapien/real2sim_both.py [path-to-gp_T.npy] [path-to-fp_T.npy]
```



