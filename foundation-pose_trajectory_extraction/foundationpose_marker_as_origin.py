# change the foundationpose's output into given marker as the origin
# save the result in a [N 4 4] array

import os
import pdb
import numpy as np
import sys

# path to data folder
# data_path = "/mnt/data/cup_with_clamp"
data_path = sys.argv[1]
ref_T_path = sys.argv[2]

# reference T[ground_marker/camera], calculated by get_reference_T.py
ref_T = np.load(ref_T_path)

files = os.listdir(data_path + "/ob_in_cam")
files.sort()

output_T = []

for file in files:
    file_path = data_path + "/ob_in_cam/" + file

    ori_T = np.loadtxt(file_path)   # T[camera/object]
    relative_T = np.matmul(ref_T, ori_T)   # T[groung_marker/object]
    
    # pdb.set_trace()
    output_T.append(relative_T)

output_T = np.array(output_T)
np.save(file=data_path+"/fp_ground_marker_origin_T.npy", arr=output_T)



