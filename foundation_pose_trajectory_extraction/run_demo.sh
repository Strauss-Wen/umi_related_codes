mesh_file=$1
scene_dir=$2


source ~/miniconda3/etc/profile.d/conda.sh
source ~/.bashrc

conda activate foundationpose
cd /mnt/robotics/FoundationPose-main

export CUDA_HOME=/usr/local/cuda

python run_demo.py --mesh_file $mesh_file --test_scene_dir $scene_dir