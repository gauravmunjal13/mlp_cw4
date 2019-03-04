#!/bin/sh
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --partition=Standard
#SBATCH --gres=gpu:1
#SBATCH --mem=12000  # memory in Mb
#SBATCH --time=0-08:00:00

export CUDA_HOME=/opt/cuda-9.0.176.1/

export CUDNN_HOME=/opt/cuDNN-7.0/

export STUDENT_ID=$(whoami)

export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH

export CPATH=${CUDNN_HOME}/include:$CPATH

export PATH=${CUDA_HOME}/bin:${PATH}

export PYTHON_PATH=$PATH

mkdir -p /disk/scratch/${STUDENT_ID}


export TMPDIR=/disk/scratch/${STUDENT_ID}/
export TMP=/disk/scratch/${STUDENT_ID}/

mkdir -p ${TMP}/datasets/
export DATASET_DIR=${TMP}/datasets/
# Activate the relevant virtual environment:


source /home/${STUDENT_ID}/miniconda3/bin/activate mlp

# Default dataset is cifar10
# Train baseline
python main.py --exp_type=train --exp_name=smallcnn
# Test baseline on clean test set
python main.py --exp_type=test --exp_name=smallcnn
# Test baseline on adversarial test set
python main.py --exp_type=test --exp_name=smallcnn --attack_type=fgsm --attack_args=1
# Adversarially train on training set
python main.py --exp_type=train --exp_name=smallcnn-adv --attack_type=fgsm --attack_args=1
# Test adversarially trained model on clean test set
python main.py --exp_type=test --exp_name=smallcnn-adv
# Test adversarially trained model on adversarial test set
python main.py --exp_type=test --exp_name=smallcnn-adv --attack_type=fgsm --attack_args=1

# For bird-or-bicycle, we need to change dataset_name, num_classes, and decrease batch_size (higher mem requirement due to large images)
# Train baseline
python main.py --exp_type=train --exp_name=smallcnn --dataset_name=bird_or_bicycle --num_classes=2 --batch_size=10
# Test baseline on clean test set
python main.py --exp_type=test --exp_name=smallcnn --dataset_name=bird_or_bicycle --num_classes=2 --batch_size=10
# Test baseline on adversarial test set
python main.py --exp_type=test --exp_name=smallcnn --attack_type=fgsm --attack_args=1 --dataset_name=bird_or_bicycle --num_classes=2 --batch_size=10
# Adversarially train on training set
python main.py --exp_type=train --exp_name=smallcnn-adv --attack_type=fgsm --attack_args=1 --dataset_name=bird_or_bicycle --num_classes=2 --batch_size=10
# Test adversarially trained model on clean test set
python main.py --exp_type=test --exp_name=smallcnn-adv
# Test adversarially trained model on adversarial test set
python main.py --exp_type=test --exp_name=smallcnn-adv --attack_type=fgsm --attack_args=1 --dataset_name=bird_or_bicycle --num_classes=2 --batch_size=10
