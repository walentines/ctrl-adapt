#!/bin/bash

# export PYTHONPATH=$PYTHONPATH:/bigdata/userhome/andrei.tarca/shared/MIRPR-proiectCONTROL-NET/TrainingControlnetNew/diffusers/src

export MODEL_DIR="stable-diffusion-v1-5/stable-diffusion-v1-5"
export OUTPUT_DIR="train_one_controlnet_depth_new_hope_it_works"
# export OUTPUT_DIR_IMAGES="/bigdata/userhome/ionut.serban/shared/MIRPR-proiectCONTROL-NET/TrainingControlnetNew/training_two_controlnet_seg_depth_adaptive_epsilon/output_validation" 
export CONTROLNET_MODEL_DIR_SEG="lllyasviel/sd-controlnet-seg"
export CONTROLNET_MODEL_DIR_DEPTH="lllyasviel/sd-controlnet-depth"

python Trainer2.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --resolution=512 \
 --learning_rate=2e-5 \
 --train_batch_size=1 \
 --gradient_accumulation_steps=4 \
 --validation_steps=2 \
 --num_train_epochs=50 \
 --checkpoints_total_limit=1 \
 --controlnet_model_name_or_path_seg=$CONTROLNET_MODEL_DIR_SEG \
 --controlnet_model_name_or_path_depth=$CONTROLNET_MODEL_DIR_DEPTH \
 --seed=10002