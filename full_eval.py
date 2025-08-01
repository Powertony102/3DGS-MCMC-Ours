#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
from argparse import ArgumentParser

mipnerf360_outdoor_scenes = ["bicycle", "flowers", "garden", "stump", "treehill"]
mipnerf360_indoor_scenes = ["room", "counter", "kitchen", "bonsai"]
tanks_and_temples_scenes = ["truck", "train"]
deep_blending_scenes = ["drjohnson", "playroom"]

# Maximum number of Gaussians for each scene
MAX_N_GAUSSIAN = {
    "bicycle": 5033049, 
    "garden": 4184336, 
    "stump": 4303916, 
    "flowers": 2950914, 
    "treehill": 3310948, 
    "room": 1346463, 
    "kitchen": 1634800, 
    "counter": 1105841, 
    "bonsai": 1095842, 
    "drjohnson": 3231872, 
    "playroom": 1913528, 
    "train": 1108608, 
    "truck": 2101586, 
}

parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--skip_rendering", action="store_true")
parser.add_argument("--skip_metrics", action="store_true")
parser.add_argument("--output_path", default="./eval")
parser.add_argument("--wandb_project", type=str, default="3dgs-mcmc-eval", help="Wandb project name for evaluation")
parser.add_argument("--wandb_entity", type=str, default=None, help="Wandb entity/username")
parser.add_argument("--disable_wandb", action="store_true", help="Disable wandb logging")
args, _ = parser.parse_known_args()

all_scenes = []
all_scenes.extend(mipnerf360_outdoor_scenes)
all_scenes.extend(mipnerf360_indoor_scenes)
all_scenes.extend(tanks_and_temples_scenes)
all_scenes.extend(deep_blending_scenes)

if not args.skip_training or not args.skip_rendering:
    parser.add_argument('--mipnerf360', "-m360", default="/home/jovyan/work/gs_compression/HAC-main/data/mipnerf360", type=str)
    parser.add_argument("--tanksandtemples", "-tat", default="/home/jovyan/shared/xinzeli/tandt_db/tandt", type=str)
    parser.add_argument("--deepblending", "-db", default="/home/jovyan/shared/xinzeli/tandt_db/db", type=str)
    args = parser.parse_args()

if not args.skip_training:
    # Build wandb arguments
    wandb_args = ""
    if not args.disable_wandb:
        wandb_args += f" --wandb_project {args.wandb_project}"
        if args.wandb_entity:
            wandb_args += f" --wandb_entity {args.wandb_entity}"
    else:
        wandb_args += " --disable_wandb"
    
    common_args = " --quiet --eval --test_iterations -1 " + wandb_args
    """
    for scene in mipnerf360_outdoor_scenes:
        source = args.mipnerf360 + "/" + scene
        cap_max = MAX_N_GAUSSIAN.get(scene, 1000000)  # Default fallback
        os.system("python train.py -s " + source + " -i images_4 -m " + args.output_path + "/" + scene + common_args + f" --cap_max {cap_max}")
    for scene in mipnerf360_indoor_scenes:
        source = args.mipnerf360 + "/" + scene
        cap_max = MAX_N_GAUSSIAN.get(scene, 1000000)  # Default fallback
        os.system("python train.py -s " + source + " -i images_2 -m " + args.output_path + "/" + scene + common_args + f" --cap_max {cap_max}")
    for scene in tanks_and_temples_scenes:
        source = args.tanksandtemples + "/" + scene
        cap_max = MAX_N_GAUSSIAN.get(scene, 1000000)  # Default fallback
        os.system("python train.py -s " + source + " -m " + args.output_path + "/" + scene + common_args + f" --cap_max {cap_max}")
    """
    for scene in deep_blending_scenes:
        source = args.deepblending + "/" + scene
        cap_max = MAX_N_GAUSSIAN.get(scene, 1000000)  # Default fallback
        os.system("python train.py -s " + source + " -m " + args.output_path + "/" + scene + common_args + f" --cap_max {cap_max}")

if not args.skip_rendering:
    all_sources = []
    """
    for scene in mipnerf360_outdoor_scenes:
        all_sources.append(args.mipnerf360 + "/" + scene)
    for scene in mipnerf360_indoor_scenes:
        all_sources.append(args.mipnerf360 + "/" + scene)
    for scene in tanks_and_temples_scenes:
        all_sources.append(args.tanksandtemples + "/" + scene)
    """
    for scene in deep_blending_scenes:
        all_sources.append(args.deepblending + "/" + scene)

    common_args = " --quiet --eval --skip_train"
    for scene, source in zip(all_scenes, all_sources):
        os.system("python render.py --iteration 7000 -s " + source + " -m " + args.output_path + "/" + scene + common_args)
        os.system("python render.py --iteration 30000 -s " + source + " -m " + args.output_path + "/" + scene + common_args)

if not args.skip_metrics:
    scenes_string = ""
    for scene in all_scenes:
        scenes_string += "\"" + args.output_path + "/" + scene + "\" "

    os.system("python metrics.py -m " + scenes_string)