from copy import copy
import os
import random
# import sys
# sys.path.remove("/home/tarunc/Desktop/research/d4rl")
import time
import gym
import mujoco_py
import argparse
from PIL import ImageFont, ImageDraw, Image

from tqdm import tqdm
from planseqlearn.environments.robosuite_dm_env import make_robosuite
from planseqlearn.environments.metaworld_dm_env import make_metaworld
from planseqlearn.environments.mopa_dm_env import make_mopa
from planseqlearn.environments.kitchen_dm_env import make_kitchen
import torch
import numpy as np
from planseqlearn.psl.env_text_plans import *
import pickle
import robosuite as suite
import imageio
import cv2
import metaworld
import xml.etree.ElementTree as ET
from planseqlearn.nvisii_renderer.nvisii_renderer import NVISIIRenderer
from mopa_rl.config.default_configs import (
    LIFT_OBSTACLE_CONFIG,
    LIFT_CONFIG,
    ASSEMBLY_OBSTACLE_CONFIG,
    PUSHER_OBSTACLE_CONFIG,
)

from planseqlearn.utils import make_video
import mujoco

class SimWrapper():
    def __init__(self, model, data):
        self.model = model
        self.data = data
    def forward(self):
        pass

class ModelWrapper():
    def __init__(self, model):
        self.model = model
        
    def get_xml(self):
        # string representation of the xml
        mujoco.mj_saveLastXML('/tmp/temp.xml', self.model)
        xml = open('/tmp/temp.xml').read()
        xml_root = ET.fromstring(xml)
        for elem in xml_root.iter():
            for attr in elem.attrib:
                if attr == 'file':
                    menagerie_path = 'mujoco_menagerie'
                    elem.attrib[attr] = os.path.join(menagerie_path, 'rethink_robotics_sawyer/assets', elem.attrib[attr])
        xml = ET.tostring(xml_root, encoding='unicode')
        return xml
    def body_name2id(self, name):
        return name2id(self.model, "body", name)
def name2id(model, type_name, name):
    obj_id = mujoco.mj_name2id(
        model, mujoco.mju_str2Type(type_name.encode()), name.encode()
    )
    if obj_id < 0:
        raise ValueError('No {} with name "{}" exists.'.format(type_name, name))
    return obj_id
class DataWrapper():
    def __init__(self, data):
        self.data = data
    @property
    def qpos(self):
        return self.data.qpos
    @property
    def qvel(self):
        return self.data.qvel
    def get_geom_xpos(self, name):
        model = self.data.model
        geom_id = name2id(model, "geom", name)
        return self.data.geom_xpos[geom_id]
    def get_body_xpos(self, name):
        model = self.data.model
        body_id = name2id(model, "body", name)
        return self.data.model.body_pos[body_id]
    @property
    def body_xmat(self):
        quat = self.data.model.body_quat
        xmat = []
        for i in range(quat.shape[0]):
            xmat_ = np.zeros((9, 1))
            mujoco.mju_quat2Mat(xmat_,quat[i].reshape(-1, 1))
            xmat.append(xmat_)
        xmat = np.array(xmat)[:, :, 0]
        return xmat
class MenagerieEnv():
    def __init__(self):
        model = mujoco.MjModel.from_xml_path("mujoco_menagerie/rethink_robotics_sawyer/scene.xml")
        data = mujoco.MjData(model)
        self.sim = SimWrapper(ModelWrapper(model), DataWrapper(data))
        self.env_name = 'menagerie'
        self.data = data

    def reset(self):
        mujoco.mj_step(self.data.model, self.data)
    def step(self, action):
        pass

ENV_NAME_MAP = {
    "metaworld": {
        "assembly-v2": "MW-Assembly",
        "disassemble-v2": "MW-Disassemble",
        "bin-picking-v2": "MW-Bin-Picking",
        "hammer-v2": "MW-Hammer",
    },
    "kitchen": {
        "kitchen-slide-v0": "K-Slide",
        "kitchen-light-v0": "K-Light",
        "kitchen-tlb-v0": "K-Burner",
        "kitchen-microwave-v0": "K-Microwave",
        "kitchen-kettle-v0": "K-Kettle",
        "kitchen-ms5-v0": "K-MS5",
        "kitchen-ms10-v1": "K-MS10",
    },
    "mopa": {
        "SawyerAssemblyObstacle-v0": "OS-Assembly",
        "SawyerLiftObstacle-v0": "OS-Lift",
        "SawyerPushObstacle-v0": "OS-Push"
    }
}
    

def gen_video(env_name, camera_name, suite, clean):
    np.random.seed(0)
    random.seed(0)
    if suite == 'metaworld':
        # create environment
        mt = metaworld.MT1(env_name, seed=0)
        all_envs = {
            name: env_cls() for name, env_cls in mt.train_classes.items()
        }
        _, env = random.choice(list(all_envs.items()))
        task = random.choice(
                [task for task in mt.train_tasks if task.env_name == env_name]
            )
        env.set_task(task)
        
        def get_xml():
            import metaworld.envs as envs
            base_path = os.path.join(envs.__path__[0], "assets_v2")
            xml_root = ET.fromstring(env.sim.model.get_xml())
            # convert all ../ to absolute path
            for elem in xml_root.iter():
                for attr in elem.attrib:
                    if "../" in elem.attrib[attr]:
                        orig = copy(elem.attrib[attr].split("/")[0])
                        while "../" in elem.attrib[attr]:
                            elem.attrib[attr] = elem.attrib[attr].replace("../", "")
                        # extract first part of the path
                        first_part = elem.attrib[attr].split("/")[0]
                        if first_part == "objects":
                            # get the absolute path
                            abs_path = os.path.join(base_path, elem.attrib[attr])
                        elif first_part == 'scene':
                            abs_path = os.path.join(base_path, elem.attrib[attr])
                        elif first_part == 'scene_textures':
                            abs_path = os.path.join(base_path, 'scene', elem.attrib[attr])
                        elif first_part == 'textures':
                            abs_path = os.path.join(base_path, elem.attrib[attr])
                        else:
                            print(first_part, orig)
                        elem.attrib[attr] = abs_path
                    else:
                        first_part = elem.attrib[attr].split("/")[0]
                        if first_part == 'assets':
                            abs_path = os.path.join(base_path, 'objects', elem.attrib[attr])
                            elem.attrib[attr] = abs_path
                            
            xml = ET.tostring(xml_root, encoding='unicode')
            return xml
        env.get_modified_xml = get_xml
    elif suite == 'kitchen':
        env_kwargs = dict(
            dense=False,
            image_obs=True,
            action_scale=1,
            control_mode="end_effector",
            frame_skip=40,
            max_path_length=280,
        )
        from d4rl.kitchen.env_dict import ALL_KITCHEN_ENVIRONMENTS
        env = ALL_KITCHEN_ENVIRONMENTS[env_name](**env_kwargs)
    elif suite == 'mopa':
        if env_name == "SawyerLift-v0":
            config = LIFT_CONFIG
        elif env_name == "SawyerLiftObstacle-v0":
            config = LIFT_OBSTACLE_CONFIG
        elif env_name == "SawyerAssemblyObstacle-v0":
            config = ASSEMBLY_OBSTACLE_CONFIG
        elif env_name == "SawyerPushObstacle-v0":
            config = PUSHER_OBSTACLE_CONFIG
        config['max_episode_steps'] = 100
        env = gym.make(**config)
        
        def get_xml():
            import mopa_rl.env as envs
            base_path = os.path.join(envs.__path__[0], "assets")
            xml_root = ET.fromstring(env.sim.model.get_xml())
            # convert all ../ to absolute path
            for elem in xml_root.iter():
                for attr in elem.attrib:
                    if "../" in elem.attrib[attr]:
                        orig = copy(elem.attrib[attr].split("/")[0])
                        while "../" in elem.attrib[attr]:
                            elem.attrib[attr] = elem.attrib[attr].replace("../", "")
                        # extract first part of the path
                        first_part = elem.attrib[attr].split("/")[0]
                        if first_part == "meshes":
                            abs_path = os.path.join(base_path, elem.attrib[attr])
                        elif first_part == 'textures':
                            abs_path = os.path.join(base_path, elem.attrib[attr])
                        else:
                            print(first_part, orig)
                        elem.attrib[attr] = abs_path
                    else:
                        first_part = elem.attrib[attr].split("/")[0]
                        if first_part == 'assets':
                            abs_path = os.path.join(base_path, 'objects', elem.attrib[attr])
                            elem.attrib[attr] = abs_path
                        if './' in elem.attrib[attr]:
                            elem.attrib[attr] = elem.attrib[attr].replace('./', '')
                            first_part = elem.attrib[attr].split("/")[0]
                            if first_part == 'common':
                                abs_path = os.path.join(base_path, 'xml', elem.attrib[attr])
                                elem.attrib[attr] = abs_path
                            
            xml = ET.tostring(xml_root, encoding='unicode')
            return xml
        env.get_modified_xml = get_xml
        env.env_name = env_name
    frame_env_name = ENV_NAME_MAP[suite][env_name]
    env.reset()
    font_path = 'planseqlearn/RobotoSlab-Regular.ttf'
    font = ImageFont.truetype(font_path, 45)
    font_path = 'planseqlearn/RobotoSlab-Bold.ttf'
    font2 = ImageFont.truetype(font_path, 60)
    cfg = {
        "img_path": "images/",
        "width": 1920, #1980, # used to be 1980 by 1080
        "height": 1080,
        "spp": 512,
        "use_noise": False,
        "debug_mode": False,
        "video_mode": False,
        "video_path": "videos/",
        "video_name": "robosuite_video_0.mp4",
        "video_fps": 30,
        "verbose": 1,
        "vision_modalities": None
    }
    renderer = NVISIIRenderer(env,
                              **cfg)
    states = np.load(f"states/{env_name}_{camera_name}_states.npz")
    print(renderer.components.keys())
    renderer.reset()
    # clear images folder
    for file in os.listdir("images"):
        os.remove(os.path.join("images", file))
    # load mp idxs
    mp_idxs = list(np.load(f"mp_idxs/{env_name}_{camera_name}_mp_idxs.npz")['mp_idxs'])
    for step in tqdm(range(states["qpos"].shape[0])):
        qpos = states["qpos"][step]
        qvel = states["qvel"][step]
        env.sim.data.qpos[:] = qpos
        env.sim.data.qvel[:] = qvel
        env.sim.forward()
        renderer.update(is_mp=(step in mp_idxs) and not clean)
        renderer.render()
        if not clean:
            imfile = sorted(os.listdir("images"))[-1]
            frame = cv2.imread(f"images/{imfile}")
            img_pil = Image.fromarray(frame)
            x = 0.1 * cfg['width']
            y = 0.05 * cfg['height']
            draw = ImageDraw.Draw(img_pil)
            draw.text((x, y), frame_env_name, font=font2, fill = (0, 0, 0, 1))
            if step in mp_idxs:
                draw.ellipse(
                    (0.065 * cfg['width'], 0.14 * cfg['height'] - 5, 0.08 * cfg['width'], 0.17 * cfg['height'] - 5),
                    fill = (0, 0, 255, 1),
                )
                draw.text((0.1 * cfg["width"], 0.12 * cfg["height"]), "Motion Planner", font=font, fill = (0, 0, 0, 1))
                draw.text((0.1 * cfg["width"], 0.17 * cfg["height"]), "Local Policy", font=font, fill = (128, 128, 128, 1))
            else:
                draw.ellipse(
                    (0.065 * cfg['width'], 0.19 * cfg['height'] - 5, 0.08 * cfg['width'], 0.22 * cfg['height'] - 5),
                    fill = (0, 0, 255, 1),
                )
                draw.text((0.1 * cfg["width"], 0.12 * cfg["height"]), "Motion Planner", font=font, fill = (128, 128, 128, 1))
                draw.text((0.1 * cfg["width"], 0.17 * cfg["height"]), "Local Policy", font=font, fill = (0, 0, 0, 1))
            frame = np.array(img_pil)
            cv2.imwrite(f"images/{imfile}", frame)
    # load png files from images folder
    frames = []
    for idx in range(1, len(os.listdir("images"))):
        im_path = os.path.join("images", "image_{}.png".format(idx))
        frames.append(cv2.imread(im_path))
        if (idx - 1) not in mp_idxs:
            frames.append(cv2.imread(im_path))
    video_filename = f"{env_name}_{camera_name}.mp4"
    make_video(frames, "rendered_videos", video_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, help="Name of the environment")
    parser.add_argument("--camera_name", type=str, help="Camera name")
    parser.add_argument("--suite", type=str, help="Name of the suite")
    parser.add_argument("--clean", action="store_true", help="Generate clean video")
    args = parser.parse_args()
    gen_video(args.env_name, args.camera_name, args.suite, args.clean)
