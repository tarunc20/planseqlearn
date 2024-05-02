from copy import copy
import os
import random
import time
import gym
import mujoco_py
import argparse
import nvisii 
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
# REPLACE LINE WIDTHS WHEN LAUNCHIGN FULL
IMWIDTH = 1920#240
IMHEIGHT = 1080
def get_geom_segmentation(geom_name, renderer):
    segmentation_array = nvisii.render_data(
        width=IMWIDTH,
        height=IMHEIGHT,
        start_frame=0,
        frame_count=1,
        bounce=int(0),
        options="entity_id",
        seed=1,
    )
    segmentation_array = np.flipud(
        np.array(segmentation_array).reshape(IMHEIGHT, IMWIDTH, 4)[:, :, 0].astype(np.uint8)
    )
    all_entity_ids = set()
    all_entity_ids.add(renderer.components[geom_name].element_id)
    for i in range(len(segmentation_array)):
            for j in range(len(segmentation_array[0])):
                if segmentation_array[i][j] in renderer.parser.entity_id_class_mapping:
                    segmentation_array[i][j] = renderer.parser.entity_id_class_mapping[segmentation_array[i][j]]
                else:
                    segmentation_array[i][j] = 254
    for id in list(np.unique(segmentation_array)):
        if id not in all_entity_ids:
            segmentation_array[segmentation_array == id] = 0 
        else:
            segmentation_array[segmentation_array == id] = 1
    import matplotlib.pyplot as plt 
    plt.imshow(segmentation_array)
    plt.savefig("micro_seg.png")
    print("Array sum ", np.sum(segmentation_array))
    return segmentation_array

def gen_video(env_name, camera_name, suite):
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
    cfg = {
        "img_path": "images_2/",
        "width": IMWIDTH, # used to be 1980 by 1080
        "height": IMHEIGHT,
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
    states = np.load(f"states/{env_name}_{camera_name}_states_0.npz")
    mp_idxs = list(np.load("mp_idxs/kitchen-ms5-v0_wrist_mp_idxs_0.npz")["mp_idxs"])
    print([mp_idxs[i] for i in range(len(mp_idxs) - 1) if mp_idxs[i + 1] > mp_idxs[i] + 1])
    renderer.reset()
    # clear images folder
    for file in os.listdir("images_2"):
        os.remove(os.path.join("images_2", file))

    orig_cam_at = np.array((-.2, 1.5, 1.5))
    kettle_cam_at = np.array((-.2, 1.5, 0.75))
    orig_cam_pose = np.array(((-0.5, -1.5, 3.5)))
    kettle_cam_pose = np.array((-0.2, -0.1, 2.5))
    
    orig_to_kettle_at = [orig_cam_at + t * (kettle_cam_at - orig_cam_at) for t in np.linspace(0, 1, 20)]
    orig_to_kettle_pose = [orig_cam_pose + t * (kettle_cam_pose - orig_cam_pose) for t in np.linspace(0, 1, 20)]

    PLAN_TEXT = "Plan: [('microwave', 'grasp'), ('kettle', 'grasp'), ...]"
    # test stuff
    font_path = 'planseqlearn/arial.ttf'
    font = ImageFont.truetype(font_path, 22) # should be 22
    boldfont = ImageFont.truetype(font_path, 30) # should be 30 
    firstfont = ImageFont.truetype(font_path, 42) # should be like 42 or osmething
    font_path = 'planseqlearn/Arial Bold.ttf'
    icon_frame = Image.open("icon.png")
    icon_frame = icon_frame.resize((160, 100))
    llmfont = ImageFont.truetype(font_path, 70)
    # render starting 
    for i in range(140): # used to be 115
        renderer.render()
        
    # render first image (for displaying plan)
    for i in range(15):
        qpos = states["qpos"][0]
        qvel = states["qvel"][0]
        env.sim.data.qpos[:] = qpos
        env.sim.data.qvel[:] = qvel
        env.sim.forward()
        renderer.update(is_mp=False)
        renderer.render()
        # load previous image 
        
    # pause for plan
    for _ in range(15):
        renderer.render()
        # load previous image 
        

    microwave_cam_pose = np.array(((-0.7, -0.35, 2.0)))
    orig_to_microwave = [orig_cam_pose + (microwave_cam_pose - orig_cam_pose) * t for t in np.linspace(0, 1, 20)] # should be 20
    # change camera viewpoint and write segment microwave handle
    for i, pose in enumerate(orig_to_microwave):
        renderer._camera_configuration(
            at_vec=nvisii.vec3(-.2, 1.5, 1.5),
            up_vec=nvisii.vec3(0, 0, 1),
            eye_vec=nvisii.vec3(*pose),
            quat=nvisii.quat(-1, 0, 0, 0),
        )
        renderer.update(is_mp=False)
        
        renderer.render()
    
    # change color of microwave handle 
    orig_microwave_color = np.array((
        renderer._init_component_colors['mchandle'].x,
        renderer._init_component_colors['mchandle'].y,
        renderer._init_component_colors['mchandle'].z
    ))
    blue_color = np.array([0., 1.0, 1.0])
    gray_to_blue = [orig_microwave_color + (blue_color - orig_microwave_color) * t for t in np.linspace(0, 1, 15)]
    renderer.components['mchandle'].obj.get_material().clear_base_color_texture()
    # pause segment microwave handle
    for i in range(10):
        renderer.render()
    # # erase segment 
    for i, color in enumerate(gray_to_blue):
        renderer.components['mchandle'].obj.get_material().set_base_color(nvisii.vec3(*color))
        renderer.render()

    # write estimate pose 
    for i in range(56):
        renderer.render()

    # # # write pose
    for i in range(20):
        renderer.render()

    # reverse from microwave 
    for pose in orig_to_microwave[::-1]:
        renderer._camera_configuration(
            at_vec=nvisii.vec3(-.2, 1.5, 1.5),
            up_vec=nvisii.vec3(0, 0, 1),
            eye_vec=nvisii.vec3(*pose),
            quat=nvisii.quat(-1, 0, 0, 0),
        )
        renderer.update(is_mp=False)
        
        renderer.render()

    # # pause pose
    for _ in range(5):
        renderer.render()

    # write motion planner text 
    for _ in range(10):
        renderer.render()
        

    # microwave mp (used to be 0, 40)
    for step in tqdm(range(0, 50)):
        qpos = states["qpos"][step]
        qvel = states["qvel"][step]
        env.sim.data.qpos[:] = qpos
        env.sim.data.qvel[:] = qvel
        env.sim.forward()
        renderer.update(is_mp=False)
        
        renderer.render()
        # # write motion planning 

    for _ in range(10):
        renderer.render()

    # microwave low level (used to be 40, 51)
    for step in tqdm(range(50, 61)):
        qpos = states["qpos"][step]
        qvel = states["qvel"][step]
        env.sim.data.qpos[:] = qpos
        env.sim.data.qvel[:] = qvel
        env.sim.forward()
        renderer.update(is_mp=False)
        #renderer.components['mchandle'].obj.get_material().set_base_color(nvisii.vec3(*blue_color))
        renderer.render()

    # write segment kettle and pause 
    for i in range(15):
        renderer.render()
        

    for at, pose in zip(orig_to_kettle_at, orig_to_kettle_pose):
        renderer._camera_configuration(
            at_vec=nvisii.vec3(*at),
            up_vec=nvisii.vec3(0, 0, 1),
            eye_vec=nvisii.vec3(*pose),
            quat=nvisii.quat(-1, 0, 0, 0),
        )
        renderer.render()

    #kettle_seg = get_geom_segmentation("kettleroot0", renderer)
    kettle_obj = "kettletop"
    orig_kettle_color = np.array((
        renderer._init_component_colors[kettle_obj].x,
        renderer._init_component_colors[kettle_obj].y,
        renderer._init_component_colors[kettle_obj].z
    ))
    renderer.components[kettle_obj].obj.get_material().clear_base_color_texture()
    # change color and write segment kettle handle 
    for color in gray_to_blue:
        renderer.components[kettle_obj].obj.get_material().set_base_color(nvisii.vec3(*color))
        renderer.render()
        
    # write estimate pose 

    for i in range(56):
        renderer.render()

        
    for i in range(11):
        renderer.render()

    # pause written pose 
    for i in range(5):
        renderer.render()
        
    # reset base color texture of thing 
    renderer.components[kettle_obj].obj.get_material().clear_base_color_texture()
    texture = nvisii.texture.create_from_file(name="new_kettle", path="/home/tarunc/Desktop/research/planseqlearn/d4rl/d4rl/kitchen/adept_models/kitchen/textures/wood1.png")
    renderer.components[kettle_obj].obj.get_material().set_base_color_texture(texture)
    # create new texture again 
    for at, pose in zip(orig_to_kettle_at[::-1], orig_to_kettle_pose[::-1]):
        renderer._camera_configuration(
            at_vec=nvisii.vec3(*at),
            up_vec=nvisii.vec3(0, 0, 1),
            eye_vec=nvisii.vec3(*pose),
            quat=nvisii.quat(-1, 0, 0, 0),
        )
        renderer.update(is_mp=False)

        renderer.render()
        
    # # write motion planner
    for i in range(11):
        renderer.render()
        
    # # kettle mp 
    for step in tqdm(range(61, 111)):
        qpos = states["qpos"][step]
        qvel = states["qvel"][step]
        env.sim.data.qpos[:] = qpos
        env.sim.data.qvel[:] = qvel
        env.sim.forward()
        renderer.update(is_mp=False)

        renderer.render()

    # # write low level policy
    for i in range(11):
        renderer.render()
        
    
    # # # kettle low level (used to be 91, 99)
    for step in tqdm(range(111, 121)):
        qpos = states["qpos"][step]
        qvel = states["qvel"][step]
        env.sim.data.qpos[:] = qpos
        env.sim.data.qvel[:] = qvel
        env.sim.forward()
        renderer.update(is_mp=False)

        renderer.render()
       

    # write sequence remaining stages
    for i in range(25):
        renderer.render()
        
        
    for step in tqdm(range(121, 174)):
        qpos = states["qpos"][step]
        qvel = states["qvel"][step]
        env.sim.data.qpos[:] = qpos
        env.sim.data.qvel[:] = qvel
        env.sim.forward()
        renderer.update(is_mp=False)
        renderer.render()
        
    
    for step in tqdm(range(174, 229)):
        qpos = states["qpos"][step]
        qvel = states["qvel"][step]
        env.sim.data.qpos[:] = qpos
        env.sim.data.qvel[:] = qvel
        env.sim.forward()
        renderer.update(is_mp=False)
        renderer.render()
        
    
    for step in tqdm(range(229, len(states["qpos"]))):
        qpos = states["qpos"][step]
        qvel = states["qvel"][step]
        env.sim.data.qpos[:] = qpos
        env.sim.data.qvel[:] = qvel
        env.sim.forward()
        renderer.update(is_mp=False)
        renderer.render()
    

    frames = []
    for idx in range(1, len(os.listdir("images_mod"))):
        im_path = os.path.join("images_mod", "image_{}.png".format(idx))
        frames.append(cv2.imread(im_path))
        # if (idx - 1) not in mp_idxs:
        #     frames.append(cv2.imread(im_path))
    video_filename = f"{env_name}_{camera_name}.mp4"
    make_video(frames, "rendered_videos", video_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, help="Name of the environment")
    parser.add_argument("--camera_name", type=str, help="Camera name")
    parser.add_argument("--suite", type=str, help="Name of the suite")
    parser.add_argument("--clean", action="store_true", help="Generate clean video")
    args = parser.parse_args()
    gen_video(args.env_name, args.camera_name, args.suite)
