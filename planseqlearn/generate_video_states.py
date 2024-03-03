import mujoco_py
import argparse

import omegaconf 
from planseqlearn.environments.robosuite_dm_env import make_robosuite
from planseqlearn.environments.metaworld_dm_env import make_metaworld
from planseqlearn.environments.mopa_dm_env import make_mopa
from planseqlearn.environments.kitchen_dm_env import make_kitchen
import torch 
import numpy as np
from planseqlearn.psl.env_text_plans import *
import pickle

from planseqlearn.utils import make_video 

def make_env(cfg, is_eval, use_mp=False):
    if cfg.task_name.split("_", 1)[0] == "metaworld":
        env = make_metaworld(
            name=cfg.task_name.split("_", 1)[1],
            frame_stack=cfg.frame_stack,
            action_repeat=cfg.action_repeat,
            discount=cfg.discount,
            seed=cfg.seed,
            camera_name=cfg.camera_name,
            psl=cfg.psl,
            text_plan=cfg.text_plan,
            use_vision_pose_estimation=cfg.use_vision_pose_estimation,
            use_mp=use_mp,
        )
        inner_env = env._env._env._env._env._env
        mp_env = inner_env._env
    elif cfg.task_name.split("_", 1)[0] == "robosuite":
        env = make_robosuite(
            name=cfg.task_name.split("_", 1)[1],
            frame_stack=cfg.frame_stack,
            action_repeat=cfg.action_repeat,
            discount=cfg.discount,
            camera_name=cfg.camera_name,
            psl=cfg.psl,
            path_length=cfg.path_length,
            vertical_displacement=cfg.vertical_displacement,
            estimate_orientation=cfg.estimate_orientation,
            valid_obj_names=cfg.valid_obj_names,
            use_proprio=cfg.use_proprio,
            text_plan=cfg.text_plan,
            use_vision_pose_estimation=cfg.use_vision_pose_estimation,
            use_mp=use_mp,
        )
        inner_env = env._env._env._env._env._env
        mp_env = inner_env._env
    elif cfg.task_name.split("_", 1)[0] == "kitchen":
        env = make_kitchen(
            name=cfg.task_name.split("_", 1)[1],
            frame_stack=cfg.frame_stack,
            action_repeat=cfg.action_repeat,
            discount=cfg.discount,
            seed=cfg.seed,
            camera_name=cfg.camera_name,
            path_length=cfg.path_length,
            psl=cfg.psl,
            text_plan=cfg.text_plan,
            use_mp=use_mp,
        )
        inner_env = env._env._env._env._env._env._env
        mp_env = inner_env._env
    elif cfg.task_name.split("_", 1)[0] == "mopa":
        env = make_mopa(
            name=cfg.task_name.split("_", 1)[1],
            frame_stack=cfg.frame_stack,
            action_repeat=cfg.action_repeat,
            seed=cfg.seed,
            horizon=cfg.path_length,
            psl=cfg.psl,
            text_plan=cfg.text_plan,
            use_vision_pose_estimation=cfg.use_vision_pose_estimation,
            use_mp=use_mp,
        )
        inner_env = env._env._env._env._env._env
        mp_env = inner_env._env
    return env, inner_env, mp_env

def robosuite_gen_video(env_name, camera_name, suite, use_mp):
    # reset current hydra config if already parsed (but not passed in here)
    import hydra
    from hydra import compose, initialize
    from hydra.core.hydra_config import HydraConfig
    if HydraConfig.initialized():
        task = HydraConfig.get().runtime.choices['task']
        hydra.core.global_hydra.GlobalHydra.instance().clear()

    with initialize(config_path="./cfgs"):
        cfg = compose(config_name="train_config", overrides=[f"task={suite}_{env_name}", f"camera_name={camera_name}", "psl=True"])
    # create environment 
    agent = torch.load(f"planseqlearn/psl_policies/{suite}/{env_name}.pt")["agent"]
    env, inner_env, mp_env = make_env(cfg, is_eval=True, use_mp=use_mp)
    frames = []
    np.random.seed(0)
    o = env.reset()
    if use_mp:
        states = dict(
            qpos=mp_env.intermediate_qposes,
            qvel=mp_env.intermediate_qvels,
        )
        mp_env.intermediate_qposes = []
        mp_env.intermediate_qvels = []
        frames.extend(mp_env.intermediate_frames)
    else:
        states = dict(
            qpos=[inner_env.sim.data.qpos.copy()],
            qvel=[inner_env.sim.data.qvel.copy()],
        )
    num_success_steps = 5
    success_steps_ctr = 0
    with torch.no_grad():
        for _ in range(100):
            act = agent.act(o.observation, step=_, eval_mode=True)
            o = env.step(act)
            if use_mp:
                if len(mp_env.intermediate_qposes) > 0:
                    states['qpos'].extend(mp_env.intermediate_qposes)
                    states['qvel'].extend(mp_env.intermediate_qvels)
                    mp_env.intermediate_qposes = []
                    mp_env.intermediate_qvels = []
                    frames.extend(mp_env.intermediate_frames)
                    mp_env.intermediate_frames = []
            if suite == 'mopa':
                frames.append(env.get_vid_image())
            else:
                frames.append(env.get_image())
            states["qpos"].append(inner_env.sim.data.qpos.copy())
            states["qvel"].append(inner_env.sim.data.qvel.copy())
            if o.reward['success']:
                success_steps_ctr += 1
            if success_steps_ctr == num_success_steps:
                break
            if o.last():
                break
    print(o.reward)
    if not o.reward['success']:
        # write to a txt file the env name 
        with open('failed_envs.txt', 'a') as f:
            f.write(f"{env_name}\n")
    # assert o.reward['success'], f"Failed to complete task {env_name}"
    if use_mp and o.reward['success']:
        states["qpos"] = np.array(states["qpos"])
        states["qvel"] = np.array(states["qvel"])
        np.savez(f"states/{env_name}_{camera_name}_states.npz", **states)
    video_filename = f"{env_name}_{camera_name}.mp4"
    make_video(frames, "videos", video_filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, help='Name of the environment')
    parser.add_argument('--camera_name', type=str, help='Name of the environment')   
    parser.add_argument('--suite', type=str, default='robosuite', help='Type of environment')
    parser.add_argument('--use-mp', action='store_true', help='Use motion planner')
    args = parser.parse_args()
    robosuite_gen_video(args.env_name, args.camera_name, args.suite, args.use_mp)  