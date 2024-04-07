import colorsys
import os

import cv2
import matplotlib.cm as cm
from metaworld.envs.mujoco.mujoco_env import MujocoEnv
from mopa_rl.env.base import BaseEnv
import numpy as np
import nvisii
import open3d as o3d

import robosuite as suite
import robosuite.renderers.nvisii.nvisii_utils as utils
from robosuite.renderers.base import Renderer
from planseqlearn.nvisii_renderer.parser_nvisii import Parser
from robosuite.utils import transform_utils as T
from robosuite.utils.mjcf_utils import xml_path_completion
from robosuite.utils.transform_utils import mat2quat
from robosuite.wrappers import Wrapper

np.set_printoptions(threshold=np.inf)


class NVISIIRenderer(Renderer):
    def __init__(
        self,
        env,
        img_path="images/",
        width=500,
        height=500,
        spp=256,
        use_noise=False,
        debug_mode=False,
        video_mode=False,
        video_path="videos/",
        video_name="robosuite_video_0.mp4",
        video_fps=60,
        verbose=1,
        vision_modalities=None,
    ):
        """
        Initializes the nvisii wrapper. Wrapping any MuJoCo environment in this
        wrapper will use the NVISII wrapper for rendering.

        Args:
            env (MujocoEnv instance): The environment to wrap.

            img_path (string): Path to images.

            width (int, optional): Width of the rendered image. Defaults to 500.

            height (int, optional): Height of the rendered image. Defaults to 500.

            spp (int, optional): Sample-per-pixel for each image. Larger spp will result
                                 in higher quality images but will take more time to render
                                 each image. Higher quality images typically use an spp of
                                 around 512.

            use_noise (bool, optional): Use noise or denoise. Deafults to false.

            debug_mode (bool, optional): Use debug mode for nvisii. Deafults to false.

            video_mode (bool, optional): By deafult, the NVISII wrapper saves the results as
                                         images. If video_mode is set to true, a video is
                                         produced and will be stored in the directory defined
                                         by video_path. Defaults to false.

            video_path (string, optional): Path to store the video. Required if video_mode is
                                           set to true. Defaults to 'videos/'.

            video_name (string, optional): Name for the file for the video. Defaults to
                                           'robosuite_video_0.mp4'.

            video_fps (int, optional): Frames per second for video. Defaults to 60.

            verbose (int, optional): If verbose is set to 1, the wrapper will print the image
                                     number for each image rendered. If verbose is set to 0,
                                     nothing will be printed. Defaults to 1.

            vision_modalities (string, optional): Options to render image with different ground truths
                                              for NVISII. Options include "normal", "texture_coordinates",
                                              "position", "depth".
        """

        super().__init__(env, renderer_type="nvisii")

        self.env = env
        self.img_path = img_path
        self.width = width
        self.height = height
        self.spp = spp
        self.use_noise = use_noise

        self.video_mode = video_mode
        self.video_path = video_path
        self.video_name = video_name
        self.video_fps = video_fps

        self.verbose = verbose
        self.vision_modalities = vision_modalities

        self.img_cntr = 0
        if issubclass(type(self.env), MujocoEnv):
            self.env_type = 'metaworld'
        elif hasattr(self.env, 'env') or hasattr(self.env, 'use_target_robot_indicator'):
            self.env_type = "mopa"
            self.env_name = self.env.env_name
            # if issubclass(type(env.env), BaseEnv):
            #     self.env_type = 'mopa'
            #     self.env_name = self.env.env_name
        elif hasattr(self.env, 'env_name'):
            if self.env.env_name == 'menagerie':
                self.env_type = 'menagerie'
        else:
            self.env_type = 'kitchen'
            self.light_set = False 

        # enable interactive mode when debugging
        if debug_mode:
            nvisii.initialize_interactive()
        else:
            nvisii.initialize(headless=True)

        # add denoiser to nvisii if not using noise
        if not use_noise:
            nvisii.configure_denoiser()
            nvisii.enable_denoiser()
            nvisii.configure_denoiser(True, True, False)

        if not os.path.exists(img_path):
            os.makedirs(img_path)

        if video_mode:
            if not os.path.exists(video_path):
                os.makedirs(video_path)
            self.video = cv2.VideoWriter(
                video_path + video_name, cv2.VideoWriter_fourcc(*"MP4V"), video_fps, (self.width, self.height)
            )
            print(f"video mode enabled")

        if vision_modalities is None:
            nvisii.sample_pixel_area(x_sample_interval=(0.0, 1.0), y_sample_interval=(0.0, 1.0))
        else:
            nvisii.sample_pixel_area(x_sample_interval=(0.5, 0.5), y_sample_interval=(0.5, 0.5))
        # keep track of initial component colors 
        self._init_component_colors = {}
        self._init_nvisii_components()

    def _init_nvisii_components(self):
        self._init_lighting()
        self._init_floor(image="plywood-4k.jpg")
        self._init_walls(image="plaster-wall-4k.jpg")
        self._init_camera()

        self._load()

    def _init_lighting(self):
        # Intiailizes the lighting
        # self.light_1 = nvisii.entity.create(
        #     name="light",
        #     mesh=nvisii.mesh.create_sphere("light"),
        #     transform=nvisii.transform.create("light"),
        # )

        # self.light_1.set_light(nvisii.light.create("light"))

        # self.light_1.get_light().set_intensity(150)  # intensity of the light
        # self.light_1.get_transform().set_scale(nvisii.vec3(0.3))  # scale the light down
        # self.light_1.get_transform().set_position(nvisii.vec3(-3, 3, 7)) # used to be 3, 3, 4  # sets the position of the light
        if self.env_type == "kitchen":
            self.light_2 = nvisii.entity.create(
                name="light2",
                mesh=nvisii.mesh.create_sphere("light2"),
                transform=nvisii.transform.create("light2"),
            )

            self.light_2.set_light(nvisii.light.create("light2"))

            self.light_2.get_light().set_intensity(150)  # intensity of the light
            self.light_2.get_transform().set_scale(nvisii.vec3(0.3))  # scale the light down
            self.light_2.get_transform().set_position(nvisii.vec3(0, 0, 7)) # used to be 3, 3, 4  # sets the position of the light
        elif self.env_type == "metaworld":
            self.light_2 = nvisii.entity.create(
                name="light2",
                mesh=nvisii.mesh.create_sphere("light2"),
                transform=nvisii.transform.create("light2"),
            )

            self.light_2.set_light(nvisii.light.create("light2"))

            self.light_2.get_light().set_intensity(150)  # intensity of the light
            self.light_2.get_transform().set_scale(nvisii.vec3(0.3))  # scale the light down
            self.light_2.get_transform().set_position(nvisii.vec3(0, 0, 5.5)) # used to be 3, 3, 4  # sets the position of the light
        elif self.env_type == "mopa":
            self.light_2 = nvisii.entity.create(
                name="light2",
                mesh=nvisii.mesh.create_sphere("light2"),
                transform=nvisii.transform.create("light2"),
            )

            self.light_2.set_light(nvisii.light.create("light2"))

            self.light_2.get_light().set_intensity(100)  # intensity of the light
            self.light_2.get_transform().set_scale(nvisii.vec3(0.2))  # scale the light down
            self.light_2.get_transform().set_position(nvisii.vec3(0, 0, 5.5)) # used to be 3, 3, 4  # sets the position of the light

    def _init_floor(self, image):
        """
        Intiailizes the floor

        Args:
            image (string): String for the file to use as an image for the floor

        """
        floor_mesh = nvisii.mesh.create_plane(name="plane", size=nvisii.vec2(3, 3))

        floor_entity = nvisii.entity.create(
            name="floor",
            mesh=floor_mesh,
            material=nvisii.material.create("plane"),
            transform=nvisii.transform.create("plane"),
        )
        floor_entity.get_transform().set_scale(nvisii.vec3(1))
        if self.env_type == 'metaworld':
            height = -1
        else:
            height = 0
        floor_entity.get_transform().set_position(nvisii.vec3(0, 0, height))

        texture_image = xml_path_completion("textures/" + image)
        texture = nvisii.texture.create_from_file(name="floor_texture", path=texture_image)

        floor_entity.get_material().set_base_color_texture(texture)
        floor_entity.get_material().set_roughness(0.4)
        floor_entity.get_material().set_specular(0)

    def _init_walls(self, image):
        """
        Intiailizes the walls

        Args:
            image (string): String for the file to use as an image for the walls
        """
        texture = nvisii.texture.create_from_file(name="wall_texture", path="wall.jpg")
        if self.env_type == "metaworld" or self.env_type == "mopa":
            wall1_entity = nvisii.entity.create(
                name="wall1",
                mesh=nvisii.mesh.create_box(name="wall1", size=nvisii.vec3(2, 10, 10)),
                transform=nvisii.transform.create("wall1"),
                material=nvisii.material.create("wall1"),
            )
            wall1_entity.get_transform().set_position(nvisii.vec3(0, 2, 0))
            wall1_entity.get_transform().set_rotation(nvisii.quat(0, 1, 0, 0))
            wall1_entity.get_material().set_base_color_texture(texture)

            wall2_entity = nvisii.entity.create(
                name="wall2",
                mesh=nvisii.mesh.create_box(name="wall2", size=nvisii.vec3(2, 10, 10)),
                transform=nvisii.transform.create("wall2"),
                material=nvisii.material.create("wall2"),
            )
            wall2_entity.get_transform().set_position(nvisii.vec3(0, -2, 0))
            wall2_entity.get_transform().set_rotation(nvisii.quat(0, 1, 0, 0))
            wall2_entity.get_material().set_base_color_texture(texture)

            wall3_entity = nvisii.entity.create(
                name="wall3",
                mesh=nvisii.mesh.create_box(name="wall3", size=nvisii.vec3(10, 2, 10)),
                transform=nvisii.transform.create("wall3"),
                material=nvisii.material.create("wall3"),
            )
            wall3_entity.get_transform().set_position(nvisii.vec3(3, 0, 0))
            wall3_entity.get_transform().set_rotation(nvisii.quat(0, 1, 0, 0))
            wall3_entity.get_material().set_base_color_texture(texture)
        elif self.env_type == "kitchen":
            wall1_entity = nvisii.entity.create(
                name="wall1",
                mesh=nvisii.mesh.create_box(name="wall1", size=nvisii.vec3(2, 10, 10)),
                transform=nvisii.transform.create("wall1"),
                material=nvisii.material.create("wall1"),
            )
            wall1_entity.get_transform().set_position(nvisii.vec3(5, 0, 0))
            wall1_entity.get_transform().set_rotation(nvisii.quat(0, 1, 0, 0))
            wall1_entity.get_material().set_base_color_texture(texture)

            wall2_entity = nvisii.entity.create(
                name="wall2",
                mesh=nvisii.mesh.create_box(name="wall2", size=nvisii.vec3(2, 10, 10)),
                transform=nvisii.transform.create("wall2"),
                material=nvisii.material.create("wall2"),
            )
            wall2_entity.get_transform().set_position(nvisii.vec3(-5, 0, 0))
            wall2_entity.get_transform().set_rotation(nvisii.quat(0, 1, 0, 0))
            wall2_entity.get_material().set_base_color_texture(texture)

            wall3_entity = nvisii.entity.create(
                name="wall3",
                mesh=nvisii.mesh.create_box(name="wall3", size=nvisii.vec3(10, 2, 10)),
                transform=nvisii.transform.create("wall3"),
                material=nvisii.material.create("wall3"),
            )
            wall3_entity.get_transform().set_position(nvisii.vec3(0, 5, 0))
            wall3_entity.get_transform().set_rotation(nvisii.quat(0, 1, 0, 0))
            wall3_entity.get_material().set_base_color_texture(texture)

        texture_image = xml_path_completion("textures/" + image)
        # texture = nvisii.texture.create_from_file(name="wall_texture", path=texture_image)

        # for wall in self.env.model.mujoco_arena.worldbody.findall("./geom[@material='walls_mat']"):

        #     name = wall.get("name")
        #     size = [float(x) for x in wall.get("size").split(" ")]

        #     pos, quat = self._get_orientation_geom(name)

        #     wall_entity = nvisii.entity.create(
        #         name=name,
        #         mesh=nvisii.mesh.create_box(name=name, size=nvisii.vec3(size[0], size[1], size[2])),
        #         transform=nvisii.transform.create(name),
        #         material=nvisii.material.create(name),
        #     )

        #     wall_entity.get_transform().set_position(nvisii.vec3(pos[0], pos[1], pos[2]))

        #     wall_entity.get_transform().set_rotation(nvisii.quat(quat[0], quat[1], quat[2], quat[3]))

        #     wall_entity.get_material().set_base_color_texture(texture)

    def _init_camera(self):
        """
        Intializes the camera for the NVISII renderer
        """

        # intializes the camera
        self.camera = nvisii.entity.create(
            name="camera",
            transform=nvisii.transform.create("camera_transform"),
        )

        self.camera.set_camera(
            nvisii.camera.create_from_fov(
                name="camera_camera", field_of_view=1, aspect=float(self.width) / float(self.height)
            )
        )

        # Sets the primary camera of the renderer to the camera entity
        nvisii.set_camera_entity(self.camera)
        
        # for kitchen:
        if self.env_type == 'kitchen':
            self._camera_configuration(
                at_vec=nvisii.vec3(-.2, 1.5, 1.5),
                up_vec=nvisii.vec3(0, 0, 1),
                eye_vec=nvisii.vec3(-0.5, -1.5, 3.5),
                quat=nvisii.quat(-1, 0, 0, 0),
            )
        elif self.env_type == 'metaworld':
            # metaworld
            self.camera.get_transform().look_at(
                at = (0,0,0.1),
                up = (0,0,1),
                eye = (1,1,0.5),
            )
        elif self.env_type == 'menagerie':
            self._camera_configuration(
                at_vec=nvisii.vec3(0, 0, 0),
                up_vec=nvisii.vec3(0, 0, 1),
                eye_vec=nvisii.vec3(1, 1, 1),
                quat=nvisii.quat(-1, 0, 0, 0),
            )
        else:
            # mopa
            if self.env.env_name == 'SawyerLiftObstacle-v0':
                self.camera.get_transform().look_at(
                    at = (0.5,0,1),
                    up = (0,0,1),
                    eye = (1.1,0,2.5), # used to be 1.75, 0, 2.75
                )
            elif self.env.env_name == "SawyerPushObstacle-v0":
                self.camera.get_transform().look_at(
                    at = (0,0,1),
                    up = (0,0,1),
                    eye = (2.0,0.0,2.0),
                )
            else:
                self.camera.get_transform().look_at(
                    at = (0,0,1),
                    up = (0,0,1),
                    eye = (1.7,0,2.0),
                )

        # Environment configuration
        self._dome_light_intensity = 1
        nvisii.set_dome_light_intensity(self._dome_light_intensity)
        nvisii.set_max_bounce_depth(4)

    def _camera_configuration(self, at_vec, up_vec, eye_vec, quat):
        """
        Sets the configuration for the NVISII camera. Configuration
        is dependent on where the camera is located and where it
        looks at
        """
        # configures the camera
        self.camera.get_transform().look_at(
            at=at_vec, up=up_vec, eye=eye_vec, previous=False  # look at (world coordinate)  # up vector
        )

        self.camera.get_transform().rotate_around(eye_vec, quat)

    def set_camera_pos_quat(self, pos, quat):
        self.camera.get_transform().set_position(pos)
        self.camera.get_transform().look_at(
            at=(0, 0, 1.06), up=(0, 0, 1), eye=pos, previous=False  # look at (world coordinate)  # up vector
        )
        # self.camera.get_transform().rotate_around(pos, quat)

    def _get_orientation_geom(self, name):
        """
        Gets the position and quaternion for a geom
        """

        pos = self.env.sim.data.geom_xpos[self.env.sim.model.geom_name2id(name)]
        R = self.env.sim.data.geom_xmat[self.env.sim.model.geom_name2id(name)].reshape(3, 3)

        quat_xyzw = mat2quat(R)
        quat = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

        return pos, quat

    def _load(self):
        """
        Loads the nessecary textures, materials, and geoms into the
        NVISII renderer
        """
        self.parser = Parser("nvisii", self.env)
        self.parser.parse_textures()
        self.parser.parse_materials()
        self.parser.parse_geometries()
        self.components = self.parser.components
        self.max_elements = self.parser.max_elements
        # set initial colors in components
        for comp in self.components:
            try:
                self._init_component_colors[comp] = \
                self.components[comp].obj.get_material().get_base_color()
            except:
                try:
                    self._init_component_colors[comp] = \
                    self.components[comp].obj.materials[0].get_base_color()
                except:
                    pass

    def update(self, is_mp=False):
        """
        Updates the states for the wrapper given a certain action

        Args:
            action (np-array): The action the robot should take
        """
        for key, value in self.components.items():
            self._update_orientation(name=key, component=value, is_mp=is_mp)
        if self.env_type == "kitchen":
            # light switch effect 
            if np.linalg.norm(self.env.sim.data.qpos[17:19] - [-0.69, -0.05]) < 0.3 and not self.light_set:
                self.light_set = True 
                self.light_s = nvisii.entity.create(
                    name="lights",
                    mesh=nvisii.mesh.create_sphere("lights"),
                    transform=nvisii.transform.create("lights"),
                )

                self.light_s.set_light(nvisii.light.create("lights"))
                self.light_s.get_light().set_falloff(4)
                self.light_s.set_visibility(
                    camera=False, 
                    diffuse=False, 
                    glossy=False, 
                    transmission=False, 
                    volume_scatter=False, 
                    shadow=False
                )
                self.light_s.get_light().set_intensity(5)  # intensity of the light
                self.light_s.get_light().set_color(nvisii.vec3(255/255, 248/255, 211/255))
                self.light_s.get_transform().set_scale(nvisii.vec3(0.02))  # scale the light down
                self.light_s.get_transform().set_position(nvisii.vec3(-0.4, 0.45,2.30))  # sets the position of the light

    def _update_orientation(self, name, component, is_mp=False):
        """
        Update position for an object or a robot in renderer.

        Args:
            name (string): name of component
            component (nvisii entity or scene): Object in renderer and other info
                                                for object.
        """
        obj = component.obj
        parent_body_name = component.parent_body_name
        geom_pos = component.geom_pos
        geom_quat = component.geom_quat
        dynamic = component.dynamic

        if not dynamic:
            return

        if self.env_type == 'kitchen':
            # kitchen body tags
            self.body_tags = [
                                "panda0_link0",
                                "panda0_link1",
                                "panda0_link2",
                                "panda0_link3",
                                "panda0_link4",
                                "panda0_link5",
                                "panda0_link6",
                                "panda0_link7",
                                "panda0_leftfinger",
                                "panda0_rightfinger",
                                "counters",
                                "ovenroot",
                                "panda0_pedestal",
                                "knob 1",
                                "knob 2",
                                "knob 3",
                                "knob 4",
                                "Burner 1",
                                "Burner 2",
                                "Burner 3",
                                "Burner 4",
                                "hoodroot",
                                "lightswitchbaseroot",
                                "lightswitchroot",
                                "lightblock_hinge",
                                "slide",
                                "slidelink",
                                "hingecab",
                                "hingeleftdoor",
                                "hingerightdoor",
                                "microroot",
                                "microdoorroot",
                                "kettleroot",
                                "wallroot",
                                "wall"
                            ]
        elif self.env_type == 'metaworld':
            # metaworld
            self.body_tags = [
                'tablelink',
                'RetainingWall',
                'controller_box',
                'pedestal_feet',
                'right_arm_base_link',
                'robot_right_l0',
                'head',
                'robot_right_l1',
                'robot_right_l2',
                'robot_right_l3',
                'robot_right_l4',
                'robot_right_l5',
                'robot_right_l6',
                'right_hand',
                'right_l1',
                'right_l2',
                'right_l4',
                'right_l6',
                'asmbly_peg',
                'peg',
                'torso',
                'pedestal',
                'screen',
                'objA',
                'binA',
                'binB',
                'hammer',
                'HammerHandle',
                'hammerblock',
                'nail_link'
            ]
        elif self.env_type == 'mopa':
            self.body_tags = [
                "controller_box",
                "pedestal_feet",
                "torso",
                "pedestal",
                "head",
                "screen",
                "clawGripper",
                "clawGripper_target",
                "table",
                "bin1",
                'right_l0',
                'right_l1',
                'right_l2',
                'right_l3',            
                'right_l4',
                'right_l5',
                'right_l6',
                '4_part4',
                '2_part2',
                '1_part1',
                '0_part0',
                'right_arm_base_link',
                'rightclaw',
                'leftclaw',
                'peg',
                'right_gripper_base',
                'r_gripper_l_finger_tip',
                'r_gripper_r_finger_tip'
            ]
        elif self.env_type == 'menagerie':
            self.body_tags = [
                "base",
                "right_l0",
                "head",
                "right_l1",
                "right_l2",
                "right_l3",
                "right_l4",
                "right_l5",
                "right_l6"
            ]
        if parent_body_name != "worldbody":
            if self.tag_in_name(name):
                pos = self.env.sim.data.get_body_xpos(parent_body_name)
            else:
                pos = self.env.sim.data.get_geom_xpos(name)
            # use original peg position for metaworld (hardcoded)
            if (name == "peg0" or name == "peg") \
                and str(type(self.env)) == "<class 'metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_disassemble_peg_v2.SawyerNutDisassembleEnvV2'>":
                if not self.set_peg:
                    self.orig_peg_pos = self.env.sim.data.body_xpos[
                        self.env.sim.model.body_name2id("asmbly_peg")
                    ].copy() +  np.array([-0.03, -0.03, 0.0,])
                    self.set_peg = True 
                pos = self.orig_peg_pos
            B = self.env.sim.data.body_xmat[self.env.sim.model.body_name2id(parent_body_name)].reshape((3, 3))
            quat_xyzw_body = mat2quat(B)
            quat_wxyz_body = np.array(
                [quat_xyzw_body[3], quat_xyzw_body[0], quat_xyzw_body[1], quat_xyzw_body[2]]
            )  # wxyz
            nvisii_quat = nvisii.quat(*quat_wxyz_body) * nvisii.quat(*geom_quat)

            if self.tag_in_name(name):
                # Add position offset if there are position offset defined in the geom tag
                homo_mat = T.pose2mat((np.zeros((1, 3), dtype=np.float32), quat_xyzw_body))
                pos_offset = homo_mat @ np.array([geom_pos[0], geom_pos[1], geom_pos[2], 1.0]).transpose()
                pos = pos + pos_offset[:3]
        else:
            pos = [0, 0, 0]
            nvisii_quat = nvisii.quat(1, 0, 0, 0)  # wxyz
        if isinstance(obj, nvisii.scene):

            # temp fix -- look into XML file for correct quat
            if "s_visual" in name:
                # single robot
                if len(self.env.robots) == 1:
                    nvisii_quat = nvisii.quat(0, 0.5, 0, 0)
                # two robots - 0
                elif len(self.env.robots) == 2 and "robot_0" in name:
                    nvisii_quat = nvisii.quat(-0, 0.5, 0.5, 0)
                # two robots - 1
                else:
                    nvisii_quat = nvisii.quat(-0, 0.5, -0.5, 0)

            obj.transforms[0].set_position(nvisii.vec3(pos[0], pos[1], pos[2]))
            obj.transforms[0].set_rotation(nvisii_quat)
        else:
            obj.get_transform().set_position(nvisii.vec3(pos[0], pos[1], pos[2]))
            obj.get_transform().set_rotation(nvisii_quat)

        # set color based on mp 
        print(f"Name: {name}")
        is_robot_body = False
        if self.env_type == "metaworld":
            is_robot_body = name in [
                "robot_right_l0",
                "base_link",
                "pedestal",
                "pedestal0",
                "robot_right_l1",
                "robot_right_l2",
                "robot_right_l3",
                "robot_right_l4",
                "robot_right_l5",
                "robot_right_l6",
                "torso0",
                "right_arm_base_link0",
                "rightpad_geom",
                "leftpad_geom",
                "head0",
                "controller_box0",
                "right_hand0", 
                "right_hand1"
            ]
        if self.env_type == "kitchen":
            is_robot_body = name in [
                "panda0_link00",
                "panda0_link10",
                "panda0_link20",
                "panda0_link30",
                "panda0_link40",
                "panda0_link50",
                "panda0_link60",
                "panda0_link70",
                "panda0_link72",
                "panda0_leftfinger0",
                "panda0_leftfinger1",
                "panda0_leftfinger2",
                "panda0_leftfinger3",
                "panda0_leftfinger4",
                "panda0_leftfinger5",
                "panda0_leftfinger6",
                "panda0_leftfinger7",
                "panda0_leftfinger8",
                "panda0_leftfinger9",
                "panda0_leftfinger10",
                "panda0_leftfinger11",
                "panda0_leftfinger12",
                "panda0_leftfinger13",
                "panda0_rightfinger0",
                "panda0_rightfinger1",
                "panda0_rightfinger2",
                "panda0_rightfinger3",
                "panda0_rightfinger4",
                "panda0_rightfinger5",
                "panda0_rightfinger6",
                "panda0_rightfinger7",
                "panda0_rightfinger8",
                "panda0_rightfinger9",
                "panda0_rightfinger10",
                "panda0_rightfinger11",
                "panda0_rightfinger12",
                "panda0_rightfinger13",
            ]
        if self.env_type == "mopa":
            is_robot_body = name in [
                "controller_box0",
                "pedestal_feet0",
                "torso0",
                "right_arm_base_link0",
                "right_l0_g0",
                "head_g0",
                "right_l10",
                "right_l20",
                "right_l30",
                "right_l40",
                "right_l50",
                "right_l60",
                "clawGripper0",
                "clawGripper1",
                "rightclaw_it",
                "leftclaw_it0",
            ]
        if is_mp and is_robot_body:
            if "scene" in str(type(obj)):
                obj.materials[0].set_base_color(nvisii.vec3(0., 0., 1.))
            else:
                obj.get_material().set_base_color(nvisii.vec3(0., 0., 1.))
        else:
            if "scene" in str(type(obj)):
                obj.materials[0].set_base_color(self._init_component_colors[name])
            else:
                obj.get_material().set_base_color(self._init_component_colors[name])

    def tag_in_name(self, name):
        """
        Checks if one of the tags in body tags in the name

        Args:
            name (string): Name of component
        """
        for tag in self.body_tags:
            if tag in name:
                return True
        return False

    def render(self, render_type="png"):
        """
        Renders an image of the NVISII renderer

        Args:
            render_type (string, optional): Type of file to save as. Defaults to 'png'
        """
        self.img_cntr += 1
        verbose_word = "frame" if self.video_mode else "image"

        img_file = f"{self.img_path}/image_{self.img_cntr}.{render_type}"
        self.render_to_file(img_file)

        if self.verbose == 1:
            print(f"Rendering {verbose_word}... {self.img_cntr}")

    def render_to_file(self, img_file):
        nvisii.render_to_file(width=self.width, height=self.height, samples_per_pixel=self.spp, file_path=img_file)

    def render_segmentation_data(self, img_file):

        segmentation_array = nvisii.render_data(
            width=int(self.width),
            height=int(self.height),
            start_frame=0,
            frame_count=1,
            bounce=int(0),
            options="entity_id",
            seed=1,
        )
        segmentation_array = np.array(segmentation_array).reshape(self.height, self.width, 4)[:, :, 0]
        segmentation_array[segmentation_array > 3.4028234663852886e37] = 0
        segmentation_array[segmentation_array < 3.4028234663852886e-37] = 0
        segmentation_array = np.flipud(segmentation_array)

        rgb_data = self.segmentation_to_rgb(segmentation_array.astype(dtype=np.uint8))

        from PIL import Image

        rgb_img = Image.fromarray(rgb_data)
        rgb_img.save(img_file)

    def render_data_to_file(self, img_file):

        if self.vision_modalities == "depth" and self.img_cntr != 1:

            depth_data = nvisii.render_data(
                width=self.width,
                height=self.height,
                start_frame=0,
                frame_count=1,
                bounce=int(0),
                options=self.vision_modalities,
            )

            depth_data = np.array(depth_data).reshape(self.height, self.width, 4)
            depth_data = np.flipud(depth_data)[:, :, [0, 1, 2]]

            # normalize depths
            depth_data[:, :, 0] = (depth_data[:, :, 0] - np.min(depth_data[:, :, 0])) / (
                np.max(depth_data[:, :, 0]) - np.min(depth_data[:, :, 0])
            )
            depth_data[:, :, 1] = (depth_data[:, :, 1] - np.min(depth_data[:, :, 1])) / (
                np.max(depth_data[:, :, 1]) - np.min(depth_data[:, :, 1])
            )
            depth_data[:, :, 2] = (depth_data[:, :, 2] - np.min(depth_data[:, :, 2])) / (
                np.max(depth_data[:, :, 2]) - np.min(depth_data[:, :, 2])
            )

            from PIL import Image

            depth_image = Image.fromarray(((1 - depth_data) * 255).astype(np.uint8))
            depth_image.save(img_file)

        elif self.vision_modalities == "normal" and self.img_cntr != 1:

            normal_data = nvisii.render_data(
                width=self.width,
                height=self.height,
                start_frame=0,
                frame_count=1,
                bounce=int(0),
                options="screen_space_normal",
            )

            normal_data = np.array(normal_data).reshape(self.height, self.width, 4)
            normal_data = np.flipud(normal_data)[:, :, [0, 1, 2]]

            normal_data[:, :, 0] = (normal_data[:, :, 0] + 1) / 2 * 255  # R
            normal_data[:, :, 1] = (normal_data[:, :, 1] + 1) / 2 * 255  # G
            normal_data[:, :, 2] = 255 - ((normal_data[:, :, 2] + 1) / 2 * 255)  # B

            from PIL import Image

            normal_image = Image.fromarray((normal_data).astype(np.uint8))
            normal_image.save(img_file)

        else:

            nvisii.render_data_to_file(
                width=self.width,
                height=self.height,
                start_frame=0,
                frame_count=1,
                bounce=int(0),
                options=self.vision_modalities,
                file_path=img_file,
            )

    def randomize_colors(self, N, bright=True):
        """
        Modified from https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/visualize.py#L59
        Generate random colors.
        To get visually distinct colors, generate them in HSV space then
        convert to RGB.
        """
        brightness = 1.0 if bright else 0.5
        hsv = [(1.0 * i / N, 1, brightness) for i in range(N)]
        colors = np.array(list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv)))
        rstate = np.random.RandomState(seed=20)
        np.random.shuffle(colors)
        return colors

    def segmentation_to_rgb(self, seg_im, random_colors=False):
        """
        Helper function to visualize segmentations as RGB frames.
        NOTE: assumes that geom IDs go up to 255 at most - if not,
        multiple geoms might be assigned to the same color.
        """
        # ensure all values lie within [0, 255]
        seg_im = np.mod(seg_im, 256)

        if random_colors:
            colors = self.randomize_colors(N=256, bright=True)
            return (255.0 * colors[seg_im]).astype(np.uint8)
        else:

            cmap = cm.get_cmap("jet")

            max_r = 0
            if self.segmentation_type[0][0] == "element":
                max_r = np.amax(seg_im) + 1
            elif self.segmentation_type[0][0] == "class":
                max_r = self.max_classes
                for i in range(len(seg_im)):
                    for j in range(len(seg_im[0])):
                        if seg_im[i][j] in self.parser.entity_id_class_mapping:
                            seg_im[i][j] = self.parser.entity_id_class_mapping[seg_im[i][j]]
                        else:
                            seg_im[i][j] = max_r - 1
            elif self.segmentation_type[0][0] == "instance":
                max_r = self.max_instances
                for i in range(len(seg_im)):
                    for j in range(len(seg_im[0])):
                        if seg_im[i][j] in self.parser.entity_id_class_mapping:
                            seg_im[i][j] = self.parser.entity_id_class_mapping[seg_im[i][j]]
                        else:
                            seg_im[i][j] = max_r - 1

            color_list = np.array([cmap(i / (max_r)) for i in range(max_r)])

            return (color_list[seg_im] * 255).astype(np.uint8)

    def reset(self):
        nvisii.clear_all()
        if self.env_type == "metaworld":
            self.set_peg = False 
        self._init_nvisii_components()
        self.update()

    def get_pixel_obs(self):
        frame_buffer = nvisii.render(width=self.width, height=self.height, samples_per_pixel=self.spp)

        frame_buffer = np.array(frame_buffer).reshape(self.height, self.width, 4)
        frame_buffer = np.flipud(frame_buffer)

        return frame_buffer

    def close(self):
        """
        Deinitializes the nvisii rendering environment
        """
        nvisii.deinitialize()
