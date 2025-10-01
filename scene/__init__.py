from typing import List
import os
import random
import json

from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

from scene.cameras import CameraGaussianAvatars, Camera
import torch
import numpy as np
import t3d
import tloaders
from tqdm import tqdm
from scene.gaussian_model import BasicPointCloud
from utils.sh_utils import SH2RGB


class Scene:
    gaussians: GaussianModel

    def __init__(self, args: ModelParams, gaussians: GaussianModel, is_debug, novel_view, only_head, load_iteration=None, shuffle=True,
                 resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        print(args.source_path)
        if os.path.exists(os.path.join(args.source_path, "transforms.json")):
            scene_info = sceneLoadTypeCallbacks["nerfblendshape"](args.source_path, args.eval, is_debug, novel_view, only_head)
        else:
            assert False, "Could not recognize scene type!"

        self.num_points = scene_info.point_cloud.points

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply"),
                                                                   'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file: 
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling
        self.cameras_extent = scene_info.nerf_normalization["radius"]  
        for resolution_scale in resolution_scales:
            # Setting up a dataloader will reduce the memory footprint but slow it down slightly, if hardware limits, try it
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale,
                                                                            args)  
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale,
                                                                           args)
        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                 "point_cloud",
                                                 "iteration_" + str(self.loaded_iter),
                                                 "point_cloud.ply"),
                                    og_number_points=len(scene_info.point_cloud.points))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]


class SceneGaussianAvatars:
    """
    Implements Nersemble gausian avatars scene dataloading.
    """
    def __init__(
        self,
        data_root: str,
        transforms_filename: str,
        gaussians,
        id,
        prefetch: bool=False
    ) -> None:
        """
        Initialize a SceneGaussianAvatars object

        :param data_root
        :param transforms_filename
        :param prefetch
        """
        self.id = id
        self.model_path = os.path.join('output', self.id)
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        self.gaussians = gaussians
        self.dataset: tloaders.GaussianAvatars = tloaders.DatasetRegistry.get_dataset(
            "GaussianAvatars",
            data_root=data_root,
            transforms_filename=transforms_filename,
            prefetch=prefetch
        )
        self.dataset_val: tloaders.GaussianAvatars = tloaders.DatasetRegistry.get_dataset(
            "GaussianAvatars",
            data_root=data_root,
            transforms_filename="transforms_val.json",
            prefetch=prefetch
        )

        # parse our flame samples to Camera objects
        

        # calculate cameras extent
        cam_positions = []
        for i in range(15):
            cam: tloaders.FlameSample = self.dataset[i]
            cam_positions += [cam["campos"].unsqueeze(0)]
        cam_positions = torch.cat(cam_positions, dim=0)
        min_pos = cam_positions.min(dim=0)[0]
        max_pos = cam_positions.max(dim=0)[0]
        length = (max_pos - min_pos).norm()
        self.cameras_extent = length.numpy()
        
        self.cameras = self.parse_cameras(self.dataset)
        self.test_cameras = self.parse_cameras(self.dataset_val)

        self.exp_dims = 100

        num_pts = 10_000
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        self.pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
    
    def parse_cameras(self, dataset):
        parsing_loop = tqdm(
            enumerate(dataset.cam_infos),
            total=len(dataset.cam_infos),
            desc="Parsing GaussianAvatars to Scene"
        )
        cameras = []
        sample: tloaders.NersembleCamInfo
        for idx, sample in parsing_loop:
            K = sample.K
            fx = K[0, 0]
            fy = K[1, 1]
            fovx = sample.fovx
            fovy = t3d.cam.param.focal2fov(t3d.cam.param.fov2focal(fovx, sample.width), sample.height)

            
            c2w = sample.c2w
            width, height = sample.width, sample.height
            shape_params = sample.flame_id_params
            expr_params = sample.flame_ex_params
            fl_rot = sample.flame_rotation
            fl_neck = sample.flame_neck
            fl_jaw = sample.flame_jaw
            fl_eyes = sample.flame_eyes
            fl_trans = sample.flame_translation
            fl_static_offsets = sample.flame_static_offset

            image_path = sample.image_path
            msk_path = sample.mask_name
            
            self.shape_param = torch.from_numpy(shape_params).unsqueeze(0)
            self.static_offset = torch.from_numpy(fl_static_offsets)

            # dataset must be proprocessed with
            # face_parsing.
            # see preproc_ga_fparsing.py
            mask_dir = os.path.dirname(msk_path)
            mask_fname = os.path.basename(msk_path)
            mouth_mask_path = os.path.join(mask_dir, '..', 'parsing', f'{mask_fname.replace(".png","")}_mouth.png')
            
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])
            T = w2c[:3, 3]
            # T[1] -= 0.065
            # T = sample.campos
            

            # fl_trans[:, 1] += 0.07
        

            cameras += [CameraGaussianAvatars(
                uid = sample.uid,
                R=R,
                T=T,
                FoVx=fovx,
                FoVy=fovy,
                image_path=image_path,
                mask_path=msk_path,
                mouth_mask_path=mouth_mask_path,
                image=None,
                head_mask=None,
                mouth_mask=None,
                exp_param=expr_params,
                eyes_pose=fl_eyes,
                eyelids=None,
                jaw_pose=fl_jaw,
                neck_pose=fl_neck,
                rot=fl_rot,
                fl_trans=fl_trans,
                image_name=image_path,
                width=width,
                height=height,
                colmap_id=sample.uid,
                c2w=c2w,
                # trans=np.array([0.0, -0.078, 0.0]),
                campos=torch.from_numpy(sample.campos).to(dtype=torch.float32),
                K=sample.K
            )]
        return cameras

    def getCameras(self) -> List[Camera]:
        return self.cameras
    
    def getTrainCameras(self, scale=1.0):
        return self.cameras

    def getTestCameras(self):
        return self.test_cameras
    
    def save(self, iteration):
        point_cloud_path = os.path.join('output', self.id, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))