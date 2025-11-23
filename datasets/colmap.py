import os
import math
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torchvision.transforms.functional as TF

import pytorch_lightning as pl

import datasets
from datasets.colmap_utils import \
    read_cameras_binary, read_images_binary, read_points3d_binary
from models.ray_utils import get_ray_directions
from utils.misc import get_rank


def get_center(pts):
    center = pts.mean(0)
    dis = (pts - center[None,:]).norm(p=2, dim=-1)
    mean, std = dis.mean(), dis.std()
    q25, q75 = torch.quantile(dis, 0.25), torch.quantile(dis, 0.75)
    valid = (dis > mean - 1.5 * std) & (dis < mean + 1.5 * std) & (dis > mean - (q75 - q25) * 1.5) & (dis < mean + (q75 - q25) * 1.5)
    center = pts[valid].mean(0)
    return center

def normalize_poses(poses, pts, up_est_method, center_est_method):
    if center_est_method == 'camera':
        # estimation scene center as the average of all camera positions
        center = poses[...,3].mean(0)
    elif center_est_method == 'lookat':
        # estimation scene center as the average of the intersection of selected pairs of camera rays
        cams_ori = poses[...,3]
        cams_dir = poses[:,:3,:3] @ torch.as_tensor([0.,0.,-1.])
        cams_dir = F.normalize(cams_dir, dim=-1)
        A = torch.stack([cams_dir, -cams_dir.roll(1,0)], dim=-1)
        b = -cams_ori + cams_ori.roll(1,0)
        t = torch.linalg.lstsq(A, b).solution
        center = (torch.stack([cams_dir, cams_dir.roll(1,0)], dim=-1) * t[:,None,:] + torch.stack([cams_ori, cams_ori.roll(1,0)], dim=-1)).mean((0,2))
    elif center_est_method == 'point':
        # first estimation scene center as the average of all camera positions
        # later we'll use the center of all points bounded by the cameras as the final scene center
        center = poses[...,3].mean(0)
    else:
        raise NotImplementedError(f'Unknown center estimation method: {center_est_method}')

    if up_est_method == 'ground':
        # estimate up direction as the normal of the estimated ground plane
        # use RANSAC to estimate the ground plane in the point cloud
        import pyransac3d as pyrsc
        ground = pyrsc.Plane()
        plane_eq, inliers = ground.fit(pts.numpy(), thresh=0.01) # TODO: determine thresh based on scene scale
        plane_eq = torch.as_tensor(plane_eq) # A, B, C, D in Ax + By + Cz + D = 0
        z = F.normalize(plane_eq[:3], dim=-1) # plane normal as up direction
        signed_distance = (torch.cat([pts, torch.ones_like(pts[...,0:1])], dim=-1) * plane_eq).sum(-1)
        if signed_distance.mean() < 0:
            z = -z # flip the direction if points lie under the plane
    elif up_est_method == 'camera':
        # estimate up direction as the average of all camera up directions
        z = F.normalize((poses[...,3] - center).mean(0), dim=0)
    else:
        raise NotImplementedError(f'Unknown up estimation method: {up_est_method}')

    # new axis
    y_ = torch.as_tensor([z[1], -z[0], 0.])
    x = F.normalize(y_.cross(z), dim=0)
    y = z.cross(x)

    if center_est_method == 'point':
        # rotation
        Rc = torch.stack([x, y, z], dim=1)
        R = Rc.T
        poses_homo = torch.cat([poses, torch.as_tensor([[[0.,0.,0.,1.]]]).expand(poses.shape[0], -1, -1)], dim=1)
        inv_trans = torch.cat([torch.cat([R, torch.as_tensor([[0.,0.,0.]]).T], dim=1), torch.as_tensor([[0.,0.,0.,1.]])], dim=0)
        poses_norm = (inv_trans @ poses_homo)[:,:3]
        pts = (inv_trans @ torch.cat([pts, torch.ones_like(pts[:,0:1])], dim=-1)[...,None])[:,:3,0]

        # translation and scaling
        poses_min, poses_max = poses_norm[...,3].min(0)[0], poses_norm[...,3].max(0)[0]
        pts_fg = pts[(poses_min[0] < pts[:,0]) & (pts[:,0] < poses_max[0]) & (poses_min[1] < pts[:,1]) & (pts[:,1] < poses_max[1])]
        center = get_center(pts_fg)
        tc = center.reshape(3, 1)
        t = -tc
        poses_homo = torch.cat([poses_norm, torch.as_tensor([[[0.,0.,0.,1.]]]).expand(poses_norm.shape[0], -1, -1)], dim=1)
        inv_trans = torch.cat([torch.cat([torch.eye(3), t], dim=1), torch.as_tensor([[0.,0.,0.,1.]])], dim=0)
        poses_norm = (inv_trans @ poses_homo)[:,:3]
        scale = poses_norm[...,3].norm(p=2, dim=-1).min()
        poses_norm[...,3] /= scale
        pts = (inv_trans @ torch.cat([pts, torch.ones_like(pts[:,0:1])], dim=-1)[...,None])[:,:3,0]
        pts = pts / scale
    else:
        # rotation and translation
        Rc = torch.stack([x, y, z], dim=1)
        tc = center.reshape(3, 1)
        R, t = Rc.T, -Rc.T @ tc
        poses_homo = torch.cat([poses, torch.as_tensor([[[0.,0.,0.,1.]]]).expand(poses.shape[0], -1, -1)], dim=1)
        inv_trans = torch.cat([torch.cat([R, t], dim=1), torch.as_tensor([[0.,0.,0.,1.]])], dim=0)
        poses_norm = (inv_trans @ poses_homo)[:,:3] # (N_images, 4, 4)

        # scaling
        scale = poses_norm[...,3].norm(p=2, dim=-1).min()
        poses_norm[...,3] /= scale

        # apply the transformation to the point cloud
        pts = (inv_trans @ torch.cat([pts, torch.ones_like(pts[:,0:1])], dim=-1)[...,None])[:,:3,0]
        pts = pts / scale

    return poses_norm, pts

def create_spheric_poses(cameras, n_steps=120):
    center = torch.as_tensor([0.,0.,0.], dtype=cameras.dtype, device=cameras.device)
    mean_d = (cameras - center[None,:]).norm(p=2, dim=-1).mean()
    mean_h = cameras[:,2].mean()
    r = (mean_d**2 - mean_h**2).sqrt()
    up = torch.as_tensor([0., 0., 1.], dtype=center.dtype, device=center.device)

    all_c2w = []
    for theta in torch.linspace(0, 2 * math.pi, n_steps):
        cam_pos = torch.stack([r * theta.cos(), r * theta.sin(), mean_h])
        l = F.normalize(center - cam_pos, p=2, dim=0)
        s = F.normalize(l.cross(up), p=2, dim=0)
        u = F.normalize(s.cross(l), p=2, dim=0)
        c2w = torch.cat([torch.stack([s, u, -l], dim=1), cam_pos[:,None]], axis=1)
        all_c2w.append(c2w)

    all_c2w = torch.stack(all_c2w, dim=0)
    
    return all_c2w

def rotation_matrix_between( a, b):
    """Compute the rotation matrix that rotates vector a to vector b.

    Args:
        a: The vector to rotate.
        b: The vector to rotate to.
    Returns:
        The rotation matrix.
    """
    # Normalize the vectors

    a = a / torch.norm(a)
    b = b / torch.norm(b)
    
    # Compute the axis of rotation (cross product)
    v = torch.cross(a, b)

    # Handle cases where `a` and `b` are parallel
    eps = 1e-6
    if torch.sum(torch.abs(v)) < eps:
        x = torch.tensor([1.0, 0, 0]) if abs(a[0]) < eps else torch.tensor([0, 1.0, 0])
        v = torch.cross(a, x)

    v = v / torch.norm(v)

    # Skew-symmetric matrix for cross product
    skew_sym_mat = torch.tensor([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

    # Compute the angle between the vectors
    theta = torch.arccos(torch.clamp(torch.dot(a, b), -1, 1))

    # Rodrigues' rotation formula
    return torch.eye(3) + torch.sin(theta) * skew_sym_mat + (1 - torch.cos(theta)) * (skew_sym_mat @ skew_sym_mat)

def nerfstudio_default_normalize_poses(myposes):
    poses = []
    for i, c2w in enumerate(myposes):
        c2w = c2w[np.array([0, 2, 1]), :]
        c2w[2, :] *= -1
        poses.append(c2w)
    poses = torch.stack(poses, dim=0) #transform json

        
    up = torch.mean(poses[:, :3, 1], dim=0)
    up = up / torch.norm(up)

    origins = poses[..., :3, 3]
    mean_origin =torch.mean(origins, dim=0)
    translation = mean_origin

    rotation = rotation_matrix_between(up, torch.tensor([0., 0., 1.]))
    transform = torch.cat([rotation, torch.matmul(rotation, -translation[..., None])], dim=-1)
    if poses.shape[1] == 3:
        # append [0, 0, 0, 1] to the last row
        poses = torch.cat([poses, torch.tensor([[[0, 0, 0, 1]]]).expand(poses.shape[0], -1, -1)], dim=1)
    oriented_poses = torch.matmul(transform, poses)
    scale = max(torch.min(oriented_poses[..., :3, 3])*(-1.), torch.max(oriented_poses[..., :3, 3]))
    render_poses = torch.matmul(transform, poses)
    render_poses[..., :3, 3] /= scale 
    return scale, rotation, translation 

def render_path_from_json(json_path, all_c2w_json):
    import json
    scale, rotation, translation = nerfstudio_default_normalize_poses(all_c2w_json)
    with open(json_path) as f:
        data = json.load(f)
    all_c2w = []
    for frame in data['camera_path']:
        c2w = torch.tensor(frame['camera_to_world'])
        c2w = c2w.reshape(4, 4)
        all_c2w.append(c2w)
    all_c2w = torch.stack(all_c2w, dim=0)

    unscale_render_poses = all_c2w.clone()
    unscale_render_poses[...,:3,3] *=scale
    inverse_transform = torch.cat([rotation.T,  translation[..., None]], dim=-1)
    untransfromed_poses = torch.matmul(inverse_transform, unscale_render_poses)

    for i in range (untransfromed_poses.shape[0]):
        c2w = untransfromed_poses[i].clone()
        c2w[2, :] *= -1
        c2w = c2w[np.array([0, 2, 1]), :]
        untransfromed_poses[i] = c2w

    return untransfromed_poses

class ColmapDatasetBase():
    # the data only has to be processed once
    initialized = False
    properties = {}

    def setup(self, config, split):
        self.config = config
        self.split = split
        self.rank = get_rank()

        if 'img_wh' in self.config:
            w, h = self.config.img_wh
            # assert round(W / w * h) == H #TODO CHECK
        elif 'img_downscale' in self.config:
            w, h = int(W / self.config.img_downscale + 0.5), int(H / self.config.img_downscale + 0.5)
        else:
            raise KeyError("Either img_wh or img_downscale should be specified.")
        
        mask_dir = os.path.join(self.config.root_dir, 'masks')
        has_mask = os.path.exists(mask_dir) # TODO: support partial masks
        apply_mask = has_mask and self.config.apply_mask

        if not ColmapDatasetBase.initialized:
            camdata = read_cameras_binary(os.path.join(self.config.root_dir, 'sparse/0/cameras.bin'))

            H = int(camdata[1].height)
            W = int(camdata[1].width)

            img_wh = (w, h)
            factor = w / W
            self.factorX = w / W
            self.factorY = h / H

            if camdata[1].model == 'SIMPLE_RADIAL':
                fx = camdata[1].params[0] * self.factorX
                fy = camdata[1].params[0] * self.factorY
                cx = camdata[1].params[1] * self.factorX
                cy = camdata[1].params[2] * self.factorY
            elif camdata[1].model in ['PINHOLE', 'OPENCV']:
                fx = camdata[1].params[0] * self.factorX
                fy = camdata[1].params[1] * self.factorY
                cx = camdata[1].params[2] * self.factorX
                cy = camdata[1].params[3] * self.factorY
            else:
                raise ValueError(f"Please parse the intrinsics for camera model {camdata[1].model}!")
            
            directions = get_ray_directions(w, h, fx, fy, cx, cy).to(self.rank)

            imdata = read_images_binary(os.path.join(self.config.root_dir, 'sparse/0/images.bin'))

           
            all_c2w, all_images, all_fg_masks = [], [], []

            for i, d in enumerate(imdata.values()):
                R = d.qvec2rotmat()
                t = d.tvec.reshape(3, 1)
                c2w = torch.from_numpy(np.concatenate([R.T, -R.T@t], axis=1)).float()
                c2w[:,1:3] *= -1. # COLMAP => OpenGL
                all_c2w.append(c2w)
                if self.split in ['train', 'val']:
                    img_path = os.path.join(self.config.root_dir, 'images', d.name)
                    img = Image.open(img_path)
                    img = img.resize(img_wh, Image.BICUBIC)
                    img = TF.to_tensor(img).permute(1, 2, 0)[...,:3]
                    img = img.to(self.rank) if self.config.load_data_on_gpu else img.cpu()
                    if has_mask:
                        mask_paths = [os.path.join(mask_dir, d.name), os.path.join(mask_dir, d.name[3:])]
                        mask_paths = list(filter(os.path.exists, mask_paths))
                        assert len(mask_paths) == 1
                        mask = Image.open(mask_paths[0]).convert('L') # (H, W, 1)
                        mask = mask.resize(img_wh, Image.BICUBIC)
                        mask = TF.to_tensor(mask)[0]
                    else:
                        mask = torch.ones_like(img[...,0], device=img.device)
                    all_fg_masks.append(mask) # (h, w)
                    all_images.append(img)
            
            all_c2w = torch.stack(all_c2w, dim=0)   
            pts3d = read_points3d_binary(os.path.join(self.config.root_dir, 'sparse/0/points3D.bin'))
            pts3d = torch.from_numpy(np.array([pts3d[k].xyz for k in pts3d])).float()
            all_c2w, pts3d = normalize_poses(all_c2w, pts3d, up_est_method=self.config.up_est_method, center_est_method=self.config.center_est_method)
            scale = max(torch.min(all_c2w[..., :3, 3])*(-1.), torch.max(all_c2w[..., :3, 3]))
            all_c2w[..., :3, 3] /= scale 
            ColmapDatasetBase.properties = {
                'w': w,
                'h': h,
                'img_wh': img_wh,
                'factor': factor,
                'factorX': self.factorX,
                'factorY': self.factorY,
                'has_mask': has_mask,
                'apply_mask': apply_mask,
                'directions': directions,
                'pts3d': pts3d,
                'all_c2w': all_c2w,
                'all_images': all_images,
                'all_fg_masks': all_fg_masks
            }

            ColmapDatasetBase.initialized = True
        
        for k, v in ColmapDatasetBase.properties.items():
            setattr(self, k, v)

        if self.split == 'test':
            # spherical trajectory for testing
            # self.all_c2w = create_spheric_poses(self.all_c2w[:,:,3], n_steps=self.config.n_test_traj_steps)
            
            # renedring the input path
            #self.all_c2w = self.all_c2w # for input view rendering comment above line and uncomment this line
            
            # render the path from json
            self.all_c2w_path = render_path_from_json("check_path.json", self.all_c2w)
            self.all_c2w = self.all_c2w_path

            self.all_images = torch.zeros((self.all_c2w.shape[0], self.h, self.w, 3), dtype=torch.float32)
            self.all_fg_masks =  torch.zeros((self.all_c2w.shape[0], self.h, self.w), dtype=torch.float32)
        else:
            self.all_images, self.all_fg_masks = torch.stack(self.all_images, dim=0).float(), torch.stack(self.all_fg_masks, dim=0).float()

        """
        # for debug use
        from models.ray_utils import get_rays
        rays_o, rays_d = get_rays(self.directions.cpu(), self.all_c2w, keepdim=True)
        pts_out = []
        pts_out.append('\n'.join([' '.join([str(p) for p in l]) + ' 1.0 0.0 0.0' for l in rays_o[:,0,0].reshape(-1, 3).tolist()]))

        t_vals = torch.linspace(0, 1, 8)
        z_vals = 0.05 * (1 - t_vals) + 0.5 * t_vals

        ray_pts = (rays_o[:,0,0][..., None, :] + z_vals[..., None] * rays_d[:,0,0][..., None, :])
        pts_out.append('\n'.join([' '.join([str(p) for p in l]) + ' 0.0 1.0 0.0' for l in ray_pts.view(-1, 3).tolist()]))

        ray_pts = (rays_o[:,0,0][..., None, :] + z_vals[..., None] * rays_d[:,self.h-1,0][..., None, :])
        pts_out.append('\n'.join([' '.join([str(p) for p in l]) + ' 0.0 0.0 1.0' for l in ray_pts.view(-1, 3).tolist()]))

        ray_pts = (rays_o[:,0,0][..., None, :] + z_vals[..., None] * rays_d[:,0,self.w-1][..., None, :])
        pts_out.append('\n'.join([' '.join([str(p) for p in l]) + ' 0.0 1.0 1.0' for l in ray_pts.view(-1, 3).tolist()]))

        ray_pts = (rays_o[:,0,0][..., None, :] + z_vals[..., None] * rays_d[:,self.h-1,self.w-1][..., None, :])
        pts_out.append('\n'.join([' '.join([str(p) for p in l]) + ' 1.0 1.0 1.0' for l in ray_pts.view(-1, 3).tolist()]))
        
        open('cameras.txt', 'w').write('\n'.join(pts_out))
        open('scene.txt', 'w').write('\n'.join([' '.join([str(p) for p in l]) + ' 0.0 0.0 0.0' for l in self.pts3d.view(-1, 3).tolist()]))

        exit(1)
        """

        self.all_c2w = self.all_c2w.float().to(self.rank)
        if self.config.load_data_on_gpu:
            self.all_images = self.all_images.to(self.rank) 
            self.all_fg_masks = self.all_fg_masks.to(self.rank)
        

class ColmapDataset(Dataset, ColmapDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __len__(self):
        return len(self.all_images)
    
    def __getitem__(self, index):
        return {
            'index': index
        }


class ColmapIterableDataset(IterableDataset, ColmapDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __iter__(self):
        while True:
            yield {}


@datasets.register('colmap')
class ColmapDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def setup(self, stage=None):
        if stage in [None, 'fit']:
            self.train_dataset = ColmapIterableDataset(self.config, 'train')
        if stage in [None, 'fit', 'validate']:
            self.val_dataset = ColmapDataset(self.config, self.config.get('val_split', 'train'))
        if stage in [None, 'test']:
            self.test_dataset = ColmapDataset(self.config, self.config.get('test_split', 'test'))
        if stage in [None, 'predict']:
            self.predict_dataset = ColmapDataset(self.config, 'train')         

    def prepare_data(self):
        pass
    
    def general_loader(self, dataset, batch_size):
        sampler = None
        return DataLoader(
            dataset, 
            num_workers=os.cpu_count(), 
            batch_size=batch_size,
            pin_memory=True,
            sampler=sampler
        )
    
    def train_dataloader(self):
        return self.general_loader(self.train_dataset, batch_size=1)

    def val_dataloader(self):
        return self.general_loader(self.val_dataset, batch_size=1)

    def test_dataloader(self):
        return self.general_loader(self.test_dataset, batch_size=1) 

    def predict_dataloader(self):
        return self.general_loader(self.predict_dataset, batch_size=1)       
