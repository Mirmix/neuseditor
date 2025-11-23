import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_efficient_distloss import flatten_eff_distloss

import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_debug

import models
from models.utils import cleanup
from models.ray_utils import get_rays
import systems
from systems.base import BaseSystem
from systems.criterions import PSNR, binary_cross_entropy
from datasets.dtu import create_spheric_poses 

from torchvision.transforms import ToPILImage
to_pil = ToPILImage()

from .pds import PDS, PDSConfig

@systems.register('neus-system')
class NeuSSystem(BaseSystem):
    """
    Two ways to print to console:
    1. self.print: correctly handle progress bar
    2. rank_zero_info: use the logging module
    """
    def prepare(self):
        self.criterions = {
            'psnr': PSNR()
        }
        self.train_num_samples = self.config.model.train_num_rays * (self.config.model.num_samples_per_ray + self.config.model.get('num_samples_per_ray_bg', 0))
        self.train_num_rays = self.config.model.train_num_rays
        
        # Start with a original scene learning
        self.generative_step = False
        self.interpolation_step = False
        

    def forward(self, batch):
        return self.model(batch['rays'])
    
    def preprocess_data(self, batch, stage):
        if 'index' in batch: # validation / testing
            index = batch['index']
        else:
            if self.config.model.batch_image_sampling:
                index = torch.randint(0, len(self.dataset.all_images), size=(self.train_num_rays,), device=self.dataset.all_images.device)
            else:
                index = torch.randint(0, len(self.dataset.all_images), size=(1,), device=self.dataset.all_images.device)
        if stage in ['train']:

            c2w = self.dataset.all_c2w[index]
            x = torch.randint(
                0, self.dataset.w, size=(self.train_num_rays,), device=self.dataset.all_images.device
            )
            y = torch.randint(
                0, self.dataset.h, size=(self.train_num_rays,), device=self.dataset.all_images.device
            )
            if self.generative_step: # generate rays for 64x64/128x128 image grid
                # Sample a regular grid of size 64x64/128x128
                stride = int(self.dataset.w/self.config.diffusion.resolution)  # You can adjust this stride value to change the grid size 
                offset_diameter = self.dataset.w - self.config.diffusion.resolution*stride
                x = torch.arange(0, self.config.diffusion.resolution*stride, device=self.dataset.all_images.device)
                y = torch.arange(0, self.config.diffusion.resolution*stride, device=self.dataset.all_images.device)

                # Use torch.meshgrid to create a grid of all pixel coordinates
                X, Y = torch.meshgrid(x, y)
                #sampled_coordinates = torch.stack([X[::stride, ::stride].flatten(), Y[::stride, ::stride].flatten()], dim=-1)
                x = X[::stride, ::stride].flatten() + torch.randint(0,offset_diameter+stride,size=(1,)).item()
                y = Y[::stride, ::stride].flatten() + torch.randint(0,offset_diameter+stride,size=(1,)).item()
                               
                if self.interpolation_step == False:
                    index = torch.full((x.shape), (self.global_step + self.rank) % len(self.dataset.all_images) ,device= self.dataset.all_images.device)
                    c2w = self.dataset.all_c2w[index]    
                else: ## view interpolation
                    scale=1.0
                    test_c2w = create_spheric_poses(self.dataset.all_c2w[:49,:,3], n_steps=self.dataset.config.n_test_traj_steps, reverse_up=True)
                    index = torch.full((x.shape), (self.global_step + self.rank) % self.dataset.config.n_test_traj_steps ,device= self.dataset.all_images.device)
                    c2w = test_c2w[index]
             
            if self.dataset.directions.ndim == 3: # (H, W, 3)
                directions = self.dataset.directions[y, x]
            elif self.dataset.directions.ndim == 4: # (N, H, W, 3)
                if self.interpolation_step:
                    directions = self.dataset.directions[0, y, x]
                else:
                    directions = self.dataset.directions[index, y, x]

            rays_o, rays_d = get_rays(directions, c2w)
            rgb = None
            fg_mask = None
            if self.interpolation_step==False:
                rgb = self.dataset.all_images[index, y, x].view(-1, self.dataset.all_images.shape[-1]).to(self.rank)
                fg_mask = self.dataset.all_fg_masks[index, y, x].view(-1).to(self.rank)        
        else:
            c2w = self.dataset.all_c2w[index][0]
            if self.dataset.directions.ndim == 3: # (H, W, 3)
                directions = self.dataset.directions
            elif self.dataset.directions.ndim == 4: # (N, H, W, 3)
                directions = self.dataset.directions[index][0] 
            rays_o, rays_d = get_rays(directions, c2w)
            rgb = self.dataset.all_images[index].view(-1, self.dataset.all_images.shape[-1]).to(self.rank)
            fg_mask = self.dataset.all_fg_masks[index].view(-1).to(self.rank)

        rays = torch.cat([rays_o, F.normalize(rays_d, p=2, dim=-1)], dim=-1)

        if stage in ['train']:
            if self.config.model.background_color == 'white':
                self.model.background_color = torch.ones((3,), dtype=torch.float32, device=self.rank)
            elif self.config.model.background_color == 'random':
                self.model.background_color = torch.rand((3,), dtype=torch.float32, device=self.rank)
            else:
                raise NotImplementedError
        else:
            self.model.background_color = torch.ones((3,), dtype=torch.float32, device=self.rank)
        
        if self.dataset.apply_mask:
            rgb = rgb * fg_mask[...,None] + self.model.background_color * (1 - fg_mask[...,None])
        
        batch.update({
            'rays': rays,
            'rgb': rgb,
            'fg_mask': fg_mask
        })      
    
    def training_step(self, batch, batch_idx):

        res = self.config.diffusion.resolution
        loss = 0.     
                
        if (self.generative_step):

            render_img = None
            render_phong_img = None
            phong_shading = None
            orig_img = None

            if(self.pds==None):
                pds_conf = PDSConfig()
                pds_conf.src_prompt = self.config.diffusion.src_prompt
                pds_conf.tgt_prompt = self.config.diffusion.tgt_prompt
                pds_conf.guidance_scale = self.config.diffusion.guidance_scale
                pds_conf.prompt_noising_scale = self.config.diffusion.prompt_noising_scale
                self.pds = PDS(pds_conf)
                if self.var_param_grad_turn_off:
                    # turn off gradients for self.model.variance
                    for p in self.model.variance.parameters():
                        p.requires_grad_(False)
                with torch.no_grad():
                    if self.config.model.geometry.xyz_encoding_config.otype == "ProgressiveBandHashGrid":
                        self.model.geometry_edit.encoding.encoding.encoding.params.copy_(self.model.geometry.encoding.encoding.encoding.params)
                    else:
                        self.model.geometry_edit.encoding.encoding.params.copy_(self.model.geometry.encoding.encoding.params)

            out = self(batch) #forward pass
            render_img = out['comp_rgb_full'].view( res, res, 3).permute(1,0,2) # (W, H, 3) -> (H, W, 3)
            render_img = render_img.unsqueeze(0).permute(0,3,1,2) # (H, W, 3) -> (1, 3, H, W)
            
            orig_img = None
            if (self.interpolation_step):
                orig_img = out['identity_rgb_full'].view( res, res, 3).permute(1,0,2) # (W, H, 3) -> (H, W, 3)
            else:
                orig_img = batch['rgb'].view( res, res, 3).permute(1,0,2)
            # orig_img = batch['rgb'].view( res, res, 3).permute(1,0,2) # (W, H, 3) -> (H, W, 3)
            orig_img = orig_img.unsqueeze(0).permute(0,3,1,2) # (H, W, 3) -> (1, 3, H, W)
            h, w = orig_img.shape[2:]
           
            l = min(h, w)
            h = int(h * 512 / l)
            w = int(w * 512 / l)  # resize an image such that the smallest length is 512.
            original_image_512 = F.interpolate(orig_img, size=(h, w), mode="bilinear")
            original_image_512 = torch.clamp(original_image_512, 0., 1.)
            src_x0 = self.pds.encode_image(original_image_512)
            
            rendered_image_512 = F.interpolate(render_img, size=(h, w), mode="bilinear")
            x0 = self.pds.encode_image(rendered_image_512)
            dic = self.pds(tgt_x0=x0, src_x0=src_x0, return_dict=True)
            original_pds_loss = dic["loss"] * self.config.system.loss.lambda_pds 
            self.log('train/original_pds_loss', original_pds_loss)
            loss += original_pds_loss 

            del rendered_image_512
            del src_x0, x0, dic
            
            normal_map = out['comp_normal'].view( res, res, 3).permute(1,0,2) # (W, H, 3) -> (H, W, 3)
            
            light_dir = torch.tensor([0.5773, 0.5773, 0.5773], device=normal_map.device).view(1, 1, 3)
            light_dir = F.normalize(light_dir, p=2, dim=-1)

            view_dir = torch.tensor([0.0, 0.0, 1.0],  device=normal_map.device).view(1, 1, 3)  # View direction (camera looking along z-axis)
            view_dir = F.normalize(view_dir, p=2, dim=-1)

            normal_map = F.normalize(normal_map, p=2, dim=-1)
            cos_theta = F.relu(torch.sum(normal_map * light_dir, dim=-1))
            cos_theta = cos_theta.unsqueeze(-1)
            diffuse = self.config.phong.diffuse * cos_theta

            # Specular reflection
            R = 2 * cos_theta * normal_map - light_dir
            cos_alpha = F.relu(torch.sum(R * view_dir, dim=-1))
            cos_alpha = cos_alpha.unsqueeze(-1)
            specular = self.config.phong.specular * cos_alpha ** self.config.phong.shininess

            # Ambient reflection
            ambient = self.config.phong.ambient

            # Combine all three components
            phong_shading = ambient + diffuse + specular
            phong_shading = phong_shading.repeat(1, 1, 3)
            render_phong_img = phong_shading
            render_img = render_phong_img.unsqueeze(0).permute(0,3,1,2) # (H, W, 3) -> (1, 3, H, W)
        
            orig_img = None
            if (self.interpolation_step):
                orig_img = out['identity_rgb_full'].view( res, res, 3).permute(1,0,2) # (W, H, 3) -> (H, W, 3)
            else:
                orig_img = batch['rgb'].view( res, res, 3).permute(1,0,2)
            orig_img = orig_img.unsqueeze(0).permute(0,3,1,2) # (H, W, 3) -> (1, 3, H, W)
            h, w = orig_img.shape[2:]
            
            l = min(h, w)
            h = int(h * 512 / l)
            w = int(w * 512 / l)  # resize an image such that the smallest length is 512.
            original_image_512 = F.interpolate(orig_img, size=(h, w), mode="bilinear")

            original_image_512 = torch.clamp(original_image_512, 0., 1.)
            src_x0 = self.pds.encode_image(original_image_512)
            
            rendered_image_512 = F.interpolate(render_img, size=(h, w), mode="bilinear")
            x0 = self.pds.encode_image(rendered_image_512)
            dic = self.pds(tgt_x0=x0, src_x0=src_x0, return_dict=True)
            original_pds_phong_loss = dic["loss"] * self.config.system.loss.lambda_phong
            self.log('train/original_pds_phong_loss', original_pds_phong_loss)
            loss += original_pds_phong_loss 
            
            del normal_map, light_dir, view_dir, cos_theta, diffuse, R, cos_alpha, specular, ambient, phong_shading
            del render_img, render_phong_img, orig_img, original_image_512, src_x0, x0, dic
                                       
            if(self.var_param_grad_turn_off == False): 
                # keep the identity if the variance is still being learned             
                loss_rgb_mse = 0
                if(self.interpolation_step==False):
                    loss_rgb_mse = F.mse_loss(out['identity_rgb_full'][out['rays_valid_full'][...,0]], batch['rgb'][out['rays_valid_full'][...,0]])
                self.log('train/loss_rgb_mse', loss_rgb_mse)
                loss += loss_rgb_mse * self.C(self.config.system.loss.lambda_rgb_mse)

                loss_rgb_l1 = 0
                if(self.interpolation_step==False):
                    loss_rgb_l1 = F.l1_loss(out['identity_rgb_full'][out['rays_valid_full'][...,0]], batch['rgb'][out['rays_valid_full'][...,0]])
                self.log('train/loss_rgb', loss_rgb_l1)
                loss += loss_rgb_l1 * self.C(self.config.system.loss.lambda_rgb_l1)        

                loss_eikonal = ((torch.linalg.norm(out['sdf_grad_samples'], ord=2, dim=-1) - 1.)**2).mean()
                self.log('train/loss_eikonal', loss_eikonal)
                loss += loss_eikonal * self.C(self.config.system.loss.lambda_eikonal)
                    
            # Optimization mode: seen views or unseen views
            total_optimization = self.config.diffusion.unseen_optimization + self.config.diffusion.seen_optimization
            if (batch_idx % (total_optimization) == 0 and self.config.diffusion.unseen_views):
                self.interpolation_step = True
            elif (batch_idx % (total_optimization) >= self.config.diffusion.unseen_optimization and self.interpolation_step):
                self.interpolation_step = False
                      
            return {
                'loss': loss
            } 
        
        if(self.model.generative_step): # for sanity check, TODO: remove this block?
            self.model.generative_step = False
        out = self(batch) 
        # update train_num_rays
        if self.config.model.dynamic_ray_sampling:
            train_num_rays = int(self.train_num_rays * (self.train_num_samples / out['num_samples_full'].sum().item()))        
            self.train_num_rays = min(int(self.train_num_rays * 0.9 + train_num_rays * 0.1), self.config.model.max_train_num_rays)

        loss_rgb_mse = F.mse_loss(out['identity_rgb_full'][out['rays_valid_full'][...,0]], batch['rgb'][out['rays_valid_full'][...,0]])
        self.log('train/loss_rgb_mse', loss_rgb_mse)
        loss += loss_rgb_mse * self.C(self.config.system.loss.lambda_rgb_mse)

        loss_rgb_l1 = F.l1_loss(out['identity_rgb_full'][out['rays_valid_full'][...,0]], batch['rgb'][out['rays_valid_full'][...,0]])
        self.log('train/loss_rgb', loss_rgb_l1)
        loss += loss_rgb_l1 * self.C(self.config.system.loss.lambda_rgb_l1)        

        loss_eikonal = ((torch.linalg.norm(out['sdf_grad_samples'], ord=2, dim=-1) - 1.)**2).mean()
        self.log('train/loss_eikonal', loss_eikonal)
        loss += loss_eikonal * self.C(self.config.system.loss.lambda_eikonal)
        
        opacity = torch.clamp(out['opacity'].squeeze(-1), 1.e-3, 1.-1.e-3)
        loss_mask = binary_cross_entropy(opacity, batch['fg_mask'].float())
        self.log('train/loss_mask', loss_mask)
        loss += loss_mask * (self.C(self.config.system.loss.lambda_mask) if self.dataset.has_mask else 0.0)

        loss_opaque = binary_cross_entropy(opacity, opacity)
        self.log('train/loss_opaque', loss_opaque)
        loss += loss_opaque * self.C(self.config.system.loss.lambda_opaque)

        loss_sparsity = torch.exp(-self.config.system.loss.sparsity_scale * out['sdf_samples'].abs()).mean()
        self.log('train/loss_sparsity', loss_sparsity)
        loss += loss_sparsity * self.C(self.config.system.loss.lambda_sparsity)

        if self.C(self.config.system.loss.lambda_curvature) > 0:
            assert 'sdf_laplace_samples' in out, "Need geometry.grad_type='finite_difference' to get SDF Laplace samples"
            loss_curvature = out['sdf_laplace_samples'].abs().mean()
            self.log('train/loss_curvature', loss_curvature)
            loss += loss_curvature * self.C(self.config.system.loss.lambda_curvature)

        # distortion loss proposed in MipNeRF360
        # an efficient implementation from https://github.com/sunset1995/torch_efficient_distloss
        if self.C(self.config.system.loss.lambda_distortion) > 0:
            loss_distortion = flatten_eff_distloss(out['weights'], out['points'], out['intervals'], out['ray_indices'])
            self.log('train/loss_distortion', loss_distortion)
            loss += loss_distortion * self.C(self.config.system.loss.lambda_distortion)    

        if self.config.model.learned_background and self.C(self.config.system.loss.lambda_distortion_bg) > 0:
            loss_distortion_bg = flatten_eff_distloss(out['weights_bg'], out['points_bg'], out['intervals_bg'], out['ray_indices_bg'])
            self.log('train/loss_distortion_bg', loss_distortion_bg)
            loss += loss_distortion_bg * self.C(self.config.system.loss.lambda_distortion_bg)        

        losses_model_reg = self.model.regularizations(out)
        for name, value in losses_model_reg.items():
            self.log(f'train/loss_{name}', value)
            loss_ = value * self.C(self.config.system.loss[f"lambda_{name}"])
            loss += loss_
        
        self.log('train/inv_s', out['inv_s'], prog_bar=True)

        for name, value in self.config.system.loss.items():
            if name.startswith('lambda'):
                self.log(f'train_params/{name}', self.C(value))

        self.log('train/num_rays', float(self.train_num_rays), prog_bar=True)       
        
        if (batch_idx>=self.config.diffusion.generation_step and self.generative_step == False):

            self.generative_step = True
            self.model.generative_step = True
            self.var_param_grad_turn_off = True
            self.background_edit = False

            if self.generative_step and self.background_edit==False:
                # turn on edit mode
                for p in self.model.geometry_edit.parameters():
                    p.requires_grad_(True)
                for p in self.model.texture_edit.parameters():
                    p.requires_grad_(True)
                    
                # turn off original scene learning
                for p in self.model.geometry.parameters():
                    p.requires_grad_(False)
                for p in self.model.texture.parameters():
                    p.requires_grad_(False)
                
                # turn off background learning
                if self.config.model.learned_background:
                    for p in self.model.geometry_bg.parameters():
                        p.requires_grad_(False)
                    for p in self.model.texture_bg.parameters():
                        p.requires_grad_(False)

            if self.background_edit:
                # turn off original scene learning
                for p in self.model.geometry.parameters():
                    p.requires_grad_(False)
                for p in self.model.texture.parameters():
                    p.requires_grad_(False)

            
        return {
            'loss': loss
        }
    
    """
    # aggregate outputs from different devices (DP)
    def training_step_end(self, out):
        pass
    """
    
    """
    # aggregate outputs from different iterations
    def training_epoch_end(self, out):
        pass
    """
    
    def validation_step(self, batch, batch_idx):
        if(self.config.cmd_args.get("resume",None)): # sorry for ugly hack, dealing here with unzeroing sdf_grad_edit before generation_step
            if(self.config.diffusion.generation_step > torch.load(self.config.cmd_args.get("resume",None))['global_step']):
                self.model.generative_step = False
        out = self(batch)
        psnr = self.criterions['psnr'](out['comp_rgb_full'].to(batch['rgb']), batch['rgb'])
        W, H = self.dataset.img_wh
        self.save_image_grid(f"it{self.global_step}-{batch['index'][0].item()}.png", [
            {'type': 'rgb', 'img': batch['rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': out['identity_rgb_full'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': out['comp_rgb_full'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}}
        ] + ([
            {'type': 'rgb', 'img': out['comp_rgb_bg'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': out['comp_rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
        ] if self.config.model.learned_background else []) + [
            {'type': 'grayscale', 'img': out['depth'].view(H, W), 'kwargs': {}},
            {'type': 'rgb', 'img': out['comp_normal'].view(H, W, 3), 'kwargs': {'data_format': 'HWC', 'data_range': (-1, 1)}}
        ])
        return {
            'psnr': psnr,
            'index': batch['index']
        }
          
    
    """
    # aggregate outputs from different devices when using DP
    def validation_step_end(self, out):
        pass
    """
    
    def validation_epoch_end(self, out):
        out = self.all_gather(out)
        if self.trainer.is_global_zero:
            out_set = {}
            for step_out in out:
                # DP
                if step_out['index'].ndim == 1:
                    out_set[step_out['index'].item()] = {'psnr': step_out['psnr']}
                # DDP
                else:
                    for oi, index in enumerate(step_out['index']):
                        out_set[index[0].item()] = {'psnr': step_out['psnr'][oi]}
            psnr = torch.mean(torch.stack([o['psnr'] for o in out_set.values()]))
            self.log('val/psnr', psnr, prog_bar=True, rank_zero_only=True)         

    def test_step(self, batch, batch_idx):
        if(self.config.cmd_args.get("resume",None)): # sorry for ugly hack, dealing here with unzeroing sdf_grad_edit before generation_step
            if(self.config.diffusion.generation_step > torch.load(self.config.cmd_args.get("resume",None))['global_step']):
                self.model.generative_step = False

        out = self(batch)
        psnr = self.criterions['psnr'](out['comp_rgb_full'].to(batch['rgb']), batch['rgb'])
        W, H = self.dataset.img_wh
        self.save_image_grid(f"it{self.global_step}-test/{batch['index'][0].item()}.png", [
            {'type': 'rgb', 'img': out['identity_rgb_full'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': out['comp_rgb_full'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}}
        ] + ([
            #{'type': 'rgb', 'img': out['comp_rgb_bg'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            # {'type': 'rgb', 'img': out['comp_rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
        ] if self.config.model.learned_background else []) + [
            # {'type': 'grayscale', 'img': out['depth'].view(H, W), 'kwargs': {}},
            # {'type': 'rgb', 'img': out['comp_normal'].view(H, W, 3), 'kwargs': {'data_format': 'HWC', 'data_range': (-1, 1)}}
        ])
        return {
            'psnr': psnr,
            'index': batch['index']
        }      
    
    def test_epoch_end(self, out):
        """
        Synchronize devices.
        Generate image sequence using test outputs.
        """
        out = self.all_gather(out)
        if self.trainer.is_global_zero:
            out_set = {}
            for step_out in out:
                # DP
                if step_out['index'].ndim == 1:
                    out_set[step_out['index'].item()] = {'psnr': step_out['psnr']}
                # DDP
                else:
                    for oi, index in enumerate(step_out['index']):
                        out_set[index[0].item()] = {'psnr': step_out['psnr'][oi]}
            psnr = torch.mean(torch.stack([o['psnr'] for o in out_set.values()]))
            self.log('test/psnr', psnr, prog_bar=True, rank_zero_only=True)    

            self.save_img_sequence(
                f"it{self.global_step}-test",
                f"it{self.global_step}-test",
                '(\d+)\.png',
                save_format='mp4',
                fps=30
            )
            self.export()
    
    def export(self):
        mesh = self.model.export(self.config.export)
        self.save_mesh(
            f"it{self.global_step}-{self.config.model.geometry.isosurface.method}{self.config.model.geometry.isosurface.resolution}.obj",
            **mesh
        )