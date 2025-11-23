import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models.base import BaseModel
from models.utils import chunk_batch, scale_anything, cleanup
from systems.utils import update_module_step
from nerfacc import ContractionType, OccupancyGrid, ray_marching, render_weight_from_density, render_weight_from_alpha, accumulate_along_rays
from nerfacc.intersection import ray_aabb_intersect
from models.geometry import contract_to_unisphere

class VarianceNetwork(nn.Module):
    def __init__(self, config):
        super(VarianceNetwork, self).__init__()
        self.config = config
        self.init_val = self.config.init_val
        self.register_parameter('variance', nn.Parameter(torch.tensor(self.config.init_val)))
        self.modulate = self.config.get('modulate', False)
        if self.modulate:
            self.mod_start_steps = self.config.mod_start_steps
            self.reach_max_steps = self.config.reach_max_steps
            self.max_inv_s = self.config.max_inv_s
    
    @property
    def inv_s(self):
        val = torch.exp(self.variance * 10.0)
        if self.modulate and self.do_mod:
            val = val.clamp_max(self.mod_val)
        return val

    def forward(self, x):
        return torch.ones([len(x), 1], device=self.variance.device) * self.inv_s
    
    def update_step(self, epoch, global_step):
        if self.modulate:
            self.do_mod = global_step > self.mod_start_steps
            if not self.do_mod:
                self.prev_inv_s = self.inv_s.item()
            else:
                self.mod_val = min((global_step / self.reach_max_steps) * (self.max_inv_s - self.prev_inv_s) + self.prev_inv_s, self.max_inv_s)


@models.register('neus')
class NeuSModel(BaseModel):
    def setup(self):
        self.geometry = models.make(self.config.geometry.name, self.config.geometry)
        self.texture = models.make(self.config.texture.name, self.config.texture)
        self.geometry.contraction_type = ContractionType.AABB

        # Set for editting
        self.geometry_edit = models.make(self.config.geometry_edit.name, self.config.geometry_edit)
        self.texture_edit = models.make(self.config.texture_edit.name, self.config.texture_edit)
        self.geometry_edit.contraction_type = ContractionType.AABB

        self.generative_step = True

        if self.config.learned_background:
            self.geometry_bg = models.make(self.config.geometry_bg.name, self.config.geometry_bg)
            self.texture_bg = models.make(self.config.texture_bg.name, self.config.texture_bg)
            self.geometry_bg.contraction_type = ContractionType.UN_BOUNDED_SPHERE
            self.near_plane_bg, self.far_plane_bg = 0.1, 1e3
            self.cone_angle_bg = 10**(math.log10(self.far_plane_bg) / self.config.num_samples_per_ray_bg) - 1.
            self.render_step_size_bg = 0.01 

            # Set for editting
            # self.geometry_bg_edit = models.make(self.config.geometry_bg_edit.name, self.config.geometry_bg_edit)
            # self.texture_bg_edit = models.make(self.config.texture_bg_edit.name, self.config.texture_bg_edit)
            # self.geometry_bg_edit.contraction_type = ContractionType.UN_BOUNDED_SPHERE
           

        self.variance = VarianceNetwork(self.config.variance)
        self.register_buffer('scene_aabb', torch.as_tensor([-self.config.radius, -self.config.radius, -self.config.radius, self.config.radius, self.config.radius, self.config.radius], dtype=torch.float32))
        if self.config.grid_prune:
            self.occupancy_grid = OccupancyGrid(
                roi_aabb=self.scene_aabb,
                resolution=128,
                contraction_type=ContractionType.AABB
            )
            if self.config.learned_background:
                self.occupancy_grid_bg = OccupancyGrid(
                    roi_aabb=self.scene_aabb,
                    resolution=256,
                    contraction_type=ContractionType.UN_BOUNDED_SPHERE
                )
        self.randomized = self.config.randomized
        self.background_color = None
        self.render_step_size = 1.732 * 2 * self.config.radius / self.config.num_samples_per_ray
    
    def update_step(self, epoch, global_step):
        update_module_step(self.geometry, epoch, global_step)
        update_module_step(self.texture, epoch, global_step)
        
        # Set for editting
        update_module_step(self.geometry_edit, epoch, global_step)
        update_module_step(self.texture_edit, epoch, global_step)
        
        if self.config.learned_background:
            update_module_step(self.geometry_bg, epoch, global_step)
            update_module_step(self.texture_bg, epoch, global_step)
            
            # Set for editting
            # update_module_step(self.geometry_bg_edit, epoch, global_step)
            # update_module_step(self.texture_bg_edit, epoch, global_step)
        
        update_module_step(self.variance, epoch, global_step)

        cos_anneal_end = self.config.get('cos_anneal_end', 0)
        self.cos_anneal_ratio = 1.0 if cos_anneal_end == 0 else min(1.0, global_step / cos_anneal_end)

        def occ_eval_fn(x):
            # sdf = self.geometry(x, with_grad=False, with_feature=False)
            sdf, feature = self.geometry(x, with_grad=False, with_feature=True)
            # Set for editting
            # sdf_edit = self.geometry_edit(x, with_grad=False, with_feature=False)
            sdf_edit = self.geometry_edit(x, feature, with_grad=False, with_feature=False)
            sdf_sum = sdf_edit

            inv_s = self.variance(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)
            inv_s = inv_s.expand(sdf_sum.shape[0], 1)
            estimated_next_sdf = sdf_sum[...,None] - self.render_step_size * 0.5
            estimated_prev_sdf = sdf_sum[...,None] + self.render_step_size * 0.5
            prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
            next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)
            p = prev_cdf - next_cdf
            c = prev_cdf
            alpha = ((p + 1e-5) / (c + 1e-5)).view(-1, 1).clip(0.0, 1.0)
            return alpha
        
        def occ_eval_fn_bg(x):
            density, _ = self.geometry_bg(x)
            # Set for editting
            # density_edit, _ = self.geometry_bg_edit(x)
            density_sum = density

            # approximate for 1 - torch.exp(-density[...,None] * self.render_step_size_bg) based on taylor series
            return density_sum[...,None] * self.render_step_size_bg
        
        if self.training and self.config.grid_prune:
            self.occupancy_grid.every_n_step(step=global_step, occ_eval_fn=occ_eval_fn, occ_thre=self.config.get('grid_prune_occ_thre', 0.01))
            if self.config.learned_background:
                self.occupancy_grid_bg.every_n_step(step=global_step, occ_eval_fn=occ_eval_fn_bg, occ_thre=self.config.get('grid_prune_occ_thre_bg', 0.01))

    def isosurface(self):
        mesh = self.geometry.isosurface()
        return mesh

    def get_alpha(self, sdf, normal, dirs, dists):
        inv_s = self.variance(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)           # Single parameter
        inv_s = inv_s.expand(sdf.shape[0], 1)

        true_cos = (dirs * normal).sum(-1, keepdim=True)

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - self.cos_anneal_ratio) +
                     F.relu(-true_cos) * self.cos_anneal_ratio)  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf[...,None] + iter_cos * dists.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf[...,None] - iter_cos * dists.reshape(-1, 1) * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).view(-1).clip(0.0, 1.0)
        return alpha

    def forward_bg_(self, rays):
        n_rays = rays.shape[0]
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6] # both (N_rays, 3)

        def sigma_fn(t_starts, t_ends, ray_indices):
            ray_indices = ray_indices.long()
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends) / 2.
            density, _ = self.geometry_bg(positions)
            
            return density[...,None]            

        _, t_max = ray_aabb_intersect(rays_o, rays_d, self.scene_aabb)
        # if the ray intersects with the bounding box, start from the farther intersection point
        # otherwise start from self.far_plane_bg
        # note that in nerfacc t_max is set to 1e10 if there is no intersection
        near_plane = torch.where(t_max > 1e9, self.near_plane_bg, t_max)
        with torch.no_grad():
            ray_indices, t_starts, t_ends = ray_marching(
                rays_o, rays_d,
                scene_aabb=None,
                grid=self.occupancy_grid_bg if self.config.grid_prune else None,
                sigma_fn=sigma_fn,
                near_plane=near_plane, far_plane=self.far_plane_bg,
                render_step_size=self.render_step_size_bg,
                stratified=self.randomized,
                cone_angle=self.cone_angle_bg,
                alpha_thre=0.0
            )       
        
        ray_indices = ray_indices.long()
        t_origins = rays_o[ray_indices]
        t_dirs = rays_d[ray_indices]
        midpoints = (t_starts + t_ends) / 2.
        positions = t_origins + t_dirs * midpoints  
        intervals = t_ends - t_starts

        density, feature = self.geometry_bg(positions)
        rgb = self.texture_bg(feature, t_dirs)
        weights = render_weight_from_density(t_starts, t_ends, density[...,None], ray_indices=ray_indices, n_rays=n_rays)
        opacity = accumulate_along_rays(weights, ray_indices, values=None, n_rays=n_rays)
        depth = accumulate_along_rays(weights, ray_indices, values=midpoints, n_rays=n_rays)
        comp_rgb = accumulate_along_rays(weights, ray_indices, values=rgb, n_rays=n_rays)
        comp_rgb = comp_rgb + self.background_color * (1.0 - opacity)       

        out = {
            'comp_rgb': comp_rgb,
            'opacity': opacity,
            'depth': depth,
            'rays_valid': opacity > 0,
            'num_samples': torch.as_tensor([len(t_starts)], dtype=torch.int32, device=rays.device)
        }

        if self.training:
            out.update({
                'weights': weights.view(-1),
                'points': midpoints.view(-1),
                'intervals': intervals.view(-1),
                'ray_indices': ray_indices.view(-1)
            })

        return out

    def forward_(self, rays):
        n_rays = rays.shape[0]
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6] # both (N_rays, 3)

        with torch.no_grad():
            ray_indices, t_starts, t_ends = ray_marching(
                rays_o, rays_d,
                scene_aabb=self.scene_aabb,
                grid=self.occupancy_grid if self.config.grid_prune else None,
                alpha_fn=None,
                near_plane=None, far_plane=None,
                render_step_size=self.render_step_size,
                stratified=self.randomized,
                cone_angle=0.0,
                alpha_thre=0.0
            )
        
        ray_indices = ray_indices.long()
        t_origins = rays_o[ray_indices]
        t_dirs = rays_d[ray_indices]
        midpoints = (t_starts + t_ends) / 2.
        positions = t_origins + t_dirs * midpoints
        dists = t_ends - t_starts
        sdf_laplace_edit = None # set to None for now TODO: remove this line?
        if self.config.geometry.grad_type == 'finite_difference':
            sdf, sdf_grad, feature, sdf_laplace, feature_grad = self.geometry(positions, with_grad=True, with_feature=True, with_laplace=True, with_feature_grad=True)
            # Set for editting
            sdf_edit, sdf_grad_edit, feature_edit, sdf_laplace_edit = self.geometry_edit(positions, feature, with_grad=True, with_feature=True, with_laplace=True,feature_grad=feature_grad)

            # sdf_edit, sdf_grad_edit, feature_edit, sdf_laplace_edit = self.geometry_edit(positions, with_grad=True, with_feature=True, with_laplace=True)
            # sdf_laplace_sum = sdf_laplace + sdf_laplace_edit
        else:
            sdf, sdf_grad, feature = self.geometry(positions, with_grad=True, with_feature=True)
            # Set for editting
            sdf_edit, sdf_grad_edit, feature_edit = self.geometry_edit(positions,  feature, with_grad=True, with_feature=True)
            #sdf_edit, sdf_grad_edit, feature_edit = self.geometry_edit(positions, with_grad=True, with_feature=True)
               
       
        sdf_sum = sdf_edit
        feature_sum = feature_edit
        sdf_laplace_sum = sdf_laplace_edit # set to None for now TODO: remove this line?

        sdf_grad_sum = sdf_grad_edit #TODO this is dangerous/ somehow the sdf_grad_edit set to zero while running in training
        #workaround: use sdf_grad_sum = sdf_grad for non-genrative training validation and testing
        # import pdb; pdb.set_trace()
        # print("SDF_grad_edit: ", sdf_grad_edit, "SDF_grad: ", sdf_grad)
        # if(self.generative_step == False and self.training == False): # sorry for ugly hack
        #     print("Warning: sdf_grad_edit set to zero")
        #     print("SDF_grad_edit: ", sdf_grad_edit, "SDF_grad: ", sdf_grad)
        #     sdf_grad_sum = sdf_grad
            
        # testing removing view dependency 
        normal = F.normalize(sdf_grad, p=2, dim=-1)
        rgb = self.texture(feature, t_dirs, normal)
        
        alpha           = self.get_alpha(sdf, normal, t_dirs, dists)[...,None]
        weights         = render_weight_from_alpha(alpha, ray_indices=ray_indices, n_rays=n_rays)
        opacity_orig    = accumulate_along_rays(weights, ray_indices, values=None, n_rays=n_rays)
        depth_orig      = accumulate_along_rays(weights, ray_indices, values=midpoints, n_rays=n_rays)
        identity_rgb    = accumulate_along_rays(weights, ray_indices, values=rgb, n_rays=n_rays)

        # Set for editting
        dir_zero = torch.zeros_like(t_dirs) # set the viewing directions to zero
        normal = F.normalize(sdf_grad_sum, p=2, dim=-1)
        rgb_edit = self.texture_edit(feature_sum, dir_zero, normal)
        rgb_sum = rgb + rgb_edit
        
        alpha           = self.get_alpha(sdf_sum, normal, t_dirs, dists)[...,None]
        weights         = render_weight_from_alpha(alpha, ray_indices=ray_indices, n_rays=n_rays)
        opacity         = accumulate_along_rays(weights, ray_indices, values=None, n_rays=n_rays)
        depth           = accumulate_along_rays(weights, ray_indices, values=midpoints, n_rays=n_rays)
        comp_rgb        = accumulate_along_rays(weights, ray_indices, values=rgb_sum, n_rays=n_rays)

        comp_normal     = accumulate_along_rays(weights, ray_indices, values=normal, n_rays=n_rays)
        comp_normal     = F.normalize(comp_normal, p=2, dim=-1)

        out = {
            'identity_rgb': identity_rgb,
            'opacity_orig': opacity_orig,
            'comp_rgb': comp_rgb,
            'comp_normal': comp_normal,
            'opacity': opacity,
            'depth': depth,
            'depth_orig': depth_orig,
            'rays_valid': opacity > 0,
            'num_samples': torch.as_tensor([len(t_starts)], dtype=torch.int32, device=rays.device)
        }

        if self.training:
            out.update({
                'sdf_samples': sdf_sum,
                'sdf_grad_samples': sdf_grad_sum,
                'sdf_grad_identity': sdf_grad,
                'weights': weights.view(-1),
                'points': midpoints.view(-1),
                'intervals': dists.view(-1),
                'ray_indices': ray_indices.view(-1)                
            })
            if self.config.geometry.grad_type == 'finite_difference':
                out.update({
                    'sdf_laplace_samples': sdf_laplace_sum
                })

        if self.config.learned_background:
            out_bg = self.forward_bg_(rays)
        else:
            out_bg = {
                'comp_rgb': self.background_color[None,:].expand(*comp_rgb.shape),
                'num_samples': torch.zeros_like(out['num_samples']),
                'rays_valid': torch.zeros_like(out['rays_valid'])
            }      

        out_full = {
            # everytime i think about this, i get confused about the correct way to combine the background and the object
            # at the beginning it seems intuitive, but when i think about it more, it start hurting my brain
            # everytime i come back to it, i do not remember what i was thinking about then it starts hurting my brain again [unhealthy cycle] :)
            'comp_rgb': out['comp_rgb'] + out_bg['comp_rgb'] * (1.0 - out['opacity']), # not sure if this is correct, probably should be 'opacity'
            'identity_rgb': out['identity_rgb'] + out_bg['comp_rgb'] * (1.0 - out['opacity_orig']), # not sure if this is correct, probably should be 'opacity_orig'
            'num_samples': out['num_samples'] + out_bg['num_samples'],
            'rays_valid': out['rays_valid'] | out_bg['rays_valid']
        }

        return {
            **out,
            **{k + '_bg': v for k, v in out_bg.items()},
            **{k + '_full': v for k, v in out_full.items()}
        }

    def forward(self, rays):
        if self.training:
            out = self.forward_(rays)
        else:
            out = chunk_batch(self.forward_, self.config.ray_chunk, True, rays)
        return {
            **out,
            'inv_s': self.variance.inv_s
        }

    def train(self, mode=True):
        self.randomized = mode and self.config.randomized
        return super().train(mode=mode)
    
    def eval(self):
        self.randomized = False
        return super().eval()
    
    def regularizations(self, out):
        losses = {}
        losses.update(self.geometry.regularizations(out))
        losses.update(self.texture.regularizations(out))
        # Set for editting
        losses.update(self.geometry_edit.regularizations(out))
        losses.update(self.texture_edit.regularizations(out))
        return losses
    
    @torch.no_grad()
    def isosurface2_(self, vmin, vmax):
        def batch_func(x):
            x = torch.stack([
                scale_anything(x[...,0], (0, 1), (vmin[0], vmax[0])),
                scale_anything(x[...,1], (0, 1), (vmin[1], vmax[1])),
                scale_anything(x[...,2], (0, 1), (vmin[2], vmax[2])),
            ], dim=-1).to(self.rank)
            points = contract_to_unisphere(x, self.geometry.radius, self.geometry.contraction_type) # points normalized to (0, 1)
            out = self.geometry.network(self.geometry.encoding(points.view(-1, 3))).view(*points.shape[:-1], self.geometry.n_output_dims).float()
            sdf, feature = out[...,0], out
            out = self.geometry_edit.network(self.geometry_edit.encoding(points.view(-1, 3))).view(*points.shape[:-1], self.geometry_edit.n_output_dims).float()
            out = torch.cat([feature, out], dim=-1)
            out = torch.cat([out,self.geometry_edit.encoding(points.view(-1, 3))], dim=-1) # Condition the network on the position encoding 
            out = (self.geometry_edit.network_edit(out)).view(*points.shape[:-1], self.geometry_edit.n_output_dims) + feature
            sdf_edit = out[...,0]
            rv_sum = sdf_edit.cpu()
            cleanup()
            return rv_sum

        levelsum = chunk_batch(batch_func, self.geometry.config.isosurface.chunk, True, self.geometry.helper.grid_vertices())
        mesh = self.geometry.helper(levelsum, threshold=self.geometry.config.isosurface.threshold)
        mesh['v_pos'] = torch.stack([
            scale_anything(mesh['v_pos'][...,0], (0, 1), (vmin[0], vmax[0])),
            scale_anything(mesh['v_pos'][...,1], (0, 1), (vmin[1], vmax[1])),
            scale_anything(mesh['v_pos'][...,2], (0, 1), (vmin[2], vmax[2]))
        ], dim=-1)
       
        return mesh
    
    @torch.no_grad()
    def export(self, export_config):
        mesh_coarse = self.isosurface2_((-self.geometry.radius, -self.geometry.radius, -self.geometry.radius), (self.geometry.radius, self.geometry.radius, self.geometry.radius))
        vmin, vmax = mesh_coarse['v_pos'].amin(dim=0), mesh_coarse['v_pos'].amax(dim=0)
        vmin_ = (vmin - (vmax - vmin) * 0.1).clamp(-self.geometry.radius, self.geometry.radius)
        vmax_ = (vmax + (vmax - vmin) * 0.1).clamp(-self.geometry.radius, self.geometry.radius)
        mesh = self.isosurface2_(vmin_, vmax_)
        if export_config.export_vertex_color:
            if self.config.geometry.grad_type == 'analytic':
                _, sdf_grad, feature = chunk_batch(self.geometry, export_config.chunk_size, False, mesh['v_pos'].to(self.rank), with_grad=True, with_feature=True)
                _, sdf_grad_edit, feature_edit = chunk_batch(self.geometry_edit, export_config.chunk_size, False, mesh['v_pos'].to(self.rank),feature, with_grad=True, with_feature=True)
            else:
                _, sdf_grad, feature, sdf_laplace, feature_grad = chunk_batch(self.geometry, export_config.chunk_size, False, mesh['v_pos'].to(self.rank), with_grad=True, with_feature=True,with_laplace=True, with_feature_grad=True)
                _, sdf_grad_edit, feature_edit, sdf_laplace_edit = chunk_batch(self.geometry_edit, export_config.chunk_size, False, mesh['v_pos'].to(self.rank), feature, with_grad=True, with_feature=True, with_laplace=True, feature_grad=feature_grad)
            sdf_grad_sum = sdf_grad_edit  
            feature_sum = feature_edit 
            normal = F.normalize(sdf_grad, p=2, dim=-1)
            normal_sum = F.normalize(sdf_grad_sum, p=2, dim=-1)

            dir_zero = torch.zeros_like(normal_sum)
            rgb = self.texture(feature, -normal, normal) # set the viewing directions to the normal to get "albedo"
            rgb_edit = self.texture_edit(feature_sum, dir_zero, normal_sum) # set the viewing directions to zero to get view independent color
            rgb_sum = rgb + rgb_edit 
            # clip the color to [0, 1]
            mesh['v_rgb'] = rgb_sum.cpu().clamp(0.0, 1.0)
        return mesh