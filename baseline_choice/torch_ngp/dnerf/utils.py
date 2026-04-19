from nerf.utils import *
from nerf.utils import Trainer as _Trainer


class Trainer(_Trainer):
    def __init__(self, 
                 name,                          
                 opt,             
                 model,           
                 criterion=None,                                                                     
                 optimizer=None,            
                 ema_decay=None,                            
                 lr_scheduler=None,            
                 metrics=[],                                                                                                   
                 local_rank=0,                 
                 world_size=1,                    
                 device=None,                                                                     
                 mute=False,                            
                 fp16=False,                     
                 eval_interval=1,                          
                 max_keep_ckpt=2,                                 
                 workspace='workspace',                                 
                 best_mode='min',                                        
                 use_loss_as_metric=True,                               
                 report_metric_at_train=False,                                  
                 use_checkpoint="latest",                                 
                 use_tensorboardX=True,                                         
                 scheduler_update_every_step=False,                                                          
                 ):

        self.optimizer_fn = optimizer
        self.lr_scheduler_fn = lr_scheduler

        super().__init__(name, opt, model, criterion, optimizer, ema_decay, lr_scheduler, metrics, local_rank, world_size, device, mute, fp16, eval_interval, max_keep_ckpt, workspace, best_mode, use_loss_as_metric, report_metric_at_train, use_checkpoint, use_tensorboardX, scheduler_update_every_step)
        
                                       

    def train_step(self, data):

        rays_o = data['rays_o']            
        rays_d = data['rays_d']            
        time = data['time']         

                                                           
        if 'images' not in data:

            B, N = rays_o.shape[:2]
            H, W = data['H'], data['W']

                                                          
            outputs = self.model.render(rays_o, rays_d, time, staged=False, bg_color=None, perturb=True, force_all_rays=True, **vars(self.opt))
            pred_rgb = outputs['image'].reshape(B, H, W, 3).permute(0, 3, 1, 2).contiguous()

                                                                     
                                      

            loss = self.clip_loss(pred_rgb)
            
            return pred_rgb, None, loss

        images = data['images']              

        B, N, C = images.shape

        if self.opt.color_space == 'linear':
            images[..., :3] = srgb_to_linear(images[..., :3])

        if C == 3 or self.model.bg_radius > 0:
            bg_color = 1
                                                                                           
        else:
                                                                                       
                                                                                   
            bg_color = torch.rand_like(images[..., :3])                             

        if C == 4:
            gt_rgb = images[..., :3] * images[..., 3:] + bg_color * (1 - images[..., 3:])
        else:
            gt_rgb = images

        outputs = self.model.render(rays_o, rays_d, time, staged=False, bg_color=bg_color, perturb=True, force_all_rays=False, **vars(self.opt))
    
        pred_rgb = outputs['image']

        loss = self.criterion(pred_rgb, gt_rgb).mean(-1)                       

                                                          
        if len(loss.shape) == 3:            
            loss = loss.mean(0)

                          
        if self.error_map is not None:
            index = data['index']      
            inds = data['inds_coarse']         

                                                                                 
            error_map = self.error_map[index]             

                                                               
                                              
                                                                 
                                                                                   
                                                                   
                                                                                                                    

            error = loss.detach().to(error_map.device)                            
            
                        
            ema_error = 0.1 * error_map.gather(1, inds) + 0.9 * error
            error_map.scatter_(1, inds, ema_error)

                      
            self.error_map[index] = error_map

        loss = loss.mean()

                               
        if 'deform' in outputs and outputs['deform'] is not None:
            loss = loss + 1e-3 * outputs['deform'].abs().mean()
        
        return pred_rgb, gt_rgb, loss

    def eval_step(self, data):

        rays_o = data['rays_o']            
        rays_d = data['rays_d']            
        time = data['time']         
        images = data['images']                 
        B, H, W, C = images.shape

        if self.opt.color_space == 'linear':
            images[..., :3] = srgb_to_linear(images[..., :3])

                                          
        bg_color = 1
        if C == 4:
            gt_rgb = images[..., :3] * images[..., 3:] + bg_color * (1 - images[..., 3:])
        else:
            gt_rgb = images
        
        outputs = self.model.render(rays_o, rays_d, time, staged=True, bg_color=bg_color, perturb=False, **vars(self.opt))

        pred_rgb = outputs['image'].reshape(B, H, W, 3)
        pred_depth = outputs['depth'].reshape(B, H, W)

        loss = self.criterion(pred_rgb, gt_rgb).mean()

        return pred_rgb, pred_depth, gt_rgb, loss

                                                                 
    def test_step(self, data, bg_color=None, perturb=False):  

        rays_o = data['rays_o']            
        rays_d = data['rays_d']            
        time = data['time']         
        H, W = data['H'], data['W']

        if bg_color is not None:
            bg_color = bg_color.to(self.device)

        outputs = self.model.render(rays_o, rays_d, time, staged=True, bg_color=bg_color, perturb=perturb, **vars(self.opt))

        pred_rgb = outputs['image'].reshape(-1, H, W, 3)
        pred_depth = outputs['depth'].reshape(-1, H, W)

        return pred_rgb, pred_depth

                                  
    def test_gui(self, pose, intrinsics, W, H, time=0, bg_color=None, spp=1, downscale=1):
        
                                                                         
        rH = int(H * downscale)
        rW = int(W * downscale)
        intrinsics = intrinsics * downscale

        pose = torch.from_numpy(pose).unsqueeze(0).to(self.device)

        rays = get_rays(pose, intrinsics, rH, rW, -1)

        data = {
            'time': torch.FloatTensor([[time]]).to(self.device),                                
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'H': rH,
            'W': rW,
        }
        
        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.fp16):
                                                          
                preds, preds_depth = self.test_step(data, bg_color=bg_color, perturb=spp)

        if self.ema is not None:
            self.ema.restore()

                                                  
        if downscale != 1:
                                                       
            preds = F.interpolate(preds.permute(0, 3, 1, 2), size=(H, W), mode='nearest').permute(0, 2, 3, 1).contiguous()
            preds_depth = F.interpolate(preds_depth.unsqueeze(1), size=(H, W), mode='nearest').squeeze(1)

        if self.opt.color_space == 'linear':
            preds = linear_to_srgb(preds)

        pred = preds[0].detach().cpu().numpy()
        pred_depth = preds_depth[0].detach().cpu().numpy()

        outputs = {
            'image': pred,
            'depth': pred_depth,
        }

        return outputs        

    def save_mesh(self, time, save_path=None, resolution=256, threshold=10):
                                
        time = torch.FloatTensor([[time]]).to(self.device)

        if save_path is None:
            save_path = os.path.join(self.workspace, 'meshes', f'{self.name}_{self.epoch}.ply')

        self.log(f"==> Saving mesh to {save_path}")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        def query_func(pts):
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    sigma = self.model.density(pts.to(self.device), time)['sigma']
            return sigma

        vertices, triangles = extract_geometry(self.model.aabb_infer[:3], self.model.aabb_infer[3:], resolution=resolution, threshold=threshold, query_func=query_func)

        mesh = trimesh.Trimesh(vertices, triangles, process=False)                                                
        mesh.export(save_path)

        self.log(f"==> Finished saving mesh.")