import torch
import torch.nn as nn
import numpy as np


class PointPillarScatter(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.nx, self.ny, self.nz = grid_size
        assert self.nz == 1
        self.blocks = nn.ModuleList()
        cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(64, 8, kernel_size=3,stride=1, padding=0, bias=False),
                nn.BatchNorm2d(8, eps=1e-3, momentum=0.01),
                nn.ReLU(),
                nn.ZeroPad2d(1),
                nn.Conv2d(8, 1, kernel_size=3,stride=1, padding=0, bias=False),
                nn.BatchNorm2d(1, eps=1e-3, momentum=0.01),
                nn.Sigmoid()
            ]
        self.blocks.append(nn.Sequential(*cur_layers))

    def forward(self, batch_dict, **kwargs):
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        pillar_cls_label = batch_dict['voxel_cls'] #[31781, 32]
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1
        batch_indices = []
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)
            batch_indices.append(indices)


        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)

        spatial_features = batch_spatial_features
        spatial_features_1 = batch_spatial_features
        # print(spatial_features.shape)
        spatial_features = self.blocks[0](spatial_features)

        batch_spatial_features_output = []
        select_cls_label_output = []
        pre_score = []
        for batch_idx in range(batch_size):
            batch_spatial_features_reduce = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)
            batch_mask = coords[:, 0] == batch_idx
            cls_label = pillar_cls_label[batch_mask,:]
            indice = batch_indices[batch_idx]
            features = spatial_features[batch_idx,:,:,:].squeeze(0).view(-1) #可能有错
            features = features[indice]
            score, index = features.topk(2000,dim=0,largest=True)
            select_cls_label = cls_label[index,:]
            select_cls_label = select_cls_label.sum(dim=1, keepdim=True)
            select_cls_label_output.append(select_cls_label.unsqueeze(dim=0).cpu().detach().numpy())
            pre_score.append(score.unsqueeze(dim=0).cpu().detach().numpy())
            
            pillars_index = indice[index]
            spatial_features_1 = batch_spatial_features[batch_idx,:,:,:].view(64,-1)
            batch_spatial_features_reduce[:, pillars_index] = spatial_features_1[:,pillars_index]
            batch_spatial_features_output.append(batch_spatial_features_reduce)

        batch_spatial_features_output = torch.stack(batch_spatial_features_output, 0)
        batch_spatial_features_output = batch_spatial_features_output.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
        batch_spatial_features = batch_spatial_features_output
        select_cls_label_output = torch.tensor(np.concatenate(select_cls_label_output,axis=0)).cuda()
        pre_score = torch.tensor(np.concatenate(pre_score,axis=0)).cuda()

        batch_dict.update({'select_cls_label_output':select_cls_label_output})
        batch_dict.update({'pre_score':pre_score})



        batch_dict['spatial_features'] = batch_spatial_features
        return batch_dict
