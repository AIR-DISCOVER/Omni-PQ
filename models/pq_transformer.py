
import torch
import torch.nn.functional as F
from torch import nn
import sys
import os
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
from backbone_module import Pointnet2Backbone
from transformer import TransformerDecoderLayer
from utils.pointnet_util import FPSModule,PointsObjClsModule,GeneralSamplingModule
from pointnet2_modules import PointnetSAModuleVotes
from voting_module import VotingModule

class PositionEmbeddingLearned(nn.Module):
    """
    
    Absolute pos embedding, learned.
    """
    def __init__(self, input_channel, num_pos_feats=288):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Conv1d(input_channel, num_pos_feats, kernel_size=1),
            nn.BatchNorm1d(num_pos_feats),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_pos_feats, num_pos_feats, kernel_size=1))

    def forward(self, xyz):
        xyz = xyz.transpose(1, 2).contiguous()
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding

def decode_scores(base_xyz,objectness_scores, center, heading_scores, heading_residuals_normalized, size_scores, size_residuals_normalized,
            sem_cls_scores, end_points, num_class, num_heading_bin, num_size_cluster, mean_size_arr,prefix):
    batch_size = objectness_scores.shape[0]
    num_proposal = objectness_scores.shape[1]
    end_points[f'{prefix}objectness_scores'] = objectness_scores    
    end_points[f'{prefix}center'] = center # (batch_size, num_proposal, 3)
    end_points[f'{prefix}heading_scores'] = heading_scores # Bxnum_proposalxnum_heading_bin
    end_points[f'{prefix}heading_residuals_normalized'] = heading_residuals_normalized # Bxnum_proposalxnum_heading_bin (should be -1 to 1)
    end_points[f'{prefix}heading_residuals'] = heading_residuals_normalized * (np.pi/num_heading_bin) # Bxnum_proposalxnum_heading_bin
    end_points[f'{prefix}size_scores'] = size_scores
    size_residuals_normalized = size_residuals_normalized.view([batch_size, num_proposal, num_size_cluster, 3])
    end_points[f'{prefix}size_residuals_normalized'] = size_residuals_normalized
    mean_size_arr = torch.from_numpy(mean_size_arr.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(0) 
    size_residuals = size_residuals_normalized * mean_size_arr
    end_points[f'{prefix}size_residuals'] = size_residuals
    size_recover = size_residuals + mean_size_arr
    pred_size_class = torch.argmax(size_scores, -1)
    pred_size_class = pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, 3)
    pred_size = torch.gather(size_recover, 2, pred_size_class) 
    pred_size = pred_size.squeeze_(2)
    end_points[f'{prefix}pred_size'] = pred_size
    end_points[f'{prefix}sem_cls_scores'] = sem_cls_scores
    return end_points,pred_size


class PredictHead(nn.Module):
    def __init__(self,hidden_dim,num_heading_bin,num_size_cluster,num_class,mean_size_arr): 
        super().__init__()

        self.num_class = num_class
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        self.num_heading_bin = num_heading_bin
        self.objectness_scores_head = torch.nn.Conv1d(hidden_dim, 2, 1)
        self.center_head = torch.nn.Conv1d(hidden_dim, 3, 1)
        self.heading_class_head = torch.nn.Conv1d(hidden_dim, num_heading_bin, 1)
        self.heading_residual_head = torch.nn.Conv1d(hidden_dim, num_heading_bin, 1)
        self.size_class_head = torch.nn.Conv1d(hidden_dim, num_size_cluster, 1)
        self.size_residual_head = torch.nn.Conv1d(hidden_dim, num_size_cluster * 3, 1)
        self.sem_cls_scores_head = torch.nn.Conv1d(hidden_dim,num_class, 1)
        self.conv1 = torch.nn.Conv1d(hidden_dim,hidden_dim,1)
        self.conv2 = torch.nn.Conv1d(hidden_dim,hidden_dim,1)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
    def forward(self,net,base_xyz,end_points,prefix):
        net = F.relu(self.bn1(self.conv1(net)))
        net = F.relu(self.bn2(self.conv2(net)))
        objectness_scores = self.objectness_scores_head(net).transpose(2, 1)  # (batch_size, num_proposal, 1)
        center = self.center_head(net).transpose(2, 1) + base_xyz # (batch_size, num_proposal, 3)
        heading_scores = self.heading_class_head(net).transpose(2, 1)  # (batch_size, num_proposal, num_heading_bin)
        heading_residuals_normalized = self.heading_residual_head(net).transpose(2, 1)
        size_scores = self.size_class_head(net).transpose(2, 1)
        size_residuals_normalized = self.size_residual_head(net).transpose(2, 1)
        sem_cls_scores = self.sem_cls_scores_head(net).transpose(2, 1)
        end_points,pred_size = decode_scores(base_xyz,objectness_scores, center, heading_scores, heading_residuals_normalized, size_scores, size_residuals_normalized,
        sem_cls_scores, end_points, self.num_class, self.num_heading_bin, self.num_size_cluster,  self.mean_size_arr, prefix)
        return center,pred_size,end_points


class QuadPredictHead(nn.Module):
    def __init__(self,hidden_dim): 
        super().__init__()
        self.quad_scores_head = torch.nn.Conv1d(hidden_dim, 2, 1)
        self.center_head = torch.nn.Conv1d(hidden_dim, 3, 1)
        self.normal_vector_head = torch.nn.Conv1d(hidden_dim, 3, 1)
        self.size_head = torch.nn.Conv1d(hidden_dim, 2, 1)
        #self.direction_head = torch.nn.Conv1d(hidden_dim, 1, 1)
        self.conv1 = torch.nn.Conv1d(hidden_dim,hidden_dim,1)
        self.conv2 = torch.nn.Conv1d(hidden_dim,hidden_dim,1)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
    def forward(self,net,base_xyz,end_points,prefix):
        net = F.relu(self.bn1(self.conv1(net)))
        net = F.relu(self.bn2(self.conv2(net)))
        quad_scores = self.quad_scores_head(net).transpose(2, 1)  # (batch_size, num_proposal, 2)
        center = self.center_head(net).transpose(2, 1) + base_xyz # (batch_size, num_proposal, 3)
        normal_vector = self.normal_vector_head(net).transpose(2, 1) 
        normal_vector_norm = torch.norm(normal_vector, p=2)
        normal_vector = normal_vector.div(normal_vector_norm)
        size = self.size_head(net).transpose(2, 1) 
        #direction = self.direction_head(net).transpose(2, 1) 
        end_points[f'{prefix}quad_scores'] = quad_scores    
        end_points[f'{prefix}quad_center'] = center # (batch_size, num_proposal, 3)
        end_points[f'{prefix}normal_vector'] = normal_vector
        end_points[f'{prefix}quad_size'] = size
        #end_points[f'{prefix}quad_direction'] = direction
        return center, size, end_points

class PQ_Transformer(nn.Module):
    def __init__(self, input_feature_dim,num_class,num_proposal,num_quad_proposal,num_heading_bin,num_size_cluster,mean_size_arr,sampling='vote',num_layer=6,aux_loss=False,decoder_num = 1, args=None):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_proposal: number of object queries, ie detection slot. This is the maximal number of objects
        """
        super().__init__()
        self.i = 0
        
        
        self.input_feature_dim = input_feature_dim
        self.num_proposal = num_proposal
        self.num_class = num_class
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        self.num_heading_bin = num_heading_bin
        self.aux_loss = aux_loss
        self.sampling = sampling
        self.num_quad_proposal = num_quad_proposal
        self.num_layer = num_layer
        self.decoder_num = decoder_num
        #backbone
        self.backbone = Pointnet2Backbone(input_feature_dim=input_feature_dim)
        #transformer
        hidden_dim = 288 #self.transformer.d_model
        self.decoder_key_proj = nn.Conv1d(288, hidden_dim, kernel_size=1)
        self.decoder_query_proj = nn.Conv1d(288, hidden_dim, kernel_size=1)
        self.quad_decoder_query_proj = nn.Conv1d(288, hidden_dim, kernel_size=1)
        
        self.fps_module = FPSModule(self.num_quad_proposal)
        
        if self.sampling == 'vote':   
            self.vote = VotingModule(1, 288)
            self.vote_aggregation = PointnetSAModuleVotes( 
                npoint=self.num_proposal,
                radius=0.3,
                nsample=16,
                mlp=[288, 288, 288, 288],
                use_xyz=True,
                normalize_xyz=True
            )
        else:
            raise NotImplementedError

        
        #---proposal-----#
        self.quad_proposal = QuadPredictHead(hidden_dim)

        self.proposal = PredictHead(hidden_dim,num_heading_bin,num_size_cluster,num_class,mean_size_arr)

        self.prediction_heads = nn.ModuleList()
        self.prediction_quad_heads = nn.ModuleList()

        self.decoder = nn.ModuleList()
        self.decoder_self_posembeds = nn.ModuleList()
        self.decoder_cross_posembeds = nn.ModuleList()
        
        for i in range(0,6):
            self.prediction_heads.append(PredictHead(hidden_dim,num_heading_bin,num_size_cluster,num_class,mean_size_arr))
            self.prediction_quad_heads.append(QuadPredictHead(hidden_dim))
            self.decoder_cross_posembeds.append(PositionEmbeddingLearned(3, 288))
            self.decoder_self_posembeds.append(PositionEmbeddingLearned(3, 288))
            self.decoder.append(TransformerDecoderLayer(
                self_posembed=self.decoder_self_posembeds[i],cross_posembed=self.decoder_cross_posembeds[i])
                )

        self.init_weights()
        self.init_bn_momentum()
        nn.SyncBatchNorm.convert_sync_batchnorm(self)

    def forward(self, inputs):

        end_points = {}

        # Backbone
        end_points = self.backbone(inputs['point_clouds'], end_points)
        points_xyz = end_points['fp2_xyz']
        points_features = end_points['fp2_features']
        seed_xyz = end_points['fp2_xyz']
        seed_features = end_points['fp2_features']

        # Layout estimation: FPS
        xyz, features, sample_inds = self.fps_module(seed_xyz, seed_features)
        quad_cluster_feature = features
        quad_cluster_xyz = xyz
        end_points['aggregated_sample_xyz'] = xyz
        
        if self.sampling == 'vote':
            # Object detection: voting
            xyz, features = self.vote(seed_xyz, seed_features)
            features_norm = torch.norm(features, p=2, dim=1)
            features = features.div(features_norm.unsqueeze(1))
            end_points['vote_xyz'] = xyz
            end_points['vote_features'] = features
            xyz, features, fps_inds = self.vote_aggregation(xyz, features)
            cluster_feature = features
            cluster_xyz = xyz
            end_points['aggregated_vote_xyz'] = xyz
            end_points['cluster_feature'] = cluster_feature 
        else:
            raise NotImplementedError

       
        # Proposal
        proposal_center,proposal_size,end_points = self.proposal(cluster_feature,base_xyz=cluster_xyz,end_points=end_points,prefix='proposal_')  # N num_proposal 3
        
        proposal_center_,proposal_size_,end_points = self.quad_proposal(quad_cluster_feature,base_xyz=quad_cluster_xyz,end_points=end_points,prefix='proposal_')  # N num_proposal 3

        base_xyz = proposal_center.detach().clone()
        base_xyz_ = proposal_center_.detach().clone()
        bs = xyz.shape[0]

        # Transformer Decoder and Prediction
        query = self.decoder_query_proj(cluster_feature)        
        query_ = self.quad_decoder_query_proj(quad_cluster_feature)

        query_joint = torch.cat([query, query_], -1)

        # Position Embedding for Cross-Attention
        key = self.decoder_key_proj(points_features)
        key_pos = points_xyz

        for i in range(self.num_layer):
            prefix = 'last_' if (i == self.num_layer-1) else f'{i}head_'
            #pos embeding
            query_pos = base_xyz
            query_pos_ = base_xyz_

            query_pos_joint =  torch.cat([query_pos, query_pos_], 1)
            query_joint = self.decoder[i](query_joint, key, query_pos_joint, key_pos)
            query = query_joint[:,:,0:self.num_proposal]
            query_ = query_joint[:,:,self.num_proposal:]

            # Prediction
            base_xyz, base_size, end_points = self.prediction_heads[i](query,base_xyz=cluster_xyz,end_points=end_points,prefix=prefix)
            base_xyz_, base_size_, end_points = self.prediction_quad_heads[i](query_,base_xyz=quad_cluster_xyz,end_points=end_points,prefix=prefix)
    
            base_xyz = base_xyz.detach().clone()  # I don't understand why here detached?????
            base_xyz_ = base_xyz_.detach().clone()


        return end_points

    def init_weights(self):
        # initialize transformer
        for m in self.decoder.parameters():
            if m.dim() > 1:
                nn.init.xavier_uniform_(m)

    def init_bn_momentum(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = 0.1

