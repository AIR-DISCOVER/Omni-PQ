import torch

from models.loss_helper_pq import get_2d_box, projection2d

def get_arkit_pc_loss(end_points, batch_data_unlabeled, config):
    pc_loss, collisions = 0.0, 0
    #start_time = time.time()
    prefix = 'last_' #+ [f'{i}head_' for i in range(5)]

    gt_centers = batch_data_unlabeled['center_label']
    gt_sizes = batch_data_unlabeled['size_label']
    box_nums = batch_data_unlabeled['num_gt_boxes'][..., 0]
    
    batch_size = gt_centers.shape[0]
    pred_quad_centers = end_points[f'{prefix}quad_center'][batch_size:]
    pred_normal_vectors = end_points[f'{prefix}normal_vector'][batch_size:]
    pred_quad_sizes = end_points[f'{prefix}quad_size'][batch_size:]
    quad_scores = torch.softmax(end_points[f'{prefix}quad_scores'], dim=-1)[..., 1][batch_size:]
    quad_num = pred_quad_centers.shape[1]
    
    
    for b in range(batch_size):
        num_box = box_nums[b]
        gt_center = gt_centers[b, :box_nums[b], ...]
        box_size = gt_sizes[b, :box_nums[b], ...]
        pred_quad_center = pred_quad_centers[b]
        pred_normal_vector = pred_normal_vectors[b]
        pred_quad_size = pred_quad_sizes[b]
        
        
        pred_corners_2d = get_2d_box(box_size[None, ...], gt_center[None, ...])[0, ...]
        corner_point = pred_corners_2d.reshape(-1, 2)
    
        # Make pred quad point inwards the scene
        pseudo_scene_center = torch.tensor([0., 0., 1.]).cuda()
        offset = pseudo_scene_center - pred_quad_center.detach()
        offset[..., 2] = 0.
        reverse_mask = torch.bmm(offset.reshape(quad_num, 1, 3), pred_normal_vector.reshape(quad_num, 3, 1))[..., 0, 0] < 0
        reverse_mask = reverse_mask[..., None]
        
        pred_normal_vector_inwards = pred_normal_vector * reverse_mask * -1 + pred_normal_vector * (~reverse_mask)


        for k in range(quad_num):
            if quad_scores[b, k] > 0.1:
                loss, collision = \
                    projection2d(corner_point, pred_quad_center[k], pred_normal_vector_inwards[k], pred_quad_size[k])

                pc_loss = pc_loss + loss / num_box
                collisions = collisions + collision

    return pc_loss, collisions