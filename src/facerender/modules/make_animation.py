from scipy.spatial import ConvexHull
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm 

def normalize_kp(kp_source, kp_driving, kp_driving_initial, adapt_movement_scale=False,
                 use_relative_movement=False, use_relative_jacobian=False):
    if adapt_movement_scale:
        source_area = ConvexHull(kp_source['value'][0].data.cpu().numpy()).volume
        driving_area = ConvexHull(kp_driving_initial['value'][0].data.cpu().numpy()).volume
        adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)
    else:
        adapt_movement_scale = 1

    kp_new = {k: v for k, v in kp_driving.items()}

    if use_relative_movement:
        kp_value_diff = (kp_driving['value'] - kp_driving_initial['value'])
        kp_value_diff *= adapt_movement_scale
        kp_new['value'] = kp_value_diff + kp_source['value']

        if use_relative_jacobian:
            jacobian_diff = torch.matmul(kp_driving['jacobian'], torch.inverse(kp_driving_initial['jacobian']))
            kp_new['jacobian'] = torch.matmul(jacobian_diff, kp_source['jacobian'])

    return kp_new

def headpose_pred_to_degree(pred):
    device = pred.device
    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).type_as(pred).to(device)
    pred = F.softmax(pred)
    degree = torch.sum(pred*idx_tensor, 1) * 3 - 99
    return degree

def get_rotation_matrix(yaw, pitch, roll):
    yaw = yaw / 180 * 3.14
    pitch = pitch / 180 * 3.14
    roll = roll / 180 * 3.14

    roll = roll.unsqueeze(1)
    pitch = pitch.unsqueeze(1)
    yaw = yaw.unsqueeze(1)

    pitch_mat = torch.cat([torch.ones_like(pitch), torch.zeros_like(pitch), torch.zeros_like(pitch), 
                          torch.zeros_like(pitch), torch.cos(pitch), -torch.sin(pitch),
                          torch.zeros_like(pitch), torch.sin(pitch), torch.cos(pitch)], dim=1)
    pitch_mat = pitch_mat.view(pitch_mat.shape[0], 3, 3)

    yaw_mat = torch.cat([torch.cos(yaw), torch.zeros_like(yaw), torch.sin(yaw), 
                           torch.zeros_like(yaw), torch.ones_like(yaw), torch.zeros_like(yaw),
                           -torch.sin(yaw), torch.zeros_like(yaw), torch.cos(yaw)], dim=1)
    yaw_mat = yaw_mat.view(yaw_mat.shape[0], 3, 3)

    roll_mat = torch.cat([torch.cos(roll), -torch.sin(roll), torch.zeros_like(roll),  
                         torch.sin(roll), torch.cos(roll), torch.zeros_like(roll),
                         torch.zeros_like(roll), torch.zeros_like(roll), torch.ones_like(roll)], dim=1)
    roll_mat = roll_mat.view(roll_mat.shape[0], 3, 3)

    rot_mat = torch.einsum('bij,bjk,bkm->bim', pitch_mat, yaw_mat, roll_mat)

    return rot_mat

def keypoint_transformation(kp_canonical, he, wo_exp=False):
    kp = kp_canonical['value']    # (bs, k, 3) 
    yaw, pitch, roll= he['yaw'], he['pitch'], he['roll']      
    yaw = headpose_pred_to_degree(yaw) 
    pitch = headpose_pred_to_degree(pitch)
    roll = headpose_pred_to_degree(roll)

    if 'yaw_in' in he:
        yaw = he['yaw_in']
    if 'pitch_in' in he:
        pitch = he['pitch_in']
    if 'roll_in' in he:
        roll = he['roll_in']

    rot_mat = get_rotation_matrix(yaw, pitch, roll)    # (bs, 3, 3)

    t, exp = he['t'], he['exp']
    if wo_exp:
        exp =  exp*0  
    
    # keypoint rotation
    kp_rotated = torch.einsum('bmp,bkp->bkm', rot_mat, kp)

    # keypoint translation
    t[:, 0] = t[:, 0]*0
    t[:, 2] = t[:, 2]*0
    t = t.unsqueeze(1).repeat(1, kp.shape[1], 1)
    kp_t = kp_rotated + t

    # add expression deviation 
    exp = exp.view(exp.shape[0], -1, 3)
    kp_transformed = kp_t + exp

    return {'value': kp_transformed}



def make_animation(source_image, source_semantics, target_semantics,
                            generator, kp_detector, he_estimator, mapping, 
                            yaw_c_seq=None, pitch_c_seq=None, roll_c_seq=None,
                            use_exp=True, use_half=False, rank=0, p_num=1):
    print(f"rank, p_num: {rank}, {p_num}")
    # ++++ quantize
    import os
    if not os.path.exists("generator_int8"):
        # do the quantization!
        print(target_semantics.shape[1])
        def calib_func(model):
            kp_canonical = kp_detector(source_image)
            he_source = mapping(source_semantics)
            kp_source = keypoint_transformation(kp_canonical, he_source)
            for frame_idx in tqdm(range(8), 'Face Renderer:'):  # calib 8 iterations
                target_semantics_frame = target_semantics[:, frame_idx]
                he_driving = mapping(target_semantics_frame)
                if yaw_c_seq is not None:
                    he_driving['yaw_in'] = yaw_c_seq[:, frame_idx]
                if pitch_c_seq is not None:
                    he_driving['pitch_in'] = pitch_c_seq[:, frame_idx]
                if roll_c_seq is not None:
                    he_driving['roll_in'] = roll_c_seq[:, frame_idx]

                kp_driving = keypoint_transformation(kp_canonical, he_driving)

                kp_norm = kp_driving
                out = generator(source_image, kp_source=kp_source, kp_driving=kp_norm)

        from neural_compressor import PostTrainingQuantConfig, quantization
        conf = PostTrainingQuantConfig()
        model = quantization.fit(generator, conf, calib_func=calib_func)
        generator = model
    else:
        from neural_compressor.utils.pytorch import load
        generator = load("generator_int8", generator)
    # ++++
    with torch.no_grad():
        predictions = []
        import time
        import os
        #with torch.cpu.amp.autocast():
        kp_canonical = kp_detector(source_image)
        he_source = mapping(source_semantics)
        kp_source = keypoint_transformation(kp_canonical, he_source)
        for frame_idx in tqdm(range(target_semantics.shape[1]), 'Face Renderer:'):
            if frame_idx % p_num != rank:
                continue
            #if frame_idx >24:
            #    continue
            target_semantics_frame = target_semantics[:, frame_idx]
            he_driving = mapping(target_semantics_frame)
            if yaw_c_seq is not None:
                he_driving['yaw_in'] = yaw_c_seq[:, frame_idx]
            if pitch_c_seq is not None:
                he_driving['pitch_in'] = pitch_c_seq[:, frame_idx]
            if roll_c_seq is not None:
                he_driving['roll_in'] = roll_c_seq[:, frame_idx]
            kp_driving = keypoint_transformation(kp_canonical, he_driving)
            kp_norm = kp_driving
            out = generator(source_image, kp_source=kp_source, kp_driving=kp_norm)
            #print(f"{rank}:{frame_idx}")
            predictions.append(out['prediction'])
        folder_name = "logs"
        file_name = f'{p_num}_{rank}.npz'
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        file_path = os.path.join(folder_name, file_name)
        f = open(file_path, "w")
        np.savez(file_path, *predictions)
        # master process will be pending here to collect all the predictions
        # ... pending ...
        import re
        def atoi(text):
            return int(text) if text.isdigit() else text
        def natural_keys(text):
            return [ atoi(c) for c in re.split(r'(\d+)', text) ]
        if rank == 0:
            while len(os.listdir(folder_name)) < p_num:
                time.sleep(0.5)
            # load all the npz arrays, merge by sequence
            #npz_file_paths = sorted(os.listdir(folder_name))
            npz_file_paths=os.listdir(folder_name)
            npz_file_paths.sort(key=natural_keys)
            print("start to merge...")
            print(f"npz_file_paths: {npz_file_paths}")
        else:
            exit(0)
        aggregated_lst = []
        for npz_file_path in npz_file_paths:
            npz_file = np.load(os.path.join(folder_name, npz_file_path))
            aggregated_lst.append([npz_file[i] for i in npz_file.files])
        #aggregated_predictions = [torch.from_numpy(x) for y in zip(*aggregated_lst) for x in y]
        # agg lst elements may have different length!
        #exit(0)
        import itertools
        padded_preds = [x for y in itertools.zip_longest(*aggregated_lst) for x in y]
        print("padded preds length:")
        print(len(padded_preds))
        aggregated_predictions = [torch.from_numpy(i) for i in padded_preds if i is not None]
        #predictions_ts = torch.stack(predictions, dim=1)

        predictions_ts = torch.stack(aggregated_predictions, dim=1)
    return predictions_ts

class AnimateModel(torch.nn.Module):
    """
    Merge all generator related updates into single model for better multi-gpu usage
    """

    def __init__(self, generator, kp_extractor, mapping):
        super(AnimateModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.mapping = mapping

        self.kp_extractor.eval()
        self.generator.eval()
        self.mapping.eval()

    def forward(self, x):
        
        source_image = x['source_image']
        source_semantics = x['source_semantics']
        target_semantics = x['target_semantics']
        yaw_c_seq = x['yaw_c_seq']
        pitch_c_seq = x['pitch_c_seq']
        roll_c_seq = x['roll_c_seq']

        predictions_video = make_animation(source_image, source_semantics, target_semantics,
                                        self.generator, self.kp_extractor,
                                        self.mapping, use_exp = True,
                                        yaw_c_seq=yaw_c_seq, pitch_c_seq=pitch_c_seq, roll_c_seq=roll_c_seq)
        
        return predictions_video
