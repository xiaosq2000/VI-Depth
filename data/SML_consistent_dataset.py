import numpy as np
import torch.utils.data
import modules.midas.utils as utils
from PIL import Image
from path import Path

def load_input_image(input_image_fp):
    return utils.read_image(input_image_fp)

def load_sparse_depth(input_sparse_depth_fp, depth_scale):
    input_sparse_depth = np.array(Image.open(input_sparse_depth_fp), dtype=np.float32) / depth_scale
    input_sparse_depth[input_sparse_depth <= 0] = 0.0
    return input_sparse_depth

def load_depth_image_from_npy(load_path):
    # Load the depth image from the .npy file
    depth_array = np.load(load_path)
    
    return depth_array

def generate_sample_index(num_frames, skip_frames, sequence_length):
    sample_index_list = []
    k = skip_frames
    demi_length = (sequence_length-1)//2
    shifts = list(range(-demi_length * k,
                        demi_length * k + 1, k))
    shifts.pop(demi_length)

    if num_frames > sequence_length:
        for i in range(demi_length * k, num_frames-demi_length * k):
            sample_index = {'tgt_idx': i, 'ref_idx': []}
            for j in shifts:
                sample_index['ref_idx'].append(i+j)
            sample_index_list.append(sample_index)

    return sample_index_list

class SML_consistent_dataset(torch.utils.data.Dataset):
    def __init__(self,
                 root,
                 depth_scale=256.0,
                 sequence_length=3,
                ):
        self.root = Path(root)/'training'
        scene_list_path = self.root/'train.txt'
        self.scenes = [self.root/folder[:-1]
                       for folder in open(scene_list_path)]
        
        self.crawl_folders(sequence_length)

    def crawl_folders(self, sequence_length):
        # k skip frames
        sequence_set = []

        for scene in self.scenes:
            intrinsics = np.genfromtxt(
                scene/'K.txt').astype(np.float32).reshape((3, 3))

            images_path = scene + '/image'
            imgs = sorted(images_path.files('*.png'))

            frame_index = [int(index) for index in open(scene/'frame_index.txt')]
            imgs = [imgs[d] for d in frame_index]

            ga_depth_inv = sorted((scene/'ga_depth_inv').files('*.npy'))
            ga_depth_inv = [ga_depth_inv[d] for d in frame_index]

            # interpolated scaffolding
            interp_depth = sorted((scene/'interp_scale').files('*.npy'))
            interp_depth = [interp_depth[d] for d in frame_index]

            #load pose
            poses = sorted((scene/'absolute_pose').files('*.txt'))
            poses = [poses[d] for d in frame_index]
            
            #get gt depths
            gt_depth = sorted((scene/'ground_truth').files('*.png'))
            gt_depth = [gt_depth[d] for d in frame_index]

            if len(imgs) < sequence_length:
                continue

            sample_index_list = generate_sample_index(
                len(imgs), 1, sequence_length)
            
            for sample_index in sample_index_list:
                sample = {'intrinsics': intrinsics,
                          'tgt_img': imgs[sample_index['tgt_idx']]}
                sample['tgt_ga_depth'] = ga_depth_inv[sample_index['tgt_idx']]
                sample['tgt_gt_depth'] = gt_depth[sample_index['tgt_idx']]
                sample['tgt_pose'] = poses[sample_index['tgt_idx']]
                sample['tgt_interp'] = interp_depth[sample_index['tgt_idx']]
                sample['tgt_pose'] = poses[sample_index['tgt_idx']]


                sample['ref_imgs'] = []; sample['ref_ga_depth'] = []
                sample['ref_gt_depth'] = []; sample['ref_pose'] = []
                sample['ref_interp'] = []; sample['ref_pose'] = []
                for j in sample_index['ref_idx']:
                    sample['ref_imgs'].append(imgs[j])
                    sample['ref_ga_depth'].append(ga_depth_inv[j])
                    sample['ref_interp'].append(interp_depth[j])
                    sample['ref_pose'].append(poses[j])
                    sample['ref_gt_depth'].append(gt_depth[j])
                sequence_set.append(sample)

        self.samples = sequence_set

    def __getitem__(self, index):
        sample = self.samples[index]
        tgt_img = load_input_image(sample['tgt_img'])
        tgt_gt_depth = load_sparse_depth(sample['tgt_gt_depth'], depth_scale=256.0)
        tgt_ga_depth = load_depth_image_from_npy(sample['tgt_ga_depth'])
        tgt_interp = load_depth_image_from_npy(sample['tgt_interp'])
        tgt_pose = np.loadtxt(sample['tgt_pose'])

        ref_img = [load_input_image(ref_img) for ref_img in sample['ref_imgs']]
        ref_ga_depth = [load_depth_image_from_npy(ref_ga_depth) for ref_ga_depth in sample['ref_ga_depth']]
        ref_interp = [load_depth_image_from_npy(ref_interp) for ref_interp in sample['ref_interp']]
        ref_gt_depth = [load_sparse_depth(ref_gt_depth, depth_scale=256.0) for ref_gt_depth in sample['ref_gt_depth']]
        ref_pose = [np.loadtxt(pose) for pose in sample['ref_pose']]
        intrinsics = np.copy(sample['intrinsics'])
        
        return tgt_img, tgt_gt_depth, tgt_ga_depth, tgt_interp, ref_img, \
            ref_ga_depth, ref_interp, ref_gt_depth, tgt_pose, ref_pose, intrinsics
    
    def __len__(self):
        return len(self.samples)