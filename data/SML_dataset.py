import numpy as np
import torch.utils.data
import modules.midas.utils as utils
from PIL import Image

def load_input_image(input_image_fp):
    return utils.read_image(input_image_fp)

def load_sparse_depth(input_sparse_depth_fp, depth_scale):
    input_sparse_depth = np.array(Image.open(input_sparse_depth_fp), dtype=np.float32) / depth_scale
    input_sparse_depth[input_sparse_depth <= 0] = 0.0
    return input_sparse_depth

class SML_dataset(torch.utils.data.Dataset):
    def __init__(self,
                 image_paths,
                 gt_depth_paths,
                 sparse_paths,
                 depth_scale=256.0,
                ):
        self.n_samples = len(image_paths)
        
        for paths in [image_paths, gt_depth_paths, sparse_paths]:
            assert len(paths) == self.n_samples
        
        self.image_paths = image_paths
        self.gt_depth_paths = gt_depth_paths
        self.sparse_paths = sparse_paths
        
    def __getitem__(self, index):
        image = load_input_image(self.image_paths[index])
        gt_depth = load_sparse_depth(self.gt_depth_paths[index], depth_scale=1000.0)
        sparse_depth = load_sparse_depth(self.sparse_paths[index], depth_scale=1000.0)
        
        image, gt_depth, sparse_depth = [
            T.astype(np.float32) for T in [image, gt_depth, sparse_depth]
        ]
        
        return image, gt_depth, sparse_depth
    
    def __len__(self):
        return self.n_samples
        
                 