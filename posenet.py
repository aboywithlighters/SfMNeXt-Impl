import argparse
import copy
import os
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms as transforms

import datasets
from configuration import ConfigLoader
import networks
from utils import *
from layers import *


config_loader = ConfigLoader()

def mkdir_if_not_exists(path):
    """Make a directory if it does not exist.
    
    Args:
        path (str): directory to create
    """
    if not os.path.exists(path):
        os.makedirs(path)
        
def read_cfgs():
    """Parse arguments and laod configurations

    Returns
    -------
    args : args
        arguments
    cfg : edict
        configuration dictionary
    """
    ''' Argument Parsing '''
    parser = argparse.ArgumentParser(description='posenet system')
    parser.add_argument("-s", "--seq", 
                        default=None, help="sequence")
    parser.add_argument("-d", "--default_configuration", type=str, 
                        default="/poolz2/fanliang/SfMNeXt-Impl/args_files/fanliang/ablation_posenet.yml",
                        help="default configuration files")
    parser.add_argument("-c", "--configuration", type=str,
                        default=None,
                        help="custom configuration file")
    parser.add_argument("--no_confirm", action="store_true",
                        help="no confirmation questions")
    args = parser.parse_args()

    ''' Read configuration '''
    # read default and custom config, merge cfgs
    config_files = [args.default_configuration, args.configuration]
    cfg = config_loader.merge_cfg(config_files)
    if args.seq is not None:
        if cfg.dataset == "kitti_odom":
            cfg.seq = "{:02}".format(int(args.seq))
        else:
            cfg.seq = args.seq
    cfg.seq = str(cfg.seq)

    ''' double check result directory '''
    if args.no_confirm:
        mkdir_if_not_exists(cfg.directory.result_dir)
        cfg.no_confirm = True
    else:
        cfg.no_confirm = False
        continue_flag = input("Save result in {}? [y/n]".format(cfg.directory.result_dir))
        if continue_flag == "y":
            mkdir_if_not_exists(cfg.directory.result_dir)
        else:
            exit()
    return args, cfg

def predict_poses(inputs, features):
    """Predict poses between input frames for monocular sequences.
    """
    outputs = {}

    # In this setting, we compute the pose to each source frame via a
    # separate forward pass through the pose network.

    # select what features the pose network takes as input
    pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in [0,-1,1]}

    for f_i in [-1,1]:
        if f_i != "s":
            # To maintain ordering we always pass frames in temporal order
            if f_i < 0:
                pose_inputs = [pose_feats[f_i], pose_feats[0]]
            else:
                pose_inputs = [pose_feats[0], pose_feats[f_i]]

            pose_inputs = torch.cat(pose_inputs, 1)

            axisangle, translation = models["pose"](pose_inputs)
            # print(axisangle.shape)
            # axisangle:[12, 1, 1, 3]  translation:[12, 1, 1, 3]
            outputs[("axisangle", 0, f_i)] = axisangle
            outputs[("translation", 0, f_i)] = translation

            # Invert the matrix if the frame id is negative
            outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                axisangle[:, 0], translation[:, 0], invert=(f_i < 0))
            # outputs[("cam_T_cam", 0, f_i)]: [12, 4, 4]

    return outputs

class SE3():
    """SE3 object consists rotation and translation components
    """
    def __init__(self, np_arr=None):
        if np_arr is None:
            self._pose = np.eye(4)
        else:
            self._pose = np_arr

    @property
    def pose(self):
        """ (array, [4x4]): camera pose 
        """
        return self._pose

    @pose.setter
    def pose(self, value):
        self._pose = value

    @property
    def inv_pose(self):
        """ (array, [4x4]): inverse camera pose 
        """
        return np.linalg.inv(self._pose)

    @inv_pose.setter
    def inv_pose(self, value):
        self._pose = np.linalg.inv(value)

    @property
    def R(self):
        """ (array, [3x4]): rotation matrix
        """
        return self._pose[:3, :3]

    @R.setter
    def R(self, value):
        self._pose[:3, :3] = value

    @property
    def t(self):
        """ (array, [3x1]): translation vector
        """
        return self._pose[:3, 3:]

    @t.setter
    def t(self, value):
        self._pose[:3, 3:] = value

def update_global_pose(cur_pose, new_pose, scale=1.):
        """update estimated poses w.r.t global coordinate system

        Args:
            new_pose (SE3): new pose
            scale (float): scaling factor
        """
        cur_pose.t = cur_pose.R @ new_pose.t * scale \
                            + cur_pose.t
        cur_pose.R = cur_pose.R @ new_pose.R
        global_poses = copy.deepcopy(cur_pose)
        return global_poses


if __name__ == '__main__':
    # Read config
    args, cfg = read_cfgs()

    # Set random seed
    SEED = cfg.seed
    np.random.seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.manual_seed(SEED)
    models = {}
    models["pose"] = networks.PoseCNN(2).cuda()
    models["encoder"] = networks.Unet(pretrained=True, backbone=cfg.backbone, in_channels=3, 
                                      num_classes=cfg.model_dim, decoder_channels=cfg.dec_channels).cuda()

    # load the weight of the posenet
    checkpoint_pose = torch.load(cfg.PoseNet.model_pth)
    checkpoint_encoder = torch.load(cfg.Encoder.model_pth)
    encoder_path = cfg.Encoder.model_pth
    models["pose"].load_state_dict(checkpoint_pose)
    #import pdb;pdb.set_trace()
    loaded_dict_enc = torch.load(encoder_path, map_location="cuda")
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in models["encoder"].state_dict()}
    models["encoder"].load_state_dict(filtered_dict_enc)

    # set the eval model
    models["pose"].eval()
    models["encoder"].eval()

    # load the dataset
    # data
    datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                        "kitti_odom": datasets.KITTIOdomDataset,
                        "cityscapes_preprocessed": datasets.CityscapesPreprocessedDataset}
    dataset = datasets_dict[cfg.dataset] # default="kitti"

    fpath = os.path.join(os.path.dirname(__file__), "splits", cfg.split, "test_files_10.txt")

    #train_filenames = readlines(fpath.format("train"))
    val_filenames = readlines(fpath)
    train_dataset = dataset(
        cfg.directory.img_seq_dir, val_filenames, cfg.image.height, cfg.image.width,
        [0, -1, 1], 1, is_train=False, img_ext=cfg.image.ext) # num_scales = 1
    train_loader = DataLoader(
        train_dataset, cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    
    #import pdb;pdb.set_trace()
    #ref_pose = SE3()
    cur_pose = SE3(np.eye(4))
    
    for batch_idx, inputs in enumerate(tqdm(train_loader)):
        # #visualize to check the data in the dataset
        # if batch_idx == 14:
        #     to_pil = transforms.ToPILImage()
        #     pil_image = to_pil(inputs[('color', 1, 0)][0,:,:,:])
        #     pil_image.show()
        #     import pdb;pdb.set_trace()

        for key, ipt in inputs.items():
            inputs[key] = ipt.cuda()

        # Otherwise, we only feed the image with frame_id 0 through the depth encoder
        features = models["encoder"](inputs["color_aug", 0, 0])

        outputs = predict_poses(inputs, features)
        new_pose = outputs[('cam_T_cam', 0, -1)][0,:,:,]
        #import pdb;pdb.set_trace()
        new_pose = new_pose.cpu().detach().numpy()
        new_pose = SE3(new_pose)
        cur_pose = update_global_pose(cur_pose, new_pose, scale=1.)

        # Save trajectory txt
        traj_txt = "{}{}.txt".format(cfg.directory.result_dir, cfg.seq)
        flattened_array = cur_pose._pose[:3].flatten()
        row = str(batch_idx+1) + " " + ' '.join(map(str, flattened_array))
        f = open(traj_txt,'+a')
        f.write(row[:-8] + '\n')
        
