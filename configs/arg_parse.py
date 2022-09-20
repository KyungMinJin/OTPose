import os.path as osp
import argparse


def default_parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument('--cfg', help='experiment configure file name', type=str,
                        default='./configs/17/model_RSN.yaml')
    parser.add_argument('--PE_Name', type=str, default='OTPose')
    parser.add_argument('--train', action='store_true', default=True)
    parser.add_argument('--val', action='store_true', default=False)
    parser.add_argument('--val_from_checkpoint',
                        help='exec val from the checkpoint_id. if config.yaml specifies a model file, this parameter '
                             'will invalid',
                        type=int,
                        default='-1')
    parser.add_argument('--sigma_schedule', type=int, nargs='+',
                        default=[], help='Decrease learning rate at these epochs.')
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--root_dir', type=str, default='./')
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument(
        "--config-file",
        default="./configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    args.rootDir = osp.abspath(args.root_dir)
    args.cfg = osp.join(args.rootDir, osp.abspath(args.cfg))
    args.PE_Name = args.PE_Name.upper()
    return args
