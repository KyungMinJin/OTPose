import json
import logging

import numpy as np

from utils.registry import MODEL_REGISTRY
import torch.backends.cudnn as cudnn
import os
from yacs.config import CfgNode as _CfgNode
import os.path as osp
import re
import scipy.io as sio

BASE_KEY = '_BASE_'
LOGGER = logging.getLogger(__name__)
POSETRACK18_LM_NAMES_COCO_ORDER = [
    "nose",
    "head_bottom",  # "left_eye",
    "head_top",  # "right_eye",
    "left_ear",  # will be left zeroed out
    "right_ear",  # will be left zeroed out
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]
POSETRACK18_LM_NAMES = [  # This is used to identify the IDs.
    "right_ankle",
    "right_knee",
    "right_hip",
    "left_hip",
    "left_knee",
    "left_ankle",
    "right_wrist",
    "right_elbow",
    "right_shoulder",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "head_bottom",
    "nose",
    "head_top",
]


class CfgNode(_CfgNode):

    def merge_from_file(self, cfg_filename):
        with open(cfg_filename, "r") as f:
            cfg = self.load_cfg(f)
        if BASE_KEY in cfg:
            base_cfg_file = cfg[BASE_KEY]
            if base_cfg_file.startswith("~"):
                base_cfg_file = osp.expanduser(base_cfg_file)
            else:
                base_cfg_file = osp.join(osp.dirname(cfg_filename), base_cfg_file)
            with open(base_cfg_file, "r") as base_f:
                base_cfg = self.load_cfg(base_f)
            self.merge_from_other_cfg(base_cfg)
            del cfg[BASE_KEY]
        self.merge_from_other_cfg(cfg)


def update_config(cfg: CfgNode, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    if args.rootDir:
        cfg.ROOT_DIR = args.rootDir

    cfg.OUTPUT_DIR = os.path.abspath(os.path.join(cfg.ROOT_DIR, cfg.OUTPUT_DIR))

    cfg.DATASET.JSON_DIR = os.path.abspath(os.path.join(cfg.ROOT_DIR, cfg.DATASET.JSON_DIR))
    cfg.DATASET.IMG_DIR = os.path.abspath(os.path.join(cfg.ROOT_DIR, cfg.DATASET.IMG_DIR))
    cfg.DATASET.TEST_IMG_DIR = os.path.abspath(os.path.join(cfg.ROOT_DIR, cfg.DATASET.TEST_IMG_DIR))

    cfg.MODEL.PRETRAINED = os.path.abspath(os.path.join(cfg.ROOT_DIR, cfg.MODEL.PRETRAINED))

    cfg.VAL.ANNOT_DIR = os.path.abspath(os.path.join(cfg.ROOT_DIR, cfg.VAL.ANNOT_DIR))
    cfg.VAL.COCO_BBOX_FILE = os.path.abspath(os.path.join(cfg.ROOT_DIR, cfg.VAL.COCO_BBOX_FILE))

    cfg.TEST.ANNOT_DIR = os.path.abspath(os.path.join(cfg.ROOT_DIR, cfg.TEST.ANNOT_DIR))
    cfg.TEST.COCO_BBOX_FILE = os.path.abspath(os.path.join(cfg.ROOT_DIR, cfg.TEST.COCO_BBOX_FILE))

    cfg.freeze()


def get_cfg(args) -> CfgNode:
    """
        Get a copy of the default config.
        Returns:
            a fastreid CfgNode instance.
    """

    from configs.default import _C

    return _C.clone()


def setup(args):
    cfg = get_cfg(args)
    update_config(cfg, args)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    cudnn.enabled = cfg.CUDNN.ENABLED

    return cfg


def get_dataset_name(cfg):
    dataset_name = cfg.DATASET.NAME
    if dataset_name == "PoseTrack":
        datset_version = "18" if cfg.DATASET.IS_2018 else "17"
        dataset_name = dataset_name + datset_version

    return dataset_name


def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def get_latest_checkpoint(checkpoint_save_folder):
    checkpoint_saves_paths = list_immediate_childfile_paths(checkpoint_save_folder, ext="pth")
    if len(checkpoint_saves_paths) == 0:
        return None

    checkpoint_saves_paths = [x for x in checkpoint_saves_paths if 'best' not in x]
    latest_checkpoint = checkpoint_saves_paths[0]

    # we define the format of checkpoint like "epoch_0_state.pth"
    latest_index = int(osp.basename(latest_checkpoint).split("_")[1])
    for checkpoint_save_path in checkpoint_saves_paths:
        checkpoint_save_file_name = osp.basename(checkpoint_save_path)
        now_index = int(checkpoint_save_file_name.split("_")[1])
        if now_index > latest_index:
            latest_checkpoint = checkpoint_save_path
            latest_index = now_index
    return latest_checkpoint


def get_best_checkpoint(checkpoint_save_folder):
    checkpoint_saves_paths = list_immediate_childfile_paths(checkpoint_save_folder, ext="pth")
    if len(checkpoint_saves_paths) == 0:
        return None

    checkpoint_saves_paths = [x for x in checkpoint_saves_paths if 'best' in x]
    latest_checkpoint = checkpoint_saves_paths[0]

    # we define the format of checkpoint like "epoch_0_state.pth"
    best_mAP = float(osp.basename(latest_checkpoint).split("_")[2])
    for checkpoint_save_path in checkpoint_saves_paths:
        checkpoint_save_file_name = osp.basename(checkpoint_save_path)
        now_mAP = float(checkpoint_save_file_name.split("_")[2])
        if now_mAP > best_mAP:
            latest_checkpoint = checkpoint_save_path
            best_mAP = now_mAP

    return latest_checkpoint


def list_immediate_childfile_paths(folder_path, ext=None, exclude=None):
    files_names = list_immediate_childfile_names(folder_path, ext, exclude)
    files_full_paths = [os.path.join(folder_path, file_name) for file_name in files_names]
    return files_full_paths


def list_immediate_childfile_names(folder_path, ext=None, exclude=None):
    files_names = [file_name for file_name in next(os.walk(folder_path))[2]]
    if ext is not None:
        if isinstance(ext, str):
            files_names = [file_name for file_name in files_names if file_name.endswith(ext)]
        elif isinstance(ext, list):
            temp_files_names = []
            for file_name in files_names:
                for ext_item in ext:
                    if file_name.endswith(ext_item):
                        temp_files_names.append(file_name)
            files_names = temp_files_names
    if exclude is not None:
        files_names = [file_name for file_name in files_names if not file_name.endswith(exclude)]
    natural_sort(files_names)
    return files_names


def natural_sort(given_list):
    """ Sort the given list in the way that humans expect."""
    given_list.sort(key=alphanum_key)


def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"] """
    return [tryint(c) for c in re.split('([0-9]+)', s)]


def tryint(s):
    try:
        return int(s)
    except:
        return s


def get_all_checkpoints(checkpoint_save_folder):
    checkpoint_saves_paths = list_immediate_childfile_paths(checkpoint_save_folder, ext="pth")
    if len(checkpoint_saves_paths) == 0:
        return None
    checkpoints_list = []
    # we define the format of checkpoint like "epoch_0_state.pth"
    for checkpoint_save_path in checkpoint_saves_paths:
        checkpoints_list.append(checkpoint_save_path)
    return checkpoints_list


def video2filenames(annot_dir):
    pathtodir = annot_dir

    output, L = {}, {}
    mat_files = [f for f in os.listdir(pathtodir) if
                 os.path.isfile(os.path.join(pathtodir, f)) and '.mat' in f]
    json_files = [f for f in os.listdir(pathtodir) if
                  os.path.isfile(os.path.join(pathtodir, f)) and '.json' in f]

    if len(json_files) > 1:
        files = json_files
        ext_types = '.json'
    else:
        files = mat_files
        ext_types = '.mat'

    for fname in files:
        if ext_types == '.mat':
            out_fname = fname.replace('.mat', '.json')
            data = sio.loadmat(
                os.path.join(pathtodir, fname), squeeze_me=True,
                struct_as_record=False)
            temp = data['annolist'][0].image.name

            data2 = sio.loadmat(os.path.join(pathtodir, fname))
            num_frames = len(data2['annolist'][0])
        elif ext_types == '.json':
            out_fname = fname
            with open(os.path.join(pathtodir, fname), 'r') as fin:
                data = json.load(fin)

            if 'annolist' in data:
                temp = data['annolist'][0]['image'][0]['name']
                num_frames = len(data['annolist'])
            else:
                temp = data['images'][0]['file_name']
                num_frames = data['images'][0]['nframes']


        else:
            raise NotImplementedError()
        video = os.path.dirname(temp)
        output[video] = out_fname
        L[video] = num_frames
    return output, L


def seqtype2idx(seqtype):
    if seqtype == "mpii":
        return 1
    elif seqtype == "bonn":
        return 2
    elif seqtype in ["mpiinew"]:
        return 3
    else:
        print("unknown sequence type:", seqtype)
        assert False


def posetrack18_fname2id(fname, frameidx):
    """Generates image id

    Args:
      fname: name of the PoseTrack sequence
      frameidx: index of the frame within the sequence
    """
    tok = os.path.basename(fname).split("_")
    seqidx = int(tok[0])
    seqtype_idx = seqtype2idx(tok[1])

    assert frameidx >= 0 and frameidx < 1e4
    image_id = seqtype_idx * 10000000000 + seqidx * 10000 + frameidx
    return image_id


class Person:

    """
    A PoseTrack annotated person.

    Parameters
    ==========

    track_id: int
      Unique integer representing a person track.
    """

    def __init__(self, track_id):
        self.track_id = track_id
        self.landmarks = None  # None or list of dicts with 'score', 'x', 'y', 'id'.
        self.rect_head = None  # None or dict with 'x1', 'x2', 'y1' and 'y2'.
        self.rect = None  # None or dict with 'x1', 'x2', 'y1' and 'y2'.
        self.score = None  # None or float.

    def to_new(self):
        """
        Return a dictionary representation for the PoseTrack18 format.

        The fields 'image_id' and 'id' must be added to the result.
        """
        keypoints = []
        scores = []
        write_scores = (
            len([1 for lm_info in self.landmarks if "score" in lm_info.keys()]) > 0
        )
        for landmark_name in POSETRACK18_LM_NAMES_COCO_ORDER:
            try:
                try:
                    lm_id = POSETRACK18_LM_NAMES.index(landmark_name)
                except ValueError:
                    lm_id = -1
                landmark_info = [lm for lm in self.landmarks if lm["id"] == lm_id][0]
            except IndexError:
                landmark_info = {"x": 0, "y": 0, "is_visible": 0}
            is_visible = 1
            if "is_visible" in landmark_info.keys():
                is_visible = landmark_info["is_visible"]
            keypoints.extend([landmark_info["x"], landmark_info["y"], is_visible])
            if "score" in landmark_info.keys():
                scores.append(landmark_info["score"])
            elif write_scores:
                LOGGER.warning("Landmark with missing score info detected. Using 0.")
                scores.append(0.)
        ret = {
            "track_id": self.track_id,
            "category_id": 1,
            "keypoints": keypoints,
            "scores": scores,
            # image_id and id added later.
        }
        if self.rect:
            ret["bbox"] = [
                self.rect["x1"],
                self.rect["y1"],
                self.rect["x2"] - self.rect["x1"],
                self.rect["y2"] - self.rect["y1"],
            ]
        if self.rect_head:
            ret["bbox_head"] = [
                self.rect_head["x1"],
                self.rect_head["y1"],
                self.rect_head["x2"] - self.rect_head["x1"],
                self.rect_head["y2"] - self.rect_head["y1"],
            ]
        return ret

    def to_old(self):
        """Return a dictionary representation for the PoseTrack17 format."""
        keypoints = []
        for landmark_info in self.landmarks:
            if (
                landmark_info["x"] == 0
                and landmark_info["y"] == 0
                and "is_visible" in landmark_info.keys()
                and landmark_info["is_visible"] == 0
            ):
                # The points in new format are stored like this if they're unannotated.
                # Skip in that case.
                continue
            point = {
                "id": [landmark_info["id"]],
                "x": [landmark_info["x"]],
                "y": [landmark_info["y"]],
            }
            if "score" in landmark_info.keys():
                point["score"] = [landmark_info["score"]]
            if "is_visible" in landmark_info.keys():
                point["is_visible"] = [landmark_info["is_visible"]]
            keypoints.append(point)
        # ret = {"track_id": [self.track_id], "annopoints": keypoints}
        ret = {"track_id": [self.track_id], "annopoints": [{'point': keypoints}]}
        if self.rect_head:
            ret["x1"] = [self.rect_head["x1"]]
            ret["x2"] = [self.rect_head["x2"]]
            ret["y1"] = [self.rect_head["y1"]]
            ret["y2"] = [self.rect_head["y2"]]
        if self.score:
            ret["score"] = [self.score]
        return ret

    @classmethod
    def from_old(cls, person_info):
        """Parse a dictionary representation from the PoseTrack17 format."""
        global SCORE_WARNING_EMITTED  # pylint: disable=global-statement
        person = Person(person_info["track_id"][0])
        assert len(person_info["track_id"]) == 1, "Invalid format!"
        rect_head = {}
        rect_head["x1"] = person_info["x1"][0]
        assert len(person_info["x1"]) == 1, "Invalid format!"
        rect_head["x2"] = person_info["x2"][0]
        assert len(person_info["x2"]) == 1, "Invalid format!"
        rect_head["y1"] = person_info["y1"][0]
        assert len(person_info["y1"]) == 1, "Invalid format!"
        rect_head["y2"] = person_info["y2"][0]
        assert len(person_info["y2"]) == 1, "Invalid format!"
        person.rect_head = rect_head
        try:
            person.score = person_info["score"][0]
            assert len(person_info["score"]) == 1, "Invalid format!"
        except KeyError:
            pass
        person.landmarks = []
        if "annopoints" not in person_info.keys() or not person_info["annopoints"]:
            return person
        lm_x_values = []
        lm_y_values = []
        for landmark_info in person_info["annopoints"][0]["point"]:
            lm_dict = {
                "y": landmark_info["y"][0],
                "x": landmark_info["x"][0],
                "id": landmark_info["id"][0],
            }
            lm_x_values.append(lm_dict["x"])
            lm_y_values.append(lm_dict["y"])
            if "score" in landmark_info.keys():
                lm_dict["score"] = landmark_info["score"][0]
                assert len(landmark_info["score"]) == 1, "Invalid format!"
            elif not SCORE_WARNING_EMITTED:
                LOGGER.warning("No landmark scoring information found!")
                LOGGER.warning("This will not be a valid submission file!")
                SCORE_WARNING_EMITTED = True
            if "is_visible" in landmark_info.keys():
                lm_dict["is_visible"] = landmark_info["is_visible"][0]
            person.landmarks.append(lm_dict)
            assert (
                len(landmark_info["x"]) == 1
                and len(landmark_info["y"]) == 1
                and len(landmark_info["id"]) == 1
            ), "Invalid format!"
        lm_x_values = np.array(lm_x_values)
        lm_y_values = np.array(lm_y_values)
        x_extent = lm_x_values.max() - lm_x_values.min()
        y_extent = lm_y_values.max() - lm_y_values.min()
        x_center = (lm_x_values.max() + lm_x_values.min()) / 2.
        y_center = (lm_y_values.max() + lm_y_values.min()) / 2.
        x1_final = x_center - x_extent * 0.65
        x2_final = x_center + x_extent * 0.65
        y1_final = y_center - y_extent * 0.65
        y2_final = y_center + y_extent * 0.65
        person.rect = {"x1": x1_final, "x2": x2_final, "y1": y1_final, "y2": y2_final}
        return person

    @classmethod
    def from_new(cls, person_info, conversion_table):
        """Parse a dictionary representation from the PoseTrack18 format."""
        global SCORE_WARNING_EMITTED  # pylint: disable=global-statement
        person = Person(person_info["track_id"])
        try:
            rect_head = {}
            rect_head["x1"] = person_info["bbox_head"][0]
            rect_head["x2"] = person_info["bbox_head"][0] + person_info["bbox_head"][2]
            rect_head["y1"] = person_info["bbox_head"][1]
            rect_head["y2"] = person_info["bbox_head"][1] + person_info["bbox_head"][3]
            person.rect_head = rect_head
        except KeyError:
            person.rect_head = None
        try:
            rect = {}
            rect["x1"] = person_info["bbox"][0]
            rect["x2"] = person_info["bbox"][0] + person_info["bbox"][2]
            rect["y1"] = person_info["bbox"][1]
            rect["y2"] = person_info["bbox"][1] + person_info["bbox"][3]
            person.rect = rect
        except KeyError:
            person.rect = None
        if "score" in person_info.keys():
            person.score = person_info["score"]
        try:
            landmark_scores = person_info["scores"]
        except KeyError:
            landmark_scores = None
            if not SCORE_WARNING_EMITTED:
                LOGGER.warning("No landmark scoring information found!")
                LOGGER.warning("This will not be a valid submission file!")
                SCORE_WARNING_EMITTED = True
        person.landmarks = []
        for landmark_idx, landmark_info in enumerate(
            np.array(person_info["keypoints"]).reshape(len(conversion_table), 3)
        ):
            landmark_idx_can = conversion_table[landmark_idx]
            if landmark_idx_can is not None:
                lm_info = {
                    "y": landmark_info[1],
                    "x": landmark_info[0],
                    "id": landmark_idx_can,
                    "is_visible": landmark_info[2],
                }
                if landmark_scores:
                    lm_info["score"] = landmark_scores[landmark_idx]
                person.landmarks.append(lm_info)
        return person


class Image:

    """An image with annotated people on it."""

    def __init__(self, filename, frame_id):
        self.posetrack_filename = filename
        self.frame_id = frame_id
        self.people = []
        self.ignore_regions = None  # None or tuple of (regions_x, regions_y), each a
        # list of lists of polygon coordinates.

    def to_new(self):
        """
        Return a dictionary representation for the PoseTrack18 format.

        The field 'vid_id' must still be added.
        """
        ret = {
            "file_name": self.posetrack_filename,
            "has_no_densepose": True,
            "is_labeled": (len(self.people) > 0),
            "frame_id": self.frame_id,
            # vid_id and nframes are inserted later.
        }
        if self.ignore_regions:
            ret["ignore_regions_x"] = self.ignore_regions[0]
            ret["ignore_regions_y"] = self.ignore_regions[1]
        return ret

    def to_old(self):
        """
        Return a dictionary representation for the PoseTrack17 format.

        People are added later.
        """
        ret = {"name": self.posetrack_filename}
        if self.ignore_regions:
            ir_list = []
            for plist_x, plist_y in zip(self.ignore_regions[0], self.ignore_regions[1]):
                r_list = []
                for x_val, y_val in zip(plist_x, plist_y):
                    r_list.append({"x": [x_val], "y": [y_val]})
                ir_list.append({"point": r_list})
        else:
            ir_list = None
        imgnum = int(osp.basename(self.posetrack_filename).split(".")[0]) + 1
        return ret, ir_list, imgnum

    @classmethod
    def from_old(cls, json_data):
        """Parse a dictionary representation from the PoseTrack17 format."""
        posetrack_filename = json_data["image"][0]["name"]
        assert len(json_data["image"]) == 1, "Invalid format!"
        old_seq_fp = osp.basename(osp.dirname(posetrack_filename))
        fp_wo_ending = osp.basename(posetrack_filename).split(".")[0]
        if "_" in fp_wo_ending:
            fp_wo_ending = fp_wo_ending.split("_")[0]
        old_frame_id = int(fp_wo_ending)
        try:
            frame_id = posetrack18_fname2id(old_seq_fp, old_frame_id)
        except:  # pylint: disable=bare-except
            print("I stumbled over a strange sequence. Maybe you can have a look?")
            import pdb

            pdb.set_trace()  # pylint: disable=no-member
        image = Image(posetrack_filename, frame_id)
        for person_info in json_data["annorect"]:
            image.people.append(Person.from_old(person_info))
        if "ignore_regions" in json_data.keys():
            ignore_regions_x = []
            ignore_regions_y = []
            for ignore_region in json_data["ignore_regions"]:
                x_values = []
                y_values = []
                for point in ignore_region["point"]:
                    x_values.append(point["x"][0])
                    y_values.append(point["y"][0])
                ignore_regions_x.append(x_values)
                ignore_regions_y.append(y_values)
            image.ignore_regions = (ignore_regions_x, ignore_regions_y)
        return image

    @classmethod
    def from_new(cls, track_data, image_id):
        """Parse a dictionary representation from the PoseTrack18 format."""
        image_info = [
            image_info
            for image_info in track_data["images"]
            if image_info["id"] == image_id
        ][0]
        posetrack_filename = image_info["file_name"]
        # license, coco_url, height, width, date_capture, flickr_url, id are lost.
        old_seq_fp = osp.basename(osp.dirname(posetrack_filename))
        old_frame_id = int(osp.basename(posetrack_filename).split(".")[0])
        frame_id = posetrack18_fname2id(old_seq_fp, old_frame_id)
        image = Image(posetrack_filename, frame_id)
        if (
            "ignore_regions_x" in image_info.keys()
            and "ignore_regions_y" in image_info.keys()
        ):
            image.ignore_regions = (
                image_info["ignore_regions_x"],
                image_info["ignore_regions_y"],
            )
        return image


class Video:

    """
    A PoseTrack sequence.

    Parameters
    ==========

    video_id: str.
      A five or six digit number, potentially with leading zeros, identifying the
      PoseTrack video.
    """

    def __init__(self, video_id):
        self.posetrack_video_id = video_id  # str.
        self.frames = []  # list of Image objects.

    def to_new(self):
        """Return a dictionary representation for the PoseTrack18 format."""
        result = {"images": [], "annotations": []}
        for image in self.frames:
            image_json = image.to_new()
            image_json["vid_id"] = self.posetrack_video_id
            image_json["nframes"] = len(self.frames)
            image_json["id"] = int(image.frame_id)
            result["images"].append(image_json)
            for person_idx, person in enumerate(image.people):
                person_json = person.to_new()
                person_json["image_id"] = int(image.frame_id)
                person_json["id"] = int(image.frame_id) * 100 + person_idx
                result["annotations"].append(person_json)
        # Write the 'categories' field.
        result["categories"] = [
            {
                "supercategory": "person",
                "name": "person",
                "skeleton": [
                    [16, 14],
                    [14, 12],
                    [17, 15],
                    [15, 13],
                    [12, 13],
                    [6, 12],
                    [7, 13],
                    [6, 7],
                    [6, 8],
                    [7, 9],
                    [8, 10],
                    [9, 11],
                    [2, 3],
                    [1, 2],
                    [1, 3],
                    [2, 4],
                    [3, 5],
                    [4, 6],
                    [5, 7],
                ],
                "keypoints": POSETRACK18_LM_NAMES_COCO_ORDER,
                "id": 1,
            }
        ]
        return result

    def to_old(self):
        """Return a dictionary representation for the PoseTrack17 format."""
        res = {"annolist": []}
        for image in self.frames:
            elem = {}
            im_rep, ir_list, imgnum = image.to_old()
            elem["image"] = [im_rep]
            elem["imgnum"] = [imgnum]
            if ir_list:
                elem["ignore_regions"] = ir_list
            elem["annorect"] = []
            for person in image.people:
                elem["annorect"].append(person.to_old())
            if image.people:
                elem['is_labeled'] = [1]
            else:
                elem['is_labeled'] = [0]
            res["annolist"].append(elem)
        return res

    @classmethod
    def from_old(cls, track_data):
        """Parse a dictionary representation from the PoseTrack17 format."""
        assert "annolist" in track_data.keys(), "Wrong format!"
        video = None
        for image_info in track_data["annolist"]:
            image = Image.from_old(image_info)
            if not video:
                video = Video(
                    osp.basename(osp.dirname(image.posetrack_filename)).split("_")[0]
                )
            else:
                assert (
                    video.posetrack_video_id
                    == osp.basename(osp.dirname(image.posetrack_filename)).split("_")[
                        0
                    ]
                )
            video.frames.append(image)
        return [video]

    @classmethod
    def from_new(cls, track_data):
        """Parse a dictionary representation from the PoseTrack17 format."""
        image_id_to_can_info = {}
        video_id_to_video = {}
        assert len(track_data["categories"]) == 1
        assert track_data["categories"][0]["name"] == "person"
        assert len(track_data["categories"][0]["keypoints"]) in [15, 17]
        conversion_table = []
        for lm_name in track_data["categories"][0]["keypoints"]:
            if lm_name not in POSETRACK18_LM_NAMES:
                conversion_table.append(None)
            else:
                conversion_table.append(POSETRACK18_LM_NAMES.index(lm_name))
        for lm_idx, lm_name in enumerate(POSETRACK18_LM_NAMES):
            assert lm_idx in conversion_table, "Landmark `%s` not found." % (lm_name)
        videos = []
        for image_id in [image["id"] for image in track_data["images"]]:
            image = Image.from_new(track_data, image_id)
            video_id = osp.basename(osp.dirname(image.posetrack_filename)).split(
                "_"
            )[0]
            if video_id in video_id_to_video.keys():
                video = video_id_to_video[video_id]
            else:
                video = Video(video_id)
                video_id_to_video[video_id] = video
                videos.append(video)
            video.frames.append(image)
            for person_info in track_data["annotations"]:
                if person_info["image_id"] != image_id:
                    continue
                image.people.append(Person.from_new(person_info, conversion_table))
        return videos


def convert_videos(track_data):
    """Convert between PoseTrack18 and PoseTrack17 format."""
    if "annolist" in track_data.keys():
        old_to_new = True
        #LOGGER.info("Detected PoseTrack17 format. Converting to 2018...")
    else:
        old_to_new = False
        assert "images" in track_data.keys(), "Unknown image format. :("
        #LOGGER.info("Detected PoseTrack18 format. Converting to 2017...")

    if (old_to_new):
        videos = Video.from_old(track_data)
        videos_converted = [v.to_new() for v in videos]
    else:
        videos = Video.from_new(track_data)
        videos_converted = [v.to_old() for v in videos]
    return videos_converted
