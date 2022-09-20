import copy

import numpy as np
import os.path as osp
from pycocotools.coco import COCO
import logging
from tabulate import tabulate
import cv2
import random
import torch
from collections import OrderedDict

from dataset.Base import BaseDataset
from utils.bbox import box2cs
from utils.evaluate import convert_data_to_annorect_struct, evaluate
from utils.heatmap import generate_heatmaps
from utils.rw_json import read_json_from_file, write_json_to_file
from utils.setup import create_folder, video2filenames
from utils.transform import build_transforms, half_body_transform, fliplr_joints, get_affine_transform, \
    exec_affine_transform
import torchvision.transforms as T


class PoseTrackDataset(BaseDataset):
    def __init__(self, cfg, phase, **kwargs):
        super(PoseTrackDataset, self).__init__(cfg, phase, **kwargs)

        self.train = True if phase == 'train' else False
        self.is_posetrack18 = cfg.DATASET.IS_2018

        self.flip_pairs = [[3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]
        self.joints_weight = np.array([1., 1., 1., 1., 1., 1., 1., 1.2, 1.2, 1.5, 1.5, 1., 1., 1.2, 1.2, 1.5, 1.5],
                                      dtype=np.float32).reshape((self.num_joints, 1))
        self.upper_body_ids = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        self.lower_body_ids = (11, 12, 13, 14, 15, 16)

        self.transform = build_transforms(cfg, phase)

        # temporal distance for PRF
        self.distance_whole_otherwise_segment = cfg.DISTANCE_WHOLE_OTHERWISE_SEGMENT
        self.distance = cfg.DISTANCE
        self.previous_distance = cfg.PREVIOUS_DISTANCE
        self.next_distance = cfg.NEXT_DISTANCE

        self.random_aux_frame = cfg.DATASET.RANDOM_AUX_FRAME

        self.bbox_enlarge_factor = cfg.DATASET.BBOX_ENLARGE_FACTOR
        self.sigma = cfg.MODEL.SIGMA

        self.img_dir = cfg.DATASET.IMG_DIR
        self.json_dir = cfg.DATASET.JSON_DIR
        self.test_on_train = cfg.DATASET.TEST_ON_TRAIN
        self.json_file = cfg.DATASET.JSON_FILE

        if self.phase != 'train':
            self.img_dir = cfg.DATASET.TEST_IMG_DIR
            temp_subCfgNode = cfg.VAL if self.phase == 'validate' else cfg.TEST
            self.nms_thre = temp_subCfgNode.NMS_THRE
            self.image_thre = temp_subCfgNode.IMAGE_THRE
            self.soft_nms = temp_subCfgNode.SOFT_NMS
            self.oks_thre = temp_subCfgNode.OKS_THRE
            self.in_vis_thre = temp_subCfgNode.IN_VIS_THRE
            self.bbox_file = temp_subCfgNode.COCO_BBOX_FILE
            self.use_gt_bbox = temp_subCfgNode.USE_GT_BBOX
            self.annotation_dir = temp_subCfgNode.ANNOT_DIR

        self.coco = COCO(osp.join(self.json_dir, 'posetrack_train.json' if self.train else 'posetrack_val.json'))
        # if self.phase == 'test':
        #     self.coco = COCO(osp.join(self.json_dir, 'posetrack_test.json'))
        cats = [cat['name'] for cat in self.coco.loadCats(self.coco.getCatIds())]
        self.classes = ['__background__'] + cats
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._class_to_coco_ind = dict(zip(cats, self.coco.getCatIds()))
        self._coco_ind_to_class_ind = dict(
            [(self._class_to_coco_ind[cls], self._class_to_ind[cls]) for cls in self.classes[1:]])
        self.image_set_index = self.coco.getImgIds()
        self.num_images = len(self.image_set_index)

        self.data = self._list_data()

        self.model_input_type = cfg.DATASET.INPUT_TYPE

        self.show_data_parameters()
        self.show_samples()

    def __getitem__(self, item_idx):
        data_item = copy.deepcopy(self.data[item_idx])
        if self.model_input_type == 'spatiotemporal_window':
            return self._get_spatio_temporal_window(data_item)

    def _list_data(self):
        if self.phase != 'test' or self.use_gt_bbox:
            # use bbox from annotation
            data = self._load_coco_keypoints_annotations()
        else:
            # use bbox from detection
            data = self._load_detection_results()
        return data

    def _load_coco_keypoints_annotations(self):
        """
            coco ann: [u'segmentation', u'area', u'iscrowd', u'image_id', u'bbox', u'category_id', u'id']
            iscrowd:
                crowd instances are handled by marking their overlaps with all categories to -1
                and later excluded in training
            bbox:
                [x1, y1, w, h]
        """
        gt_db = []
        for index in self.image_set_index:
            im_ann = self.coco.loadImgs(index)[0]
            width = im_ann['width']
            height = im_ann['height']

            file_name = im_ann['file_name']

            nframes = int(im_ann['nframes'])
            frame_id = int(im_ann['frame_id'])

            annIds = self.coco.getAnnIds(imgIds=index, iscrowd=False)
            objs = self.coco.loadAnns(annIds)

            # sanitize bboxes
            valid_objs = []
            for obj in objs:
                x, y, w, h = obj['bbox']
                x1 = np.max((0, x))
                y1 = np.max((0, y))
                x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
                y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
                if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                    obj['clean_bbox'] = [x1, y1, x2 - x1, y2 - y1]
                    valid_objs.append(obj)
            objs = valid_objs

            rec = []
            for obj in objs:
                cls = self._coco_ind_to_class_ind[obj['category_id']]
                if cls != 1:
                    continue

                # ignore objs without keypoints annotation
                if max(obj['keypoints']) == 0:
                    continue

                joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
                joints_3d_vis = np.zeros((self.num_joints, 3), dtype=np.float)
                for ipt in range(self.num_joints):
                    joints_3d[ipt, 0] = obj['keypoints'][ipt * 3 + 0]
                    joints_3d[ipt, 1] = obj['keypoints'][ipt * 3 + 1]
                    joints_3d[ipt, 2] = 0
                    t_vis = obj['keypoints'][ipt * 3 + 2]
                    if t_vis > 1:
                        t_vis = 1
                    joints_3d_vis[ipt, 0] = t_vis
                    joints_3d_vis[ipt, 1] = t_vis
                    joints_3d_vis[ipt, 2] = 0

                center, scale = box2cs(obj['clean_bbox'][:4], self.aspect_ratio, self.bbox_enlarge_factor)
                rec.append({
                    'image': osp.join(self.img_dir, file_name),
                    'center': center,
                    'scale': scale,
                    'box': obj['clean_bbox'][:4],
                    'joints_3d': joints_3d,
                    'joints_3d_vis': joints_3d_vis,
                    'filename': '',
                    'imgnum': 0,
                    'nframes': nframes,
                    'frame_id': frame_id,
                })
            gt_db.extend(rec)
        return gt_db

    def _load_detection_results(self):
        logger = logging.getLogger(__name__)
        print("=> Load bbox file from {}".format(self.bbox_file))
        all_boxes = read_json_from_file(self.bbox_file)

        if not all_boxes:
            logger.error('=> Load %s fail!' % self.bbox_file)
            return None

        kpt_data = []
        num_boxes = 0
        for n_img in range(0, len(all_boxes)):
            det_res = all_boxes[n_img]
            if det_res['category_id'] != 1:
                continue
            img_name = det_res['image_name']
            box = det_res['bbox']  # xywh
            score = det_res['score']
            nframes = det_res['nframes']
            frame_id = det_res['frame_id']

            if score < self.image_thre:
                continue

            num_boxes = num_boxes + 1

            center, scale = box2cs(box, self.aspect_ratio, self.bbox_enlarge_factor)
            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
            joints_3d_vis = np.ones(
                (self.num_joints, 3), dtype=np.float)
            kpt_data.append({
                'image': osp.join(self.img_dir, img_name),
                'center': center,
                'scale': scale,
                'score': score,
                'joints_3d': joints_3d,
                'joints_3d_vis': joints_3d_vis,
                'filename': '',
                'imgnum': 0,
                'nframes': nframes,
                'frame_id': frame_id,
            })
        # print('=> Total boxes: {}'.format(len(all_boxes)))
        # print('=> Total boxes after filter low score@{}: {}'.format(self.image_thre, num_boxes))

        table_header = ["Total boxes", "Filter threshold", "Remaining boxes"]
        table_data = [[len(all_boxes), self.image_thre, num_boxes]]
        table = tabulate(table_data, tablefmt="pipe", headers=table_header, numalign="left")
        print(f"=> Boxes Info Table: \n" + table)

        return kpt_data

    def _get_spatio_temporal_window(self, data_item):
        filename = data_item['filename']
        img_num = data_item['imgnum']
        image_file_path = data_item['image']
        num_frames = data_item['nframes']
        data_numpy = cv2.imread(image_file_path)

        # if data_numpy == None:
        #     continue
        zero_fill = len(osp.basename(image_file_path).replace('.jpg', ''))

        if zero_fill == 6:
            is_posetrack18 = True
        else:
            is_posetrack18 = False

        current_idx = int(osp.basename(image_file_path).replace('.jpg', ''))

        # if self.distance_whole_otherwise_segment:
        farthest_distance = self.distance
        prev_delta_range = range(1, min((current_idx + 1) if is_posetrack18 else current_idx, farthest_distance + 1))
        next_delta_range = range(1,
                                 min((num_frames - current_idx) if is_posetrack18 else (num_frames - current_idx + 1),
                                     farthest_distance + 1))
        # next_delta_range = range(1,
        #                          min((num_frames - current_idx), farthest_distance + 1))
        # else:
        #     prev_delta_range = range(1, min(current_idx + 1 if is_posetrack18 else current_idx, self.previous_distance))
        #     next_delta_range = range(1, min((num_frames - current_idx) if is_posetrack18 else (num_frames - current_idx + 1),
        #                                     self.next_distance))

        prev_delta_range = list(prev_delta_range)
        next_delta_range = list(next_delta_range)
        # setting deltas

        if len(prev_delta_range) == 0:
            prev_delta = 0
            margin_left = 0
            pprev_delta = 0
            margin_lleft = 0
        elif len(prev_delta_range) == 1:
            prev_delta = prev_delta_range[0]
            margin_left = prev_delta
            pprev_delta = 0
            margin_lleft = 0
        else:
            prev_delta = prev_delta_range[0]
            margin_left = prev_delta
            pprev_delta = prev_delta_range[1]
            margin_lleft = pprev_delta

        if len(next_delta_range) == 0:
            next_delta = 0
            margin_right = 0
            nnext_delta = 0
            margin_rright = 0
        elif len(next_delta_range) == 1:
            next_delta = next_delta_range[-1]
            margin_right = next_delta
            nnext_delta = 0
            margin_rright = 0
        else:
            next_delta = next_delta_range[0]
            margin_right = next_delta
            nnext_delta = next_delta_range[0]
            margin_rright = nnext_delta

        prev_idx = current_idx - prev_delta
        next_idx = current_idx + next_delta

        pprev_idx = current_idx - pprev_delta
        nnext_idx = current_idx + nnext_delta

        prev_image_file = osp.join(osp.dirname(image_file_path), str(prev_idx).zfill(zero_fill) + '.jpg')
        next_image_file = osp.join(osp.dirname(image_file_path), str(next_idx).zfill(zero_fill) + '.jpg')
        pprev_image_file = osp.join(osp.dirname(image_file_path), str(pprev_idx).zfill(zero_fill) + '.jpg')
        nnext_image_file = osp.join(osp.dirname(image_file_path), str(nnext_idx).zfill(zero_fill) + '.jpg')

        # checking for files existence
        if not osp.exists(prev_image_file):
            prev_image_file = image_file_path
            margin_left = 0

        if not osp.exists(next_image_file):
            next_image_file = image_file_path
            margin_right = 0

        data_numpy_prev = cv2.imread(prev_image_file)
        data_numpy_next = cv2.imread(next_image_file)
        data_numpy_pprev = cv2.imread(pprev_image_file)
        data_numpy_nnext = cv2.imread(nnext_image_file)

        if data_numpy is None:
            print('=> fail to read {}'.format(image_file_path))
            raise ValueError('Fail to read {}'.format(image_file_path))
        if data_numpy_prev is None:
            print('=> PREV SUP: fail to read {}'.format(prev_image_file))
            raise ValueError('PREV SUP: Fail to read {}'.format(prev_image_file))
        if data_numpy_next is None:
            print('=> NEXT SUP: fail to read {}'.format(next_image_file))
            raise ValueError('NEXT SUP: Fail to read {}'.format(next_image_file))

        if self.color_rgb:
            # cv2 read_image  color channel is BGR
            data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)
            data_numpy_prev = cv2.cvtColor(data_numpy_prev, cv2.COLOR_BGR2RGB)
            data_numpy_next = cv2.cvtColor(data_numpy_next, cv2.COLOR_BGR2RGB)
            data_numpy_pprev = cv2.cvtColor(data_numpy_pprev, cv2.COLOR_BGR2RGB)
            data_numpy_nnext = cv2.cvtColor(data_numpy_nnext, cv2.COLOR_BGR2RGB)

        joints = data_item['joints_3d']
        joints_vis = data_item['joints_3d_vis']

        center = data_item["center"]
        scale = data_item["scale"]

        score = data_item['score'] if 'score' in data_item else 1
        r = 0

        if self.train:
            if np.sum(joints_vis[:, 0]) > self.num_joints_half_body and np.random.rand() < self.prob_half_body:
                c_half_body, s_half_body = half_body_transform(joints, joints_vis, self.num_joints, self.upper_body_ids,
                                                               self.aspect_ratio,
                                                               self.pixel_std)
                center, scale = c_half_body, s_half_body

            scale_factor = self.scale_factor
            rotation_factor = self.rotation_factor
            if isinstance(scale_factor, list) or isinstance(scale_factor, tuple):
                scale_factor = scale_factor[0]
            scale = scale * np.clip(np.random.randn() * scale_factor + 1, 1 - scale_factor, 1 + scale_factor)
            r = np.clip(np.random.randn() * rotation_factor, -rotation_factor * 2, rotation_factor * 2) \
                if random.random() <= 0.6 else 0

            if self.flip and random.random() <= 0.5:
                data_numpy = data_numpy[:, ::-1, :]
                data_numpy_prev = data_numpy_prev[:, ::-1, :]
                data_numpy_next = data_numpy_next[:, ::-1, :]
                data_numpy_pprev = data_numpy_pprev[:, ::-1, :]
                data_numpy_nnext = data_numpy_nnext[:, ::-1, :]

                joints, joints_vis = fliplr_joints(
                    joints, joints_vis, data_numpy.shape[1], self.flip_pairs)
                center[0] = data_numpy.shape[1] - center[0] - 1

            # augmentation blur 적용하기
            if random.random() <= 0.5:
                data_numpy = (T.GaussianBlur(kernel_size=(5, 9),
                                             sigma=(0.1, 5))(torch.from_numpy(data_numpy.copy()))).numpy()
                data_numpy_prev = (T.GaussianBlur(kernel_size=(5, 9),
                                                  sigma=(0.1, 5))(torch.from_numpy(data_numpy_prev.copy()))).numpy()
                data_numpy_next = (T.GaussianBlur(kernel_size=(5, 9),
                                                  sigma=(0.1, 5))(torch.from_numpy(data_numpy_next.copy()))).numpy()
                data_numpy_pprev = (T.GaussianBlur(kernel_size=(5, 9),
                                                   sigma=(0.1, 5))(
                    torch.from_numpy(data_numpy_pprev.copy()))).numpy()
                data_numpy_nnext = (T.GaussianBlur(kernel_size=(5, 9),
                                                   sigma=(0.1, 5))(
                    torch.from_numpy(data_numpy_nnext.copy()))).numpy()

        # calculate transform matrix
        trans = get_affine_transform(center, scale, r, self.image_size)
        input_x = cv2.warpAffine(data_numpy, trans, (int(self.image_size[0]), int(self.image_size[1])),
                                 flags=cv2.INTER_LINEAR)
        input_prev = cv2.warpAffine(data_numpy_prev, trans, (int(self.image_size[0]), int(self.image_size[1])),
                                    flags=cv2.INTER_LINEAR)
        input_next = cv2.warpAffine(data_numpy_next, trans, (int(self.image_size[0]), int(self.image_size[1])),
                                    flags=cv2.INTER_LINEAR)
        input_pprev = cv2.warpAffine(data_numpy_pprev, trans, (int(self.image_size[0]), int(self.image_size[1])),
                                     flags=cv2.INTER_LINEAR)
        input_nnext = cv2.warpAffine(data_numpy_nnext, trans, (int(self.image_size[0]), int(self.image_size[1])),
                                     flags=cv2.INTER_LINEAR)

        if self.transform:
            input_x = self.transform(input_x)
            input_prev = self.transform(input_prev)
            input_next = self.transform(input_next)
            input_pprev = self.transform(input_pprev)
            input_nnext = self.transform(input_nnext)

        # joint transform like image
        for i in range(self.num_joints):
            if joints_vis[i, 0] > 0.0:  # joints_vis   num_joints  x 3 (x_vis,y_vis)
                joints[i, 0:2] = exec_affine_transform(joints[i, 0:2], trans)

        # H W
        for index, joint in enumerate(joints):
            x, y, _ = joint
            if x < 0 or y < 0 or x > self.image_size[0] or y > self.image_size[1]:
                joints_vis[index] = [0, 0, 0]

        target_heatmaps, target_heatmaps_weight = generate_heatmaps(joints, joints_vis, self.sigma, self.image_size,
                                                                    self.heatmap_size,
                                                                    self.num_joints,
                                                                    use_different_joints_weight=self.use_different_joints_weight,
                                                                    joints_weight=self.joints_weight)
        target_heatmaps = torch.from_numpy(target_heatmaps)  # H W
        target_heatmaps_weight = torch.from_numpy(target_heatmaps_weight)

        meta = {
            'image': image_file_path,
            'prev_sup_image': prev_image_file,
            'next_sup_image': next_image_file,
            'prev_sup_image': pprev_image_file,
            'next_sup_image': nnext_image_file,
            'filename': filename,
            'imgnum': img_num,
            'joints': joints,
            'joints_vis': joints_vis,
            'center': center,
            'scale': scale,
            'rotation': r,
            'score': score,
            'margin_left': margin_left,
            'margin_right': margin_right,
            'margin_lleft': margin_lleft,
            'margin_rright': margin_rright,
            'width': data_numpy.shape[0],
            'height': data_numpy.shape[1]
        }

        # return input_x, input_prev, input_next, target_heatmaps, target_heatmaps_weight, meta
        return input_x, input_prev, input_next, input_pprev, input_nnext, \
               target_heatmaps, target_heatmaps_weight, meta

    def evaluate(self, cfg, preds, output_dir, boxes, img_path, *args, **kwargs):
        logger = logging.getLogger(__name__)
        logger.info("=> Start evaluate")
        if self.phase == 'validate':
            output_dir = osp.join(output_dir, 'val_set_json_results')
        else:
            output_dir = osp.join(output_dir, 'test_set_json_results')

        create_folder(output_dir)

        ### processing our preds
        video_map = {}
        vid2frame_map = {}
        vid2name_map = {}

        all_preds = []
        all_boxes = []
        all_tracks = []
        cc = 0

        # print(img_path)
        for key in img_path:
            temp = key.split('/')

            # video_name = osp.dirname(key)
            video_name = temp[len(temp) - 3] + '/' + temp[len(temp) - 2]
            img_sfx = temp[len(temp) - 3] + '/' + temp[len(temp) - 2] + '/' + temp[len(temp) - 1]

            prev_nm = temp[len(temp) - 1]
            frame_num = int(prev_nm.replace('.jpg', ''))
            if not video_name in video_map:
                video_map[video_name] = [cc]
                vid2frame_map[video_name] = [frame_num]
                vid2name_map[video_name] = [img_sfx]
            else:
                video_map[video_name].append(cc)
                vid2frame_map[video_name].append(frame_num)
                vid2name_map[video_name].append(img_sfx)

            idx_list = img_path[key]
            pose_list = []
            box_list = []
            for idx in idx_list:
                temp = np.zeros((4, 17))
                temp[0, :] = preds[idx, :, 0]
                temp[1, :] = preds[idx, :, 1]
                temp[2, :] = preds[idx, :, 2]
                temp[3, :] = preds[idx, :, 2]
                pose_list.append(temp)

                temp = np.zeros((1, 6))
                temp[0, :] = boxes[idx, :]
                box_list.append(temp)

            all_preds.append(pose_list)
            all_boxes.append(box_list)
            cc += 1

        annot_dir = self.annotation_dir
        is_posetrack18 = self.is_posetrack18

        out_data = {}
        out_filenames, L = video2filenames(annot_dir)

        for vid in video_map:
            idx_list = video_map[vid]
            c = 0
            used_frame_list = []
            if 'images/' + vid in L.keys():
                cur_length = L['images/' + vid]

                temp_kps_map = {}
                temp_track_kps_map = {}
                temp_box_map = {}

                for idx in idx_list:
                    frame_num = vid2frame_map[vid][c]
                    img_sfx = vid2name_map[vid][c]
                    c += 1

                    used_frame_list.append(frame_num)

                    kps = all_preds[idx]
                    temp_kps_map[frame_num] = (img_sfx, kps)

                    bb = all_boxes[idx]
                    temp_box_map[frame_num] = bb
                #### including empty frames
                nnz_counter = 0
                next_track_id = 0

                if not is_posetrack18:
                    sid = 1
                    fid = cur_length + 1
                else:
                    sid = 0
                    fid = cur_length
                # start id ~ finish id
                for current_frame_id in range(sid, fid):
                    frame_num = current_frame_id
                    if not current_frame_id in used_frame_list:
                        temp_sfx = vid2name_map[vid][0]
                        arr = temp_sfx.split('/')
                        if not is_posetrack18:
                            img_sfx = arr[0] + '/' + arr[1] + '/' + str(frame_num).zfill(8) + '.jpg'
                        else:
                            img_sfx = arr[0] + '/' + arr[1] + '/' + str(frame_num).zfill(6) + '.jpg'
                        kps = []
                        tracks = []
                        bboxs = []

                    else:

                        img_sfx = temp_kps_map[frame_num][0]
                        kps = temp_kps_map[frame_num][1]
                        bboxs = temp_box_map[frame_num]
                        tracks = [track_id for track_id in range(len(kps))]
                        # tracks = [1] * len(kps)

                    ### creating a data element
                    data_el = {
                        'image': {'name': img_sfx},
                        'imgnum': [frame_num],
                        'annorect': convert_data_to_annorect_struct(kps, tracks, bboxs),
                    }
                    if vid in out_data:
                        out_data[vid].append(data_el)
                    else:
                        out_data[vid] = [data_el]

        print("=> saving files for evaluation")
        #### saving files for evaluation
        for vname in out_data:
            vdata = out_data[vname]
            outfpath = osp.join(output_dir, out_filenames[osp.join('images', vname)])

            write_json_to_file({'annolist': vdata}, outfpath)

        # run evaluation
        # AP = self._run_eval(annot_dir, output_dir)[0]
        AP, mota = evaluate(annot_dir, output_dir, eval_track=False)

        name_value = [
            ('Head', AP[0]),
            ('Shoulder', AP[1]),
            ('Elbow', AP[2]),
            ('Wrist', AP[3]),
            ('Hip', AP[4]),
            ('Knee', AP[5]),
            ('Ankle', AP[6]),
            ('Mean', AP[7])
        ]

        name_value = OrderedDict(name_value)

        return name_value, name_value['Mean']
