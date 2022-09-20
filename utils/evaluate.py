import glob
import json

import numpy as np
import os
import cv2

from configs.constants import PoseTrack_Keypoint_Pairs, COLOR_DICT, \
    PoseTrack_Official_Keypoint_Pairs, PoseTrack_Official_Keypoint_Ordering, PoseTrack_COCO_Keypoint_Ordering
import logging
import motmetrics as mm

from utils.heatmap import get_max_preds
from utils.keypoints import coco2posetrack_ord
from utils.setup import convert_videos
from shapely import geometry

MIN_SCORE = -9999
MAX_TRACK_ID = 10000


def removeIgnoredPointsRects(rects, polyList):
    ridxs = list(range(len(rects)))
    for ridx in range(len(rects)):
        points = rects[ridx]["annopoints"][0]["point"]
        pidxs = list(range(len(points)))
        for pidx in range(len(points)):
            pt = geometry.Point(points[pidx]["x"][0], points[pidx]["y"][0])
            bIgnore = False
            for poidx in range(len(polyList)):
                poly = polyList[poidx]
                if poly.contains(pt):
                    bIgnore = True
                    break
            if bIgnore:
                pidxs.remove(pidx)
        points = [points[pidx] for pidx in pidxs]
        if len(points) > 0:
            rects[ridx]["annopoints"][0]["point"] = points
        else:
            ridxs.remove(ridx)
    rects = [rects[ridx] for ridx in ridxs]
    return rects


def removeIgnoredPoints(gtFramesAll, prFramesAll):
    imgidxs = []
    for imgidx in range(len(gtFramesAll)):
        if ("ignore_regions" in gtFramesAll[imgidx].keys() and
                len(gtFramesAll[imgidx]["ignore_regions"]) > 0):
            regions = gtFramesAll[imgidx]["ignore_regions"]
            polyList = []
            for ridx in range(len(regions)):
                points = regions[ridx]["point"]
                pointList = []
                for pidx in range(len(points)):
                    pt = geometry.Point(points[pidx]["x"][0], points[pidx]["y"][0])
                    pointList += [pt]
                poly = geometry.Polygon([[p.x, p.y] for p in pointList])
                polyList += [poly]

            rects = prFramesAll[imgidx]["annorect"]
            prFramesAll[imgidx]["annorect"] = removeIgnoredPointsRects(rects, polyList)
            rects = gtFramesAll[imgidx]["annorect"]
            gtFramesAll[imgidx]["annorect"] = removeIgnoredPointsRects(rects, polyList)

    return gtFramesAll, prFramesAll


def rectHasPoints(rect):
    return (("annopoints" in rect.keys()) and
            (len(rect["annopoints"]) > 0 and len(rect["annopoints"][0]) > 0) and
            ("point" in rect["annopoints"][0].keys()))


def removeRectsWithoutPoints(rects):
    idxsPr = []
    for ridxPr in range(len(rects)):
        if rectHasPoints(rects[ridxPr]):
            idxsPr += [ridxPr]
    rects = [rects[ridx] for ridx in idxsPr]
    return rects


def cleanupData(gtFramesAll, prFramesAll):
    # remove all GT frames with empty annorects and remove corresponding entries from predictions
    imgidxs = []
    for imgidx in range(len(gtFramesAll)):
        if len(gtFramesAll[imgidx]["annorect"]) > 0:
            imgidxs += [imgidx]
    gtFramesAll = [gtFramesAll[imgidx] for imgidx in imgidxs]
    prFramesAll = [prFramesAll[imgidx] for imgidx in imgidxs]

    # remove all gt rectangles that do not have annotations
    for imgidx in range(len(gtFramesAll)):
        gtFramesAll[imgidx]["annorect"] = removeRectsWithoutPoints(gtFramesAll[imgidx]["annorect"])
        prFramesAll[imgidx]["annorect"] = removeRectsWithoutPoints(prFramesAll[imgidx]["annorect"])

    return gtFramesAll, prFramesAll


class Joint:
    def __init__(self):
        self.count = 15
        self.right_ankle = 0
        self.right_knee = 1
        self.right_hip = 2
        self.left_hip = 3
        self.left_knee = 4
        self.left_ankle = 5
        self.right_wrist = 6
        self.right_elbow = 7
        self.right_shoulder = 8
        self.left_shoulder = 9
        self.left_elbow = 10
        self.left_wrist = 11
        self.neck = 12
        self.nose = 13
        self.head_top = 14

        self.name = {self.right_ankle: "right_ankle", self.right_knee: "right_knee", self.right_hip: "right_hip",
                     self.right_shoulder: "right_shoulder", self.right_elbow: "right_elbow",
                     self.right_wrist: "right_wrist", self.left_ankle: "left_ankle", self.left_knee: "left_knee",
                     self.left_hip: "left_hip", self.left_shoulder: "left_shoulder", self.left_elbow: "left_elbow",
                     self.left_wrist: "left_wrist", self.neck: "neck", self.nose: "nose", self.head_top: "head_top"}

        self.symmetric_joint = {self.right_ankle: self.left_ankle, self.right_knee: self.left_knee,
                                self.right_hip: self.left_hip, self.right_shoulder: self.left_shoulder,
                                self.right_elbow: self.left_elbow, self.right_wrist: self.left_wrist,
                                self.left_ankle: self.right_ankle, self.left_knee: self.right_knee,
                                self.left_hip: self.right_hip, self.left_shoulder: self.right_shoulder,
                                self.left_elbow: self.right_elbow, self.left_wrist: self.right_wrist, self.neck: -1,
                                self.nose: -1, self.head_top: -1}


def getCum(vals):
    cum = []
    n = -1
    cum += [(vals[[Joint().head_top, Joint().neck, Joint().nose], 0].mean())]
    # cum += [(vals[[Joint().neck],0].mean())]

    cum += [(vals[[Joint().right_shoulder, Joint().left_shoulder], 0].mean())]
    cum += [(vals[[Joint().right_elbow, Joint().left_elbow], 0].mean())]
    cum += [(vals[[Joint().right_wrist, Joint().left_wrist], 0].mean())]
    cum += [(vals[[Joint().right_hip, Joint().left_hip], 0].mean())]
    cum += [(vals[[Joint().right_knee, Joint().left_knee], 0].mean())]
    cum += [(vals[[Joint().right_ankle, Joint().left_ankle], 0].mean())]
    for i in range(Joint().count, len(vals)):
        cum += [vals[i, 0]]
    return cum


def getFormatRow(cum):
    """
        cum  - val list
    """
    row = "&"
    for i in range(len(cum) - 1):
        row += formatCell(cum[i], " &")
    row += formatCell(cum[len(cum) - 1], (" %s" % "\\" + "\\"))
    return row


def formatCell(val, delim):
    return "{:>5}".format("%1.1f" % val) + delim


def getMotHeader():
    strHeader = "&"
    strHeader += " MOTA &"
    strHeader += " MOTA &"
    strHeader += " MOTA &"
    strHeader += " MOTA &"
    strHeader += " MOTA &"
    strHeader += " MOTA &"
    strHeader += " MOTA &"
    strHeader += " MOTA &"
    strHeader += " MOTP &"
    strHeader += " Prec &"
    strHeader += " Rec  %s\n" % ("\\" + "\\")
    strHeader += "&"
    strHeader += " Head &"
    strHeader += " Shou &"
    strHeader += " Elb  &"
    strHeader += " Wri  &"
    strHeader += " Hip  &"
    strHeader += " Knee &"
    strHeader += " Ankl &"
    strHeader += " Total&"
    strHeader += " Total&"
    strHeader += " Total&"
    strHeader += " Total%s" % ("\\" + "\\")

    return strHeader


def printTable(vals, motHeader=False):
    cum = getCum(vals)
    row = getFormatRow(cum)
    if motHeader:
        header = getMotHeader()
    else:
        header = getHeader()
        # header = ["Head", "Shou", "Elb", "Wri", "Hip", "Knee", "Ankl", "# Total"]

    # result = []
    # for item in cum:
    # result.append(item)
    # table = tabulate([result], tablefmt="pipe", headers=header, numalign="left",)
    logger = logging.getLogger(__name__)
    # print()
    # logger.info(f"=> Result Table: \n" + colored(table, "magenta"))
    logger.info(header)
    logger.info(row)
    # return header+"\n", row+"\n"
    return cum


def getHeader():
    strHeader = "&"
    strHeader += " Head &"
    strHeader += " Shou &"
    strHeader += " Elb  &"
    strHeader += " Wri  &"
    strHeader += " Hip  &"
    strHeader += " Knee &"
    strHeader += " Ankl &"
    strHeader += " Total%s" % ("\\" + "\\")
    return strHeader


def evaluate_tracking(gtFramesAll, prFramesAll, outputDir, saveAll=True, saveSeq=False):
    distThresh = 0.5
    # assign predicted poses to GT poses
    #    _, _, _, motAll = eval_helpers.assignGTmulti(gtFramesAll, prFramesAll, distThresh)
    _, _, _, motAll = assignGTmulti(gtFramesAll, prFramesAll, distThresh)
    
    # compute MOT metrics per part
    metricsAll = computeMetrics(gtFramesAll, motAll, outputDir, saveAll, saveSeq)

    return metricsAll


def save_fusion_images(dir_out, img, name='', heatmaps=None):
    if not os.path.isdir(dir_out):
        os.makedirs(dir_out)

    img = (img - img.min()) / (img.max() - img.min()) * 255
    for i in range(len(PoseTrack_COCO_Keypoint_Ordering)):
        k = PoseTrack_COCO_Keypoint_Ordering[i]
        heatmap = heatmaps[i]
        heatmap = np.clip(heatmap * 255, 0, 255).astype(np.uint8)
        colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_BONE)
        colored_heatmap = cv2.resize(colored_heatmap, (img.shape[1], img.shape[0]))
        img_heatmap = img * 0.3 + colored_heatmap * 0.7
        cv2.imwrite(os.path.join(dir_out, '{}{}_img_heatmap.png'.format(name, k)), img_heatmap)


def save_f_inv_images(dir_out, img, name='', heatmaps=None):
    if not os.path.isdir(dir_out):
        os.makedirs(dir_out)

    img = (img - img.min()) / (img.max() - img.min()) * 255
    for i in range(len(PoseTrack_COCO_Keypoint_Ordering)):
        k = PoseTrack_COCO_Keypoint_Ordering[i]
        heatmap = heatmaps[i]
        heatmap = np.clip(heatmap * -255, 0, 255).astype(np.uint8)
        colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_BONE)
        colored_heatmap = cv2.resize(colored_heatmap, (img.shape[1], img.shape[0]))
        img_heatmap = img * 0.3 + colored_heatmap * 0.7
        cv2.imwrite(os.path.join(dir_out, '{}{}_img_heatmap.png'.format(name, k)), img_heatmap)


def save_inv_result_images(dir_out, img, pose, vis, name='', heatmaps=None, label=None):
    if not os.path.isdir(dir_out):
        os.makedirs(dir_out)

    img = (img - img.min()) / (img.max() - img.min()) * 255
    img = img[:,:,::-1]
    cv2.imwrite(os.path.join(dir_out, '{}img.png'.format(name)), img)
    img_pose = img.copy()
    pose = pose * [img.shape[0] / heatmaps.shape[1], img.shape[1] / heatmaps.shape[2]]
    for i in range(len(PoseTrack_Keypoint_Pairs)):
        c, p = PoseTrack_Keypoint_Pairs[i][0], PoseTrack_Keypoint_Pairs[i][1]
        if vis[c] < 0.1 or vis[p] < 0.1:
            continue
        child = tuple(pose[c].astype(int))
        parent = tuple(pose[p].astype(int))
        color = COLOR_DICT[PoseTrack_Keypoint_Pairs[i][2]]
        cv2.line(img_pose, child, parent, color, 4)

    if label is not None:
        for i in range(len(pose)):
            coord = pose[i]
            cv2.circle(img_pose, coord.astype(int), 3, (0, 255 * label[i] if label is not None else 0, 255), -1)
    cv2.imwrite(os.path.join(dir_out, '{}img_pose.png'.format(name)), img_pose)

    if heatmaps is not None:
        heatmap = np.sum(heatmaps, axis=0)
        heatmap = np.clip(heatmap * -255, 0, 255).astype(np.uint8)
        colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_BONE)
        colored_heatmap = cv2.resize(colored_heatmap, (img.shape[1], img.shape[0]))
        cv2.imwrite(os.path.join(dir_out, '{}heatmap.png'.format(name)), colored_heatmap)
        img_heatmap = img_pose * 0.7 + colored_heatmap * 0.3
        cv2.imwrite(os.path.join(dir_out, '{}img_heatmap.png'.format(name)), img_heatmap)

def save_result_images(dir_out, img, pose, vis, name='', heatmaps=None, label=None):
    if not os.path.isdir(dir_out):
        os.makedirs(dir_out)

    img = (img - img.min()) / (img.max() - img.min()) * 255
    img = img[:,:,::-1]
    cv2.imwrite(os.path.join(dir_out, '{}img.png'.format(name)), img)
    img_pose = img.copy()
    pose = pose * [img.shape[0] / heatmaps.shape[1], img.shape[1] / heatmaps.shape[2]]
    for i in range(len(PoseTrack_Keypoint_Pairs)):
        c, p = PoseTrack_Keypoint_Pairs[i][0], PoseTrack_Keypoint_Pairs[i][1]
        if vis[c] < 0.1 or vis[p] < 0.1:
            continue
        child = tuple(pose[c].astype(int))
        parent = tuple(pose[p].astype(int))
        color = COLOR_DICT[PoseTrack_Keypoint_Pairs[i][2]]
        cv2.line(img_pose, child, parent, color, 4)

    if label is not None:
        for i in range(len(pose)):
            coord = pose[i]
            cv2.circle(img_pose, coord.astype(int), 3, (0, 255 * label[i] if label is not None else 0, 255), -1)
    cv2.imwrite(os.path.join(dir_out, '{}img_pose.png'.format(name)), img_pose)

    if heatmaps is not None:
        heatmap = np.sum(heatmaps, axis=0)
        heatmap = np.clip(heatmap * 255, 0, 255).astype(np.uint8)
        colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_BONE)
        colored_heatmap = cv2.resize(colored_heatmap, (img.shape[1], img.shape[0]))
        cv2.imwrite(os.path.join(dir_out, '{}heatmap.png'.format(name)), colored_heatmap)
        img_heatmap = img_pose * 0.7 + colored_heatmap * 0.3
        cv2.imwrite(os.path.join(dir_out, '{}img_heatmap.png'.format(name)), img_heatmap)


def save_seg_images(dir_out, img, heatmaps=None, name=''):
    img = (img - img.min()) / (img.max() - img.min()) * 255
    img = img[:,:,::-1]

    if heatmaps is not None:
        heatmaps = np.asarray(heatmaps, dtype=np.uint8)
        cv2.imwrite(os.path.join(dir_out, '{}.png'.format(name)), heatmaps)
        img_heatmap = img * 0.7 + heatmaps * 0.3
        cv2.imwrite(os.path.join(dir_out, '{}img.png'.format(name)), img_heatmap)


def calc_dists(preds, target, normalize):
    preds = preds.astype(np.float32)
    target = target.astype(np.float32)
    dists = np.zeros((preds.shape[1], preds.shape[0]))
    for n in range(preds.shape[0]):  # batch num
        for c in range(preds.shape[1]):  # keypoint type
            if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                normed_preds = preds[n, c, :] / normalize[n]
                normed_targets = target[n, c, :] / normalize[n]
                dists[c, n] = np.linalg.norm(normed_preds - normed_targets)  # Euclidean distance
            else:
                dists[c, n] = -1
    return dists


def dist_acc(dists, thr=0.5, percentage=True):
    """ Return percentage below threshold while ignoring values with a -1 """
    dist_cal = np.not_equal(dists, -1)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        less_thr_count = np.less(dists[dist_cal], thr).sum() * 1.0
        if percentage:
            return less_thr_count / num_dist_cal
        else:
            return less_thr_count, num_dist_cal  # less_thr_count = match  / num_dist_cal （val）
    else:
        if percentage:
            return -1
        else:
            return -1, -1


def accuracy(output, target, hm_type='gaussian', thr=0.5):
    """
    Calculate accuracy according to PCK (),
    but uses ground truth heatmap rather than x,y locations
    First value to be returned is average accuracy across 'idxs',
    followed by individual accuracies
    """
    idx = list(range(output.shape[1]))
    norm = 1.0
    if hm_type == 'gaussian':
        pred, _ = get_max_preds(output)
        target, _ = get_max_preds(target)
        h = output.shape[2]
        w = output.shape[3]
        norm = np.ones((pred.shape[0], 2)) * np.array([h, w]) / 10
    dists = calc_dists(pred, target, norm)  # use a fixed length as a measure rather than the length of body parts

    acc = np.zeros((len(idx) + 1))
    avg_acc = 0
    cnt = 0

    for i in range(len(idx)):
        acc[i + 1] = dist_acc(dists[idx[i]], thr)
        if acc[i + 1] >= 0:
            avg_acc = avg_acc + acc[i + 1]
            cnt += 1

    avg_acc = avg_acc / cnt if cnt != 0 else 0
    if cnt != 0:
        acc[0] = avg_acc

    return acc, avg_acc, cnt, pred


def pckh5(output, target, hm_type='gaussian', thr=0.5):
    """
    Calculate accuracy according to PCK (),
    but uses ground truth heatmap rather than x,y locations
    First value to be returned is average accuracy across 'idxs',
    followed by individual accuracies
    """
    idx = list(range(output.shape[1]))
    norm = 1.0
    if hm_type == 'gaussian':
        pred, _ = get_max_preds(output)
        target, _ = get_max_preds(target)
        h = output.shape[2]
        w = output.shape[3]
        norm = np.ones((pred.shape[0], 2)) * np.array([h, w]) / 10
    dists = calc_dists(pred, target, norm)  # use a fixed length as a measure rather than the length of body parts

    acc = np.zeros((len(idx) + 1))
    avg_acc = 0
    cnt = 0

    for i in range(len(idx)):
        acc[i + 1] = dist_acc(dists[idx[i]], thr)
        if acc[i + 1] >= 0:
            avg_acc = avg_acc + acc[i + 1]
            cnt += 1

    avg_acc = avg_acc / cnt if cnt != 0 else 0
    if cnt != 0:
        acc[0] = avg_acc

    return acc, avg_acc, cnt, pred


def get_point_gt_by_id(points, pidx):
    point = []
    for i in range(len(points)):
        if points[i]["id"] != None and points[i]["id"][0] == pidx:  # if joint id matches
            point = points[i]
            break

    return point


def get_head_size(x1, y1, x2, y2):
    headSize = 0.6 * np.linalg.norm(np.subtract([x2, y2], [x1, y1]))
    return headSize


def assignGTmulti(gtFrames, prFrames, distThresh):
    assert (len(gtFrames) == len(prFrames))

    nJoints = Joint().count
    # part detection scores
    scoresAll = {}
    # positive / negative labels
    labelsAll = {}
    # number of annotated GT joints per image
    nGTall = np.zeros([nJoints, len(gtFrames)])
    for pidx in range(nJoints):
        scoresAll[pidx] = {}
        labelsAll[pidx] = {}
        for imgidx in range(len(gtFrames)):
            scoresAll[pidx][imgidx] = np.zeros([0, 0], dtype=np.float32)
            labelsAll[pidx][imgidx] = np.zeros([0, 0], dtype=np.int8)

    # GT track IDs
    trackidxGT = []

    # prediction track IDs
    trackidxPr = []

    # number of GT poses
    nGTPeople = np.zeros((len(gtFrames), 1))
    # number of predicted poses
    nPrPeople = np.zeros((len(gtFrames), 1))

    # container to save info for computing MOT metrics
    motAll = {}

    for imgidx in range(len(gtFrames)):
        # distance between predicted and GT joints
        dist = np.full((len(prFrames[imgidx]["annorect"]), len(gtFrames[imgidx]["annorect"]), nJoints), np.inf)
        # score of the predicted joint
        score = np.full((len(prFrames[imgidx]["annorect"]), nJoints), np.nan)
        # body joint prediction exist
        hasPr = np.zeros((len(prFrames[imgidx]["annorect"]), nJoints), dtype=bool)
        # body joint is annotated
        hasGT = np.zeros((len(gtFrames[imgidx]["annorect"]), nJoints), dtype=bool)

        trackidxGT = []
        trackidxPr = []
        idxsPr = []
        for ridxPr in range(len(prFrames[imgidx]["annorect"])):
            if (("annopoints" in prFrames[imgidx]["annorect"][ridxPr].keys()) and
                    ("point" in prFrames[imgidx]["annorect"][ridxPr]["annopoints"][0].keys())):
                idxsPr += [ridxPr]
        prFrames[imgidx]["annorect"] = [prFrames[imgidx]["annorect"][ridx] for ridx in idxsPr]

        nPrPeople[imgidx, 0] = len(prFrames[imgidx]["annorect"])
        nGTPeople[imgidx, 0] = len(gtFrames[imgidx]["annorect"])
        # iterate over GT poses
        for ridxGT in range(len(gtFrames[imgidx]["annorect"])):
            # GT pose
            rectGT = gtFrames[imgidx]["annorect"][ridxGT]
            if "track_id" in rectGT.keys():
                trackidxGT += [rectGT["track_id"][0]]
            pointsGT = []
            if len(rectGT["annopoints"]) > 0:
                pointsGT = rectGT["annopoints"][0]["point"]
            # iterate over all possible body joints
            for i in range(nJoints):
                # GT joint in LSP format
                ppGT = get_point_gt_by_id(pointsGT, i)
                if len(ppGT) > 0:
                    hasGT[ridxGT, i] = True

        # iterate over predicted poses
        for ridxPr in range(len(prFrames[imgidx]["annorect"])):
            # predicted pose
            rectPr = prFrames[imgidx]["annorect"][ridxPr]
            if "track_id" in rectPr.keys():
                trackidxPr += [rectPr["track_id"][0]]
            pointsPr = rectPr["annopoints"][0]["point"]
            for i in range(nJoints):
                # predicted joint in LSP format
                ppPr = get_point_gt_by_id(pointsPr, i)
                if len(ppPr) > 0:
                    if not ("score" in ppPr.keys()):
                        # use minimum score if predicted score is missing
                        if imgidx == 0:
                            print('WARNING: prediction score is missing. Setting fallback score={}'.format(MIN_SCORE))
                        score[ridxPr, i] = MIN_SCORE
                    else:
                        score[ridxPr, i] = ppPr["score"][0]
                    hasPr[ridxPr, i] = True

        if len(prFrames[imgidx]["annorect"]) and len(gtFrames[imgidx]["annorect"]):
            # predictions and GT are present
            # iterate over GT poses
            for ridxGT in range(len(gtFrames[imgidx]["annorect"])):
                # GT pose
                rectGT = gtFrames[imgidx]["annorect"][ridxGT]
                # compute reference distance as head size
                headSize = get_head_size(rectGT["x1"][0], rectGT["y1"][0],
                                         rectGT["x2"][0], rectGT["y2"][0])
                pointsGT = []
                if len(rectGT["annopoints"]) > 0:
                    pointsGT = rectGT["annopoints"][0]["point"]
                # iterate over predicted poses
                for ridxPr in range(len(prFrames[imgidx]["annorect"])):
                    # predicted pose
                    rectPr = prFrames[imgidx]["annorect"][ridxPr]
                    pointsPr = rectPr["annopoints"][0]["point"]

                    # iterate over all possible body joints
                    for i in range(nJoints):
                        # GT joint
                        ppGT = get_point_gt_by_id(pointsGT, i)
                        # predicted joint
                        ppPr = get_point_gt_by_id(pointsPr, i)
                        # compute distance between predicted and GT joint locations
                        if hasPr[ridxPr, i] and hasGT[ridxGT, i]:
                            pointGT = [ppGT["x"][0], ppGT["y"][0]]
                            pointPr = [ppPr["x"][0], ppPr["y"][0]]
                            dist[ridxPr, ridxGT, i] = np.linalg.norm(np.subtract(pointGT, pointPr)) / headSize

            dist = np.array(dist)
            hasGT = np.array(hasGT)

            # number of annotated joints
            nGTp = np.sum(hasGT, axis=1)
            match = dist <= distThresh
            pck = 1.0 * np.sum(match, axis=2)
            for i in range(hasPr.shape[0]):
                for j in range(hasGT.shape[0]):
                    if nGTp[j] > 0:
                        pck[i, j] = pck[i, j] / nGTp[j]

            # preserve best GT match only
            idx = np.argmax(pck, axis=1)
            val = np.max(pck, axis=1)
            for ridxPr in range(pck.shape[0]):
                for ridxGT in range(pck.shape[1]):
                    if ridxGT != idx[ridxPr]:
                        pck[ridxPr, ridxGT] = 0
            prToGT = np.argmax(pck, axis=0)
            val = np.max(pck, axis=0)
            prToGT[val == 0] = -1

            # info to compute MOT metrics
            mot = {}
            for i in range(nJoints):
                mot[i] = {}

            for i in range(nJoints):
                ridxsGT = np.argwhere(hasGT[:, i] == True)
                ridxsGT = ridxsGT.flatten().tolist()
                ridxsPr = np.argwhere(hasPr[:, i] == True)
                ridxsPr = ridxsPr.flatten().tolist()
                mot[i]["trackidxGT"] = [trackidxGT[idx] for idx in ridxsGT]
                mot[i]["trackidxPr"] = [trackidxPr[idx] for idx in ridxsPr]
                mot[i]["ridxsGT"] = np.array(ridxsGT)
                mot[i]["ridxsPr"] = np.array(ridxsPr)
                mot[i]["dist"] = np.full((len(ridxsGT), len(ridxsPr)), np.nan)
                for iPr in range(len(ridxsPr)):
                    for iGT in range(len(ridxsGT)):
                        if match[ridxsPr[iPr], ridxsGT[iGT], i]:
                            mot[i]["dist"][iGT, iPr] = dist[ridxsPr[iPr], ridxsGT[iGT], i]

            # assign predicted poses to GT poses
            for ridxPr in range(hasPr.shape[0]):
                if ridxPr in prToGT:  # pose matches to GT
                    # GT pose that matches the predicted pose
                    ridxGT = np.argwhere(prToGT == ridxPr)
                    assert (ridxGT.size == 1)
                    ridxGT = ridxGT[0, 0]
                    s = score[ridxPr, :]
                    m = np.squeeze(match[ridxPr, ridxGT, :])
                    hp = hasPr[ridxPr, :]
                    for i in range(len(hp)):
                        if hp[i]:
                            scoresAll[i][imgidx] = np.append(scoresAll[i][imgidx], s[i])
                            labelsAll[i][imgidx] = np.append(labelsAll[i][imgidx], m[i])

                else:  # no matching to GT
                    s = score[ridxPr, :]
                    m = np.zeros([match.shape[2], 1], dtype=bool)
                    hp = hasPr[ridxPr, :]
                    for i in range(len(hp)):
                        if hp[i]:
                            scoresAll[i][imgidx] = np.append(scoresAll[i][imgidx], s[i])
                            labelsAll[i][imgidx] = np.append(labelsAll[i][imgidx], m[i])
        else:
            if not len(gtFrames[imgidx]["annorect"]):
                # No GT available. All predictions are false positives
                for ridxPr in range(hasPr.shape[0]):
                    s = score[ridxPr, :]
                    m = np.zeros([nJoints, 1], dtype=bool)
                    hp = hasPr[ridxPr, :]
                    for i in range(len(hp)):
                        if hp[i]:
                            scoresAll[i][imgidx] = np.append(scoresAll[i][imgidx], s[i])
                            labelsAll[i][imgidx] = np.append(labelsAll[i][imgidx], m[i])
            mot = {}
            for i in range(nJoints):
                mot[i] = {}
            for i in range(nJoints):
                ridxsGT = [0]
                ridxsPr = [0]
                mot[i]["trackidxGT"] = [0]
                mot[i]["trackidxPr"] = [0]
                mot[i]["ridxsGT"] = np.array(ridxsGT)
                mot[i]["ridxsPr"] = np.array(ridxsPr)
                mot[i]["dist"] = np.full((len(ridxsGT), len(ridxsPr)), np.nan)

        # save number of GT joints
        for ridxGT in range(hasGT.shape[0]):
            hg = hasGT[ridxGT, :]
            for i in range(len(hg)):
                nGTall[i, imgidx] += hg[i]

        motAll[imgidx] = mot

    return scoresAll, labelsAll, nGTall, motAll


# compute recall/precision curve (RPC) values
def compute_rpc(scores, labels, totalPos):
    precision = np.zeros(len(scores))
    recall = np.zeros(len(scores))
    npos = 0

    idxsSort = np.array(scores).argsort()[::-1]
    labelsSort = labels[idxsSort]

    for sidx in range(len(idxsSort)):
        if labelsSort[sidx] == 1:
            npos += 1
        # recall: how many true positives were found out of the total number of positives?
        recall[sidx] = 1.0 * npos / totalPos
        # precision: how many true positives were found out of the total number of samples?
        precision[sidx] = 1.0 * npos / (sidx + 1)

    return precision, recall, idxsSort


def compute_metrics(scoresAll, labelsAll, nGTall):
    apAll = np.zeros((nGTall.shape[0] + 1, 1))
    recAll = np.zeros((nGTall.shape[0] + 1, 1))
    preAll = np.zeros((nGTall.shape[0] + 1, 1))
    # iterate over joints
    for j in range(nGTall.shape[0]):
        scores = np.zeros([0, 0], dtype=np.float32)
        labels = np.zeros([0, 0], dtype=np.int8)
        # iterate over images
        for imgidx in range(nGTall.shape[1]):
            scores = np.append(scores, scoresAll[j][imgidx])
            labels = np.append(labels, labelsAll[j][imgidx])
        # compute recall/precision values
        nGT = sum(nGTall[j, :])
        precision, recall, scoresSortedIdxs = compute_rpc(scores, labels, nGT)
        if len(precision) > 0:
            apAll[j] = vocap(recall, precision) * 100
            preAll[j] = precision[len(precision) - 1] * 100
            recAll[j] = recall[len(recall) - 1] * 100
    idxs = np.argwhere(~np.isnan(apAll[:nGTall.shape[0], 0]))
    apAll[nGTall.shape[0]] = apAll[idxs, 0].mean()
    idxs = np.argwhere(~np.isnan(recAll[:nGTall.shape[0], 0]))
    recAll[nGTall.shape[0]] = recAll[idxs, 0].mean()
    idxs = np.argwhere(~np.isnan(preAll[:nGTall.shape[0], 0]))
    preAll[nGTall.shape[0]] = preAll[idxs, 0].mean()

    return apAll, preAll, recAll


# compute Average Precision using recall/precision values
def vocap(rec, prec):
    mpre = np.zeros([1, 2 + len(prec)])
    mpre[0, 1:len(prec) + 1] = prec
    mrec = np.zeros([1, 2 + len(rec)])
    mrec[0, 1:len(rec) + 1] = rec
    mrec[0, len(rec) + 1] = 1.0

    for i in range(mpre.size - 2, -1, -1):
        mpre[0, i] = max(mpre[0, i], mpre[0, i + 1])

    i = np.argwhere(~np.equal(mrec[0, 1:], mrec[0, :mrec.shape[1] - 1])) + 1
    i = i.flatten()

    # compute area under the curve
    ap = np.sum(np.multiply(np.subtract(mrec[0, i], mrec[0, i - 1]), mpre[0, i]))

    return ap


def evaluate_ap(gtFramesAll, prFramesAll):
    distThresh = 0.5

    # assign predicted poses to GT poses
    scoresAll, labelsAll, nGTall, _ = assignGTmulti(gtFramesAll, prFramesAll, distThresh)

    # compute average precision (AP), precision and recall per part
    apAll, preAll, recAll = compute_metrics(scoresAll, labelsAll, nGTall)

    return apAll, preAll, recAll


def convert_data_to_annorect_struct(poses, tracks, boxes, **kwargs):
    """
            Args:
                boxes (np.ndarray): Nx5 size matrix with boxes on this frame
                poses (list of np.ndarray): N length list with each element as 4x17 array
                tracks (list): N length list with track ID for each box/pose
    """
    num_dets = len(poses)
    annorect = []

    eval_tracking = kwargs.get("eval_tracking", False)
    tracking_threshold = kwargs.get("tracking_threshold", 0)
    for j in range(num_dets):
        score = boxes[j][0, 5]
        if eval_tracking and score > tracking_threshold:
            continue

        point = coco2posetrack_ord(poses[j], global_score=score)  # here poses 4*17
        annorect.append({'annopoints': [{'point': point}],
                         'score': [float(score)],
                         'track_id': [tracks[j]]})
    if num_dets == 0:
        annorect.append({
            'annopoints': [{'point': [{
                'id': [0],
                'x': [0],
                'y': [0],
                'score': [-100.0],
            }]}],
            'score': [0],
            'track_id': [0]})
    return annorect


def process_arguments(argv):
    mode = 'multi'

    if len(argv) > 3:
        mode = str.lower(argv[3])
    elif len(argv) < 3 or len(argv) > 4:
        help()

    gt_file = argv[1]
    pred_file = argv[2]

    if not os.path.exists(gt_file):
        help('Given ground truth directory does not exist!\n')

    if not os.path.exists(pred_file):
        help('Given prediction directory does not exist!\n')

    return gt_file, pred_file, mode


def load_data_dir(argv):
    gt_dir, pred_dir, mode = process_arguments(argv)
    if not os.path.exists(gt_dir):
        help('Given GT directory ' + gt_dir + ' does not exist!\n')
    if not os.path.exists(pred_dir):
        help('Given prediction directory ' + pred_dir + ' does not exist!\n')
    filenames = glob.glob(gt_dir + "/*.json")
    gtFramesAll = []
    prFramesAll = []

    for i in range(len(filenames)):
        # load each annotation json file
        # print('GT:')
        with open(filenames[i]) as data_file:
            data = json.load(data_file)
        if not "annolist" in data:
            data = convert_videos(data)[0]
        gt = data["annolist"]
        for imgidx in range(len(gt)):
            gt[imgidx]["seq_id"] = i
            gt[imgidx]["seq_name"] = os.path.basename(filenames[i]).split('.')[0]
            for ridxGT in range(len(gt[imgidx]["annorect"])):
                if "track_id" in gt[imgidx]["annorect"][ridxGT].keys():
                    # adjust track_ids to make them unique across all sequences
                    #                print(os.path.basename(filenames[i]).split('.')[0])
                    #                print(gt[imgidx]["annorect"][ridxGT]["track_id"][0])
                    assert (gt[imgidx]["annorect"][ridxGT]["track_id"][0] < MAX_TRACK_ID)
                    gt[imgidx]["annorect"][ridxGT]["track_id"][0] += i * MAX_TRACK_ID
        gtFramesAll += gt
        gtBasename = os.path.basename(filenames[i])
        # predFilename = pred_dir + gtBasename
        predFilename = os.path.join(pred_dir, gtBasename)
        # print(xy)

        if not os.path.exists(predFilename):
            raise IOError('Prediction file ' + predFilename + ' does not exist')

        # load predictions
        #    print('PRED:')
        with open(predFilename) as data_file:
            data = json.load(data_file)
        if not "annolist" in data:
            data = convert_videos(data)[0]
        pr = data["annolist"]
        if len(pr) != len(gt):
            raise Exception('# prediction frames %d != # GT frames %d for %s' % (len(pr), len(gt), predFilename))
        for imgidx in range(len(pr)):
            for ridxPr in range(len(pr[imgidx]["annorect"])):
                if "track_id" in pr[imgidx]["annorect"][ridxPr].keys():
                    # adjust track_ids to make them unique across all sequences
                    #                print(os.path.basename(filenames[i]).split('.')[0])
                    #                print(pr[imgidx]["annorect"][ridxPr]["track_id"][0])
                    assert (pr[imgidx]["annorect"][ridxPr]["track_id"][0] < MAX_TRACK_ID)
                    pr[imgidx]["annorect"][ridxPr]["track_id"][0] += i * MAX_TRACK_ID
        prFramesAll += pr
    #    print(xy)

    gtFramesAll, prFramesAll = cleanupData(gtFramesAll, prFramesAll)

    gtFramesAll, prFramesAll = removeIgnoredPoints(gtFramesAll, prFramesAll)
    return gtFramesAll, prFramesAll


def evaluate(gtdir, preddir, eval_pose=True, eval_track=True,
             eval_upper_bound=False):
    logger = logging.getLogger(__name__)
    gtFramesAll, prFramesAll = load_data_dir(['', gtdir, preddir])

    logger.info('# gt frames  : {}'.format(str(len(gtFramesAll))))
    logger.info('# pred frames: {}'.format(str(len(prFramesAll))))

    apAll = np.full((Joint().count + 1, 1), np.nan)
    preAll = np.full((Joint().count + 1, 1), np.nan)
    recAll = np.full((Joint().count + 1, 1), np.nan)
    cum = None
    track_cum = None
    if eval_pose:
        apAll, preAll, recAll = evaluate_ap(gtFramesAll, prFramesAll)

    logger.info('Average Precision (AP) metric:')
    # printTable(apAll)
    cum = printTable(apAll)

    metrics = np.full((Joint().count + 4, 1), np.nan)

    if eval_track:
        # print(xy)
        metricsAll = evaluate_tracking(
            gtFramesAll, prFramesAll, eval_upper_bound)

        for i in range(Joint().count + 1):
            metrics[i, 0] = metricsAll['mota'][0, i]
        metrics[Joint().count + 1, 0] = metricsAll['motp'][0, Joint().count]
        metrics[Joint().count + 2, 0] = metricsAll['pre'][0, Joint().count]
        metrics[Joint().count + 3, 0] = metricsAll['rec'][0, Joint().count]
        logger.info('Multiple Object Tracking (MOT) mmetrics:')
        # print('Multiple Object Tracking (MOT) mmetrics:')
        track_cum = printTable(metrics, motHeader=True)
    # return (apAll, preAll, recAll), metrics
    # print(xy)
    return cum, track_cum


def computeMetrics(gtFramesAll, motAll, outputDir, bSaveAll, bSaveSeq):

    assert(len(gtFramesAll) == len(motAll))
    logger = logging.getLogger(__name__)

    nJoints = Joint().count
    seqidxs = []
    for imgidx in range(len(gtFramesAll)):
        seqidxs += [gtFramesAll[imgidx]["seq_id"]]
    seqidxs = np.array(seqidxs)

    seqidxsUniq = np.unique(seqidxs)

    # intermediate metrics
    metricsMidNames = ['num_misses', 'num_switches', 'num_false_positives', 'num_objects', 'num_detections']

    # final metrics computed from intermediate metrics
    metricsFinNames = ['mota','motp','pre','rec']

    # initialize intermediate metrics
    metricsMidAll = {}
    for name in metricsMidNames:
        metricsMidAll[name] = np.zeros([1,nJoints])
    metricsMidAll['sumD'] = np.zeros([1,nJoints])

    # initialize final metrics
    metricsFinAll = {}
    for name in metricsFinNames:
        metricsFinAll[name] = np.zeros([1,nJoints+1])

    # create metrics
    mh = mm.metrics.create()

    imgidxfirst = 0
    # iterate over tracking sequences
    # seqidxsUniq = seqidxsUniq[:20]
    nSeq = len(seqidxsUniq)

    # initialize per-sequence metrics
    metricsSeqAll = {}
    for si in range(nSeq):
        metricsSeqAll[si] = {}
        for name in metricsFinNames:
            metricsSeqAll[si][name] = np.zeros([1,nJoints+1])

    names = Joint().name
    names['15'] = 'total'

    for si in range(nSeq):
    #for si in range(5):
        logger.info("seqidx: %d/%d" % (si+1,nSeq))

        # init per-joint metrics accumulator
        accAll = {}
        for i in range(nJoints):
            accAll[i] = mm.MOTAccumulator(auto_id=True)

        # extract frames IDs for the sequence
        imgidxs = np.argwhere(seqidxs == seqidxsUniq[si])
        imgidxs = imgidxs[:-1].copy()
        seqName = gtFramesAll[imgidxs[0,0]]["seq_name"]
        logger.info(seqName)
        # create an accumulator that will be updated during each frame
        # iterate over frames
        for j in range(len(imgidxs)):
            imgidx = imgidxs[j,0]
            # iterate over joints
            for i in range(nJoints):
                # GT tracking ID
                trackidxGT = motAll[imgidx][i]["trackidxGT"]
                # prediction tracking ID
                trackidxPr = motAll[imgidx][i]["trackidxPr"]
                # distance GT <-> pred part to compute MOT metrics
                # 'NaN' means force no match
                dist = motAll[imgidx][i]["dist"]
                # Call update once per frame
                accAll[i].update(
                    trackidxGT,                 # Ground truth objects in this frame
                    trackidxPr,                 # Detector hypotheses in this frame
                    dist                        # Distances from objects to hypotheses
                )

        # compute intermediate metrics per joint per sequence
        for i in range(nJoints):
            metricsMid = mh.compute(accAll[i], metrics=metricsMidNames, return_dataframe=False, name='acc')
            for name in metricsMidNames:
                metricsMidAll[name][0,i] += metricsMid[name]
            s = accAll[i].events['D'].sum()
            if (np.isnan(s)):
                s = 0
            metricsMidAll['sumD'][0,i] += s

#        if (bSaveSeq):
        if False:
            # compute metrics per joint per sequence
            for i in range(nJoints):
                metricsMid = mh.compute(accAll[i], metrics=metricsMidNames, return_dataframe=False, name='acc')

                # compute final metrics per sequence
                if (metricsMid['num_objects'] > 0):
                    numObj = metricsMid['num_objects']
                else:
                    numObj = np.nan
                numFP = metricsMid['num_false_positives']
                metricsSeqAll[si]['mota'][0,i] = 100*(1. - 1.*(metricsMid['num_misses'] +
                                                    metricsMid['num_switches'] +
                                                    numFP) /
                                                    numObj)
                numDet = metricsMid['num_detections']
                s = accAll[i].events['D'].sum()
                if (numDet == 0 or np.isnan(s)):
                    metricsSeqAll[si]['motp'][0,i] = 0.0
                else:
                    metricsSeqAll[si]['motp'][0,i] = 100*(1. - (1.*s / numDet))
                if (numFP+numDet > 0):
                    totalDet = numFP+numDet
                else:
                    totalDet = np.nan
                metricsSeqAll[si]['pre'][0,i]  = 100*(1.*numDet /
                                                totalDet)
                metricsSeqAll[si]['rec'][0,i]  = 100*(1.*numDet /
                                        numObj)

            # average metrics over all joints per sequence
            idxs = np.argwhere(~np.isnan(metricsSeqAll[si]['mota'][0,:nJoints]))
            metricsSeqAll[si]['mota'][0,nJoints] = metricsSeqAll[si]['mota'][0,idxs].mean()
            idxs = np.argwhere(~np.isnan(metricsSeqAll[si]['motp'][0,:nJoints]))
            metricsSeqAll[si]['motp'][0,nJoints] = metricsSeqAll[si]['motp'][0,idxs].mean()
            idxs = np.argwhere(~np.isnan(metricsSeqAll[si]['pre'][0,:nJoints]))
            metricsSeqAll[si]['pre'][0,nJoints]  = metricsSeqAll[si]['pre'] [0,idxs].mean()
            idxs = np.argwhere(~np.isnan(metricsSeqAll[si]['rec'][0,:nJoints]))
            metricsSeqAll[si]['rec'][0,nJoints]  = metricsSeqAll[si]['rec'] [0,idxs].mean()

            metricsSeq = metricsSeqAll[si].copy()
            metricsSeq['mota'] = metricsSeq['mota'].flatten().tolist()
            metricsSeq['motp'] = metricsSeq['motp'].flatten().tolist()
            metricsSeq['pre'] = metricsSeq['pre'].flatten().tolist()
            metricsSeq['rec'] = metricsSeq['rec'].flatten().tolist()
            metricsSeq['names'] = names

            filename = outputDir + '/' + seqName + '_MOT_metrics.json'
            logger.info('saving results to', filename)
            #eval_helpers.writeJson(metricsSeq,filename)
            writeJson(metricsSeq,filename)

    # compute final metrics per joint for all sequences
    for i in range(nJoints):
        if (metricsMidAll['num_objects'][0,i] > 0):
            numObj = metricsMidAll['num_objects'][0,i]
        else:
            numObj = np.nan
        numFP = metricsMidAll['num_false_positives'][0,i]
        metricsFinAll['mota'][0,i] = 100*(1. - (metricsMidAll['num_misses'][0,i] +
                                                metricsMidAll['num_switches'][0,i] +
                                                numFP) /
                                                numObj)
        numDet = metricsMidAll['num_detections'][0,i]
        s = metricsMidAll['sumD'][0,i]
        if (numDet == 0 or np.isnan(s)):
            metricsFinAll['motp'][0,i] = 0.0
        else:
            metricsFinAll['motp'][0,i] = 100*(1. - (s / numDet))
        if (numFP+numDet > 0):
            totalDet = numFP+numDet
        else:
            totalDet = np.nan

        metricsFinAll['pre'][0,i]  = 100*(1.*numDet /
                                       totalDet)
        metricsFinAll['rec'][0,i]  = 100*(1.*numDet /
                                       numObj)

    # average metrics over all joints over all sequences
    idxs = np.argwhere(~np.isnan(metricsFinAll['mota'][0,:nJoints]))
    metricsFinAll['mota'][0,nJoints] = metricsFinAll['mota'][0,idxs].mean()
    idxs = np.argwhere(~np.isnan(metricsFinAll['motp'][0,:nJoints]))
    metricsFinAll['motp'][0,nJoints] = metricsFinAll['motp'][0,idxs].mean()
    idxs = np.argwhere(~np.isnan(metricsFinAll['pre'][0,:nJoints]))
    metricsFinAll['pre'][0,nJoints]  = metricsFinAll['pre'] [0,idxs].mean()
    idxs = np.argwhere(~np.isnan(metricsFinAll['rec'][0,:nJoints]))
    metricsFinAll['rec'][0,nJoints]  = metricsFinAll['rec'] [0,idxs].mean()

#    if (bSaveAll):
    if False:
        metricsFin = metricsFinAll.copy()
        metricsFin['mota'] = metricsFin['mota'].flatten().tolist()
        metricsFin['motp'] = metricsFin['motp'].flatten().tolist()
        metricsFin['pre'] = metricsFin['pre'].flatten().tolist()
        metricsFin['rec'] = metricsFin['rec'].flatten().tolist()
        metricsFin['names'] = names

        filename = outputDir + '/total_MOT_metrics.json'
        logger.info('saving results to', filename)
#        eval_helpers.writeJson(metricsFin,filename)
        writeJson(metricsFin,filename)

    return metricsFinAll