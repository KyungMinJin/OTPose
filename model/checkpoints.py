import os

import torch
import os.path as osp

def resume(model, optimizer, checkpoint_file, **kwargs):
    ext_dict = {}
    checkpoint = torch.load(checkpoint_file)
    begin_epoch = checkpoint['begin_epoch'] + 1
    gpus = kwargs.get("gpus", [])
    # if len(gpus) <= 1:
    #     state_dict = {k.replace('module.', '') if k.index('module') == 0 else k: v for k, v in checkpoint['state_dict'].items()}
    # else:
    state_dict = checkpoint["state_dict"]
    model.load_state_dict(state_dict)
    optimizer.load_state_dict(checkpoint['optimizer'])

    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()

    ext_dict["tensorboard_global_steps"] = checkpoint.get("tensorboard_global_steps", 0)

    return model, optimizer, begin_epoch, ext_dict


def save_checkpoint(epoch, save_folder, model, optimizer, **kwargs):
    model_save_path = osp.join(save_folder, 'epoch_{}_state.pth'.format(epoch))
    checkpoint_dict = dict()
    checkpoint_dict['begin_epoch'] = epoch

    # Because nn.DataParallel
    # https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/5
    model_state_dict = model.state_dict()
    if list(model_state_dict.keys())[0].startswith('module.'):
        model_state_dict = {k[7:]: v for k, v in model_state_dict.items()}

    checkpoint_dict['state_dict'] = model_state_dict
    checkpoint_dict['optimizer'] = optimizer.state_dict()
    checkpoint_dict['tensorboard_global_steps'] = kwargs.get("global_steps", 0)
    torch.save(checkpoint_dict, model_save_path)

    return model_save_path


def save_best_checkpoint(epoch, save_folder, model, optimizer, mAP, **kwargs):
    model_save_path = osp.join(save_folder, 'best_mAP_{}_state.pth'.format(mAP))
    checkpoint_dict = dict()
    checkpoint_dict['begin_epoch'] = epoch

    checkpoint_saves_paths = [x for x in save_folder if 'best' in x]
    if len(checkpoint_saves_paths) > 0:
        latest_checkpoint = checkpoint_saves_paths[0]

        best_mAP = float(osp.basename(latest_checkpoint).split("_")[2])
        for checkpoint_save_path in checkpoint_saves_paths:
            if mAP > best_mAP:
                os.remove(latest_checkpoint)
                latest_checkpoint = checkpoint_save_path
                best_mAP = mAP

    # Because nn.DataParallel
    # https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/5
    model_state_dict = model.state_dict()
    if list(model_state_dict.keys())[0].startswith('module.'):
        model_state_dict = {k[7:]: v for k, v in model_state_dict.items()}

    checkpoint_dict['state_dict'] = model_state_dict
    checkpoint_dict['optimizer'] = optimizer.state_dict()
    checkpoint_dict['tensorboard_global_steps'] = kwargs.get("global_steps", 0)
    torch.save(checkpoint_dict, model_save_path)

    return model_save_path
