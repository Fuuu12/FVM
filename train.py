import timeit
from datetime import datetime
import socket
import os
import glob
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

from dataloaders.dataset import VideoDataset
from network import C3D_model, R2Plus1D_model, R3D_model

from timm.models import create_model
import json
from collections import OrderedDict
import utils
from models import *

from timm.optim.adafactor import Adafactor
from timm.optim.adahessian import Adahessian
from timm.optim.adamp import AdamP
from timm.optim.lookahead import Lookahead
from timm.optim.nadam import Nadam
# from timm.optim.novograd import NovoGrad
from timm.optim.nvnovograd import NvNovoGrad
from timm.optim.radam import RAdam
from timm.optim.rmsprop_tf import RMSpropTF
from timm.optim.sgdp import SGDP

import json

try:
    from apex.optimizers import FusedNovoGrad, FusedAdam, FusedLAMB, FusedSGD
    has_apex = True
except ImportError:
    has_apex = False

# Use GPU if available else revert to CPU
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)

nEpochs = 100  # Number of epochs for training
resume_epoch = 0  # Default is 0, change if want to resume
useTest = True # See evolution of the test set when training
nTestInterval = 20 # Run on test set every nTestInterval epochs
snapshot = 50 # Store a model every snapshot epochs

lr = 1e-5 # Learning rate

dataset = 'ucf101' # Options: hmdb51 or ucf101

if dataset == 'hmdb51':
    num_classes=51
elif dataset == 'ucf101':
    num_classes = 101
else:
    print('We only implemented hmdb and ucf datasets.')
    raise NotImplementedError

save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]

if resume_epoch != 0:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) if runs else 0
else:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0

save_dir = os.path.join(save_dir_root, 'run', 'run_' + str(run_id))
# modelName = 'C3D' # Options: C3D or R2Plus1D or R3D
modelName = 'videomamba_middle'
saveName = modelName + '-' + dataset

# add args
finetune = '/media/user/新加卷/fuyu/videoMamba-ucf101/pretrainckp/videomamba_t16_k400_f16_res224.pth'
num_frames = 16
tubelet_size = 1
input_size = 224
short_side_size = 224
save_ckpt_freq = 20
argsmodel_key = 'model|module'
delete_head = True
argsorig_t_size = 8
model_prefix = ''
weight_decay = 0.05
skip_list = None
get_num_layer = None
get_layer_scale = None
opt_eps = 1e-8
opt_betas = (0.9, 0.999)
momentum = 0.9

def get_parameter_groups(
        model, weight_decay=1e-5, skip_list=(), get_num_layer=None, 
        get_layer_scale=None,
    ):
    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if get_num_layer is not None:
            layer_id = get_num_layer(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if get_layer_scale is not None:
                scale = get_layer_scale(layer_id)
            else:
                scale = 1.

            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())


def train_model(dataset=dataset, save_dir=save_dir, num_classes=num_classes, lr=lr,
                num_epochs=nEpochs, save_epoch=snapshot, useTest=useTest, test_interval=nTestInterval):
    """
        Args:
            num_classes (int): Number of classes in the data
            num_epochs (int, optional): Number of epochs to train for.
    """

    if modelName == 'C3D':
        model = C3D_model.C3D(num_classes=num_classes, pretrained=True)
        train_params = [{'params': C3D_model.get_1x_lr_params(model), 'lr': lr},
                        {'params': C3D_model.get_10x_lr_params(model), 'lr': lr * 10}]
    elif modelName == 'R2Plus1D':
        model = R2Plus1D_model.R2Plus1DClassifier(num_classes=num_classes, layer_sizes=(2, 2, 2, 2))
        train_params = [{'params': R2Plus1D_model.get_1x_lr_params(model), 'lr': lr},
                        {'params': R2Plus1D_model.get_10x_lr_params(model), 'lr': lr * 10}]
    elif modelName == 'R3D':
        model = R3D_model.R3DClassifier(num_classes=num_classes, layer_sizes=(2, 2, 2, 2))
        train_params = model.parameters()
    elif 'videomamba' in modelName:
        # model = create_model(
        #     modelName,
        #     pretrained=False if finetune else True,
        #     num_classes=num_classes,
        #     fc_drop_rate=0.0,
        #     drop_path_rate=0.1,
        #     kernel_size=1,
        #     num_frames=8,
        #     use_checkpoint=False,
        #     checkpoint_num=0,
        # )
        # if finetune:
        #     checkpoint = torch.load(finetune, map_location='cpu')
        #     model.load_state_dict(checkpoint['model_state'], strict=False)
        pass
    else:
        print('We only implemented C3D and R2Plus1D models.')
        raise NotImplementedError

    print("trying to load vim model")
    if 'videomamba' in modelName:
        model = create_model(
            modelName,
            pretrained=False if finetune else True,
            num_classes=num_classes,
            fc_drop_rate=0.0,
            drop_path_rate=0.1,
            kernel_size=1,
            num_frames=16,
            use_checkpoint=False,
            checkpoint_num=0,
        )

    print("loaded vim model")

    patch_size = model.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    window_size = (num_frames // tubelet_size, input_size // patch_size[0], input_size // patch_size[1])

    if finetune:
        if finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(finetune, map_location='cpu')

        print("Load ckpt from %s" % finetune)
        checkpoint_model = None
        for model_key in argsmodel_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint

        if 'head.weight' in checkpoint_model.keys():
            if delete_head:
                print("Removing head from pretrained checkpoint")
                del checkpoint_model['head.weight']
                del checkpoint_model['head.bias']
            elif checkpoint_model['head.weight'].shape[0] == 710:
                if num_classes == 400:
                    checkpoint_model['head.weight'] = checkpoint_model['head.weight'][:num_classes]
                    checkpoint_model['head.bias'] = checkpoint_model['head.bias'][:num_classes]
                elif num_classes in [600, 700]:
                    # download from https://drive.google.com/drive/folders/17cJd2qopv-pEG8NSghPFjZo1UUZ6NLVm
                    map_path = f'k710/label_mixto{num_classes}.json'
                    print(f'Load label map from {map_path}')
                    with open(map_path) as f:
                        label_map = json.load(f)
                    checkpoint_model['head.weight'] = checkpoint_model['head.weight'][label_map]
                    checkpoint_model['head.bias'] = checkpoint_model['head.bias'][label_map]
                    
        all_keys = list(checkpoint_model.keys())
        new_dict = OrderedDict()
        for key in all_keys:
            if key.startswith('backbone.'):
                new_dict[key[9:]] = checkpoint_model[key]
            elif key.startswith('encoder.'):
                new_dict[key[8:]] = checkpoint_model[key]
            else:
                new_dict[key] = checkpoint_model[key]
        checkpoint_model = new_dict

        # interpolate position embedding
        if 'deit' in modelName or 'videomamba' in modelName:
            pos_embed_checkpoint = checkpoint_model['pos_embed']
            embedding_size = pos_embed_checkpoint.shape[-1] # channel dim
            num_patches = model.patch_embed.num_patches # 
            num_extra_tokens = model.pos_embed.shape[-2] - num_patches # 0/1
            orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
            # height (== width) for the new position embedding
            new_size = int(num_patches ** 0.5)

            if orig_size != new_size:
                print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
                extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                # only the position tokens are interpolated
                pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                # B, L, C -> B, H, W, C -> B, C, H, W
                pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                # B, C, H, W -> B, H, W, C ->  B, H, W, C
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(-1, new_size, new_size, embedding_size) 
                pos_tokens = pos_tokens.flatten(1, 2) # B, L, C
                new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                checkpoint_model['pos_embed'] = new_pos_embed
            
            # we use 8 frames for pretraining
            temporal_pos_embed = checkpoint_model['temporal_pos_embedding']
            orig_t_size = argsorig_t_size // model.patch_embed.tubelet_size
            new_t_size = num_frames // model.patch_embed.tubelet_size
            # height (== width) for the checkpoint position embedding
            if orig_t_size != new_t_size:
                print(f"Temporal interpolate from {orig_t_size} to {new_t_size}")
                temporal_pos_embed = temporal_pos_embed.permute(0, 2, 1)
                temporal_pos_embed = torch.nn.functional.interpolate(
                    temporal_pos_embed, size=(new_t_size,), mode='linear', align_corners=False
                )
                temporal_pos_embed = temporal_pos_embed.permute(0, 2, 1)
                checkpoint_model['temporal_pos_embedding'] = temporal_pos_embed

        elif 'pos_embed' in checkpoint_model:
            pos_embed_checkpoint = checkpoint_model['pos_embed']
            embedding_size = pos_embed_checkpoint.shape[-1] # channel dim
            num_patches = model.patch_embed.num_patches # 
            num_extra_tokens = model.pos_embed.shape[-2] - num_patches # 0/1

            # we use 8 frames for pretraining
            orig_t_size = orig_t_size // model.patch_embed.tubelet_size
            new_t_size = num_frames // model.patch_embed.tubelet_size
            # height (== width) for the checkpoint position embedding
            orig_size = int(((pos_embed_checkpoint.shape[-2] - num_extra_tokens)//(orig_t_size)) ** 0.5)
            # height (== width) for the new position embedding
            new_size = int((num_patches // new_t_size) ** 0.5)
            
            if orig_t_size != new_t_size:
                print(f"Temporal interpolate from {orig_t_size} to {new_t_size}")
                tmp_pos_embed = pos_embed_checkpoint.view(1, orig_t_size, -1, embedding_size)
                tmp_pos_embed = tmp_pos_embed.permute(0, 2, 3, 1).reshape(-1, embedding_size, orig_t_size)
                tmp_pos_embed = torch.nn.functional.interpolate(tmp_pos_embed, size=new_t_size, mode='linear')
                tmp_pos_embed = tmp_pos_embed.view(1, -1, embedding_size, new_t_size)
                tmp_pos_embed = tmp_pos_embed.permute(0, 3, 1, 2).reshape(1, -1, embedding_size)
                checkpoint_model['pos_embed'] = tmp_pos_embed
                pos_embed_checkpoint = tmp_pos_embed

            # class_token and dist_token are kept unchanged
            if orig_size != new_size:
                print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
                extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                # only the position tokens are interpolated
                pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                # B, L, C -> BT, H, W, C -> BT, C, H, W
                pos_tokens = pos_tokens.reshape(-1, new_t_size, orig_size, orig_size, embedding_size)
                pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(-1, new_t_size, new_size, new_size, embedding_size) 
                pos_tokens = pos_tokens.flatten(1, 3) # B, L, C
                new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                checkpoint_model['pos_embed'] = new_pos_embed
        
        utils.load_state_dict(model, checkpoint_model, prefix=model_prefix)
    
    opt_weight_decay = 0.05
    if weight_decay > 0:
        skip = {}
        if skip_list is not None:
            skip = skip_list
        elif hasattr(model, 'no_weight_decay'):
            skip = model.no_weight_decay()
        parameters = get_parameter_groups(
            model, weight_decay, skip, get_num_layer, get_layer_scale,
        )
        opt_weight_decay = 0.0
    else:
        parameters = model.parameters()
    
    opt_args = dict(lr=lr, weight_decay=opt_weight_decay)
    if opt_eps is not None:
        opt_args['eps'] = opt_eps
    if opt_betas is not None:
        opt_args['betas'] = opt_betas

    opt_lower = 'adamw'.lower()
    opt_split = opt_lower.split('_')
    opt_lower = opt_split[-1]
    if opt_lower == 'sgd' or opt_lower == 'nesterov':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=momentum, nesterov=True, **opt_args)
    elif opt_lower == 'momentum':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=momentum, nesterov=False, **opt_args)
    elif opt_lower == 'adam':
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == 'nadam':
        optimizer = Nadam(parameters, **opt_args)
    elif opt_lower == 'radam':
        optimizer = RAdam(parameters, **opt_args)
    elif opt_lower == 'adamp':
        optimizer = AdamP(parameters, wd_ratio=0.01, nesterov=True, **opt_args)
    elif opt_lower == 'sgdp':
        optimizer = SGDP(parameters, momentum=momentum, nesterov=True, **opt_args)
    elif opt_lower == 'adadelta':
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == 'adafactor':
        if not lr:
            opt_args['lr'] = None
        optimizer = Adafactor(parameters, **opt_args)
    elif opt_lower == 'adahessian':
        optimizer = Adahessian(parameters, **opt_args)
    elif opt_lower == 'rmsprop':
        optimizer = optim.RMSprop(parameters, alpha=0.9, momentum=momentum, **opt_args)
    elif opt_lower == 'rmsproptf':
        optimizer = RMSpropTF(parameters, alpha=0.9, momentum=momentum, **opt_args)
    # elif opt_lower == 'novograd':
    #     optimizer = NovoGrad(parameters, **opt_args)
    elif opt_lower == 'nvnovograd':
        optimizer = NvNovoGrad(parameters, **opt_args)
    elif opt_lower == 'fusedsgd':
        opt_args.pop('eps', None)
        optimizer = FusedSGD(parameters, momentum=momentum, nesterov=True, **opt_args)
    elif opt_lower == 'fusedmomentum':
        opt_args.pop('eps', None)
        optimizer = FusedSGD(parameters, momentum=momentum, nesterov=False, **opt_args)
    elif opt_lower == 'fusedadam':
        optimizer = FusedAdam(parameters, adam_w_mode=False, **opt_args)
    elif opt_lower == 'fusedadamw':
        optimizer = FusedAdam(parameters, adam_w_mode=True, **opt_args)
    elif opt_lower == 'fusedlamb':
        optimizer = FusedLAMB(parameters, **opt_args)
    elif opt_lower == 'fusednovograd':
        opt_args.setdefault('betas', (0.95, 0.98))
        optimizer = FusedNovoGrad(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"

    criterion = nn.CrossEntropyLoss()  # standard crossentropy loss for classification
    # optimizer = optim.SGD(train_params, lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20,
                                          gamma=0.1)  # the scheduler divides the lr by 10 every 10 epochs

    # if resume_epoch == 0:
    #     print("Training {} from scratch...".format(modelName))
    # else:
    #     checkpoint = torch.load(os.path.join(save_dir, 'models', saveName + '_epoch-' + str(resume_epoch - 1) + '.pth.tar'),
    #                    map_location=lambda storage, loc: storage)   # Load all tensors onto the CPU
    #     print("Initializing weights from: {}...".format(
    #         os.path.join(save_dir, 'models', saveName + '_epoch-' + str(resume_epoch - 1) + '.pth.tar')))
    #     model.load_state_dict(checkpoint['state_dict'])
    #     optimizer.load_state_dict(checkpoint['opt_dict'])

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)

    model.to(device)
    criterion.to(device)

    log_dir = os.path.join(save_dir, 'models', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir)

    print('Training model on {} dataset...'.format(dataset))
    print(orig_size, new_size, num_frames, argsorig_t_size)
    train_dataloader = DataLoader(VideoDataset(dataset=dataset, split='train',clip_len=16), batch_size=6, shuffle=True, num_workers=4)
    val_dataloader   = DataLoader(VideoDataset(dataset=dataset, split='val',  clip_len=16), batch_size=6, num_workers=4)
    test_dataloader  = DataLoader(VideoDataset(dataset=dataset, split='test', clip_len=16), batch_size=6, num_workers=4)

    trainval_loaders = {'train': train_dataloader, 'val': val_dataloader}
    trainval_sizes = {x: len(trainval_loaders[x].dataset) for x in ['train', 'val']}
    test_size = len(test_dataloader.dataset)

    for epoch in range(resume_epoch, num_epochs):
        # each epoch has a training and validation step
        for phase in ['train', 'val']:
            start_time = timeit.default_timer()

            # reset the running loss and corrects
            running_loss = 0.0
            running_corrects = 0.0

            # set model to train() or eval() mode depending on whether it is trained
            # or being validated. Primarily affects layers such as BatchNorm or Dropout.
            if phase == 'train':
                # scheduler.step() is to be called once every epoch during training
                scheduler.step()
                model.train()
            else:
                model.eval()

            for batch_idx, (inputs, labels) in enumerate(tqdm(trainval_loaders[phase])):
                # move inputs and labels to the device the training is taking place on
                inputs = Variable(inputs, requires_grad=True).to(device)
                labels = Variable(labels).to(device)
                optimizer.zero_grad()

                if phase == 'train':
                    outputs = model(inputs)
                else:
                    with torch.no_grad():
                        outputs = model(inputs)

                probs = nn.Softmax(dim=1)(outputs)
                preds = torch.max(probs, 1)[1]
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                # 每 5 个 batch 打印一次 loss
                if batch_idx % 5 == 0:
                    print(f"[{phase}] Epoch {epoch+1}/{num_epochs} | Batch {batch_idx+1} | Loss = {loss.item():.6f}")

            epoch_loss = running_loss / trainval_sizes[phase]
            epoch_acc = running_corrects.double() / trainval_sizes[phase]

            if phase == 'train':
                writer.add_scalar('data/train_loss_epoch', epoch_loss, epoch)
                writer.add_scalar('data/train_acc_epoch', epoch_acc, epoch)
            else:
                writer.add_scalar('data/val_loss_epoch', epoch_loss, epoch)
                writer.add_scalar('data/val_acc_epoch', epoch_acc, epoch)

            print("[{}] Epoch: {}/{} Loss: {} Acc: {}".format(phase, epoch+1, nEpochs, epoch_loss, epoch_acc))
            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time) + "\n")


        if epoch % save_epoch == (save_epoch - 1):
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'opt_dict': optimizer.state_dict(),
            }, os.path.join(save_dir, 'models', saveName + '_epoch-' + str(epoch) + '.pth.tar'))
            print("Save model at {}\n".format(os.path.join(save_dir, 'models', saveName + '_epoch-' + str(epoch) + '.pth.tar')))

        if useTest and epoch % test_interval == (test_interval - 1):
            model.eval()
            start_time = timeit.default_timer()

            running_loss = 0.0
            running_corrects = 0.0

            for inputs, labels in tqdm(test_dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.no_grad():
                    outputs = model(inputs)
                probs = nn.Softmax(dim=1)(outputs)
                preds = torch.max(probs, 1)[1]
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / test_size
            epoch_acc = running_corrects.double() / test_size

            writer.add_scalar('data/test_loss_epoch', epoch_loss, epoch)
            writer.add_scalar('data/test_acc_epoch', epoch_acc, epoch)

            print("[test] Epoch: {}/{} Loss: {} Acc: {}".format(epoch+1, nEpochs, epoch_loss, epoch_acc))
            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time) + "\n")

    writer.close()


if __name__ == "__main__":
    train_model()