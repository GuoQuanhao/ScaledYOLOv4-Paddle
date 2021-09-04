import argparse
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import paddle
import paddle.nn.functional as F
import paddle.optimizer as optim
import paddle.optimizer.lr as lr_scheduler
import yaml
from paddle import amp
from visualdl import LogWriter
from tqdm import tqdm

import test  # import test.py to get mAP after each epoch
from models.yolo import Model
from utils.datasets import create_dataloader
from utils.general import (
    check_img_size, paddle_distributed_zero_first, labels_to_class_weights, plot_labels, check_anchors,
    labels_to_image_weights, compute_loss, plot_images, fitness, strip_optimizer, plot_results,
    get_latest_run, check_git_status, check_file, increment_dir, print_mutation, plot_evolution)
from utils.paddle_utils import init_seeds, ModelEMA, intersect_dicts
import paddle.distributed as dist


def train(opt):
    dist.init_parallel_env()
    
    ###################################################################################################
    # Resume
    if opt.resume:
        last = get_latest_run() if opt.resume == 'get_last' else opt.resume  # resume from most recent run
        if last and not opt.weights:
            print(f'Resuming training from {last}')
        opt.weights = last if opt.resume and not opt.weights else opt.weights
    if opt.local_rank == -1 or ("RANK" in os.environ and os.environ["RANK"] == "0"):
        check_git_status()

    opt.hyp = opt.hyp or ('data/hyp.finetune.yaml' if opt.weights else 'data/hyp.scratch.yaml')
    opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
    assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
    opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
    opt.total_batch_size = opt.batch_size
    print(opt)
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)  # load hyps

    tb_writer = None
    print('Start Tensorboard with "tensorboard --logdir %s", view at http://localhost:6006/' % opt.logdir)
    tb_writer = LogWriter(logdir=increment_dir(Path(opt.logdir) / 'exp', opt.name))  # runs/exp
    ###################################################################################################
    
    
    
    print(f'Hyperparameters {hyp}')
    logdir = Path(tb_writer.logdir) if tb_writer else Path(opt.logdir) / 'evolve'  # logging directory
    wdir = str(logdir / 'weights') + os.sep  # weights directory
    os.makedirs(wdir, exist_ok=True)
    last = wdir + 'last.pdparams'
    best = wdir + 'best.pdparams'
    results_file = str(logdir / 'results.txt')
    epochs, batch_size, total_batch_size, weights = \
        opt.epochs, opt.batch_size, opt.total_batch_size, opt.weights

    # TODO: Use DDP logging. Only the first process is allowed to log.
    # Save run settings
    with open(logdir / 'hyp.yaml', 'w') as f:
        yaml.dump(hyp, f, sort_keys=False)
    with open(logdir / 'opt.yaml', 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)

    # Configure
    init_seeds(2)
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)  # model dict
    train_path = data_dict['train']
    test_path = data_dict['val']
    nc, names = (1, ['item']) if opt.single_cls else (int(data_dict['nc']), data_dict['names'])  # number classes, names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # check

    # Model
    pretrained = weights.endswith('.pdparams')
    if pretrained:
        ckpt = paddle.load(weights)  # load checkpoint
        model = Model(opt.cfg or ckpt['model'].yaml, ch=3, nc=nc)  # create
        exclude = ['anchor'] if opt.cfg else []  # exclude keys
        state_dict = ckpt['model'].astype('float32').state_dict()  # to FP32
        state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
        model.set_state_dict(state_dict)  # load
        print('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), weights))  # report
    else:
        model = Model(opt.cfg, ch=3, nc=nc)# create
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.8 + 0.2  # cosine
    scheduler = lr_scheduler.LambdaDecay(hyp['lr0'], lr_lambda=lf)

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / total_batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= total_batch_size * accumulate / nbs  # scale weight_decay

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_parameters():
        v.stop_gradient = False
        if '.bias' in k:
            pg2.append(v)  # biases
        elif '.weight' in k and '.bn' not in k:
            pg1.append(v)  # apply weight decay
        else:
            pg0.append(v)  # all else

    if opt.adam:
        optimizer = optim.Adam(parameters=pg0+pg1+pg2, learning_rate=scheduler, beta1=hyp['momentum'], beta2=0.999, weight_decay=hyp['weight_decay'])  # adjust beta1 to momentum
    else:
        optimizer = optim.Momentum(parameters=pg0+pg1+pg2, learning_rate=scheduler, momentum=hyp['momentum'], use_nesterov=True, weight_decay=hyp['weight_decay'])

    print('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2

    # Resume
    start_epoch, best_fitness = 0, 0.0
    if pretrained:
        # Optimizer
        if ckpt['optimizer'] is not None:
            optimizer.set_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']

        # Results
        if ckpt.get('training_results') is not None:
            with open(results_file, 'w') as file:
                file.write(ckpt['training_results'])  # write results.txt

        # Epochs
        start_epoch = ckpt['epoch'] + 1
        if epochs < start_epoch:
            print('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                  (weights, ckpt['epoch'], epochs))
            epochs += ckpt['epoch']  # finetune additional epochs

        del ckpt, state_dict
    
    # Image sizes
    gs = int(max(model.stride))  # grid size (max stride)
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples

    # Exponential moving average
    ema = ModelEMA(model)


    # Trainloader
    dataloader, dataset = create_dataloader(train_path, imgsz, batch_size, gs, opt, hyp=hyp, augment=True,
                                            cache=opt.cache_images, rect=opt.rect)

    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
    nb = len(dataloader)  # number of batches
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, opt.data, nc - 1)

    # Testloader
    ema.updates = start_epoch * nb // accumulate  # set EMA updates ***
    # local_rank is set to -1. Because only the first process is expected to do evaluation.
    testloader = create_dataloader(test_path, imgsz_test, batch_size, gs, opt, hyp=hyp, augment=False,
                                    cache=opt.cache_images, rect=True)[0]

    # Model parameters
    hyp['cls'] *= nc / 80.  # scale coco-tuned hyp['cls'] to current dataset
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # giou loss ratio (obj_loss = 1.0 or giou)
    model.class_weights = labels_to_class_weights(dataset.labels, nc)  # attach class weights
    model.names = names

    # Class frequency
    labels = np.concatenate(dataset.labels, 0)
    c = paddle.to_tensor(labels[:, 0])  # classes
    # plot_labels(labels, save_dir=logdir)
    if tb_writer:
        tb_writer.add_histogram('classes', c, 0)

    # Check anchors
    if not opt.noautoanchor:
        check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)

    # Start training
    t0 = time.time()
    nw = max(3 * nb, 1e3)  # number of warmup iterations, max(3 epochs, 1k iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = amp.GradScaler(enable=True, init_loss_scaling=65536.0, incr_every_n_steps=2000)
    print('Image sizes %g train, %g test' % (imgsz, imgsz_test))
    print('Using %g dataloader workers' % dataloader.num_workers)
    print('Starting training for %g epochs...' % epochs)
    model = paddle.DataParallel(model)
    model.train()
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        # Update mosaic border
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        mloss = paddle.zeros([4])  # mean losses
        pbar = enumerate(dataloader)
        print(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'targets', 'img_size'))
        pbar = tqdm(pbar, total=nb)  # progress bar
        optimizer.clear_grad()
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.astype('float32') / 255.0  # uint8 to float32, 0-255 to 0.0-1.0
            targets = targets.astype('float32')
            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round())
                if 'Momentum' in str(type(optimizer)).split('.')[-1]:
                    optimizer._momentum = np.interp(ni, xi, [0.9, hyp['momentum']])

            # Multi-scale
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Autocast
            with amp.auto_cast(enable=True):
                # Forward                
                pred = model(imgs)

                # Loss
                loss, loss_items = compute_loss(pred, targets, model._layers)  # scaled by batch_size

            # Backward
            scaled = scaler.scale(loss)  # scale the loss
            scaled.backward()

            # Optimize
            if ni % accumulate == 0:
                scaler.minimize(optimizer, scaled) # optimizer.step
                optimizer.clear_grad()
                if ema is not None:
                    ema.update(model)

            # Print
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            mem = '%.3gG' % 0  # (GB)
            s = ('%10s' * 2 + '%10.4g' * 6) % (
                '%g/%g' % (epoch, epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
            pbar.set_description(s)

            # Plot
            if ni < 3:
                f = str(logdir / ('train_batch%g.jpg' % ni))  # filename
                result = plot_images(images=imgs, targets=targets, paths=paths, fname=f)
                if tb_writer and result is not None:
                    tb_writer.add_image(f, result, dataformats='HWC', step=epoch)
                    # tb_writer.add_graph(model, imgs)  # add model to visualDL

            # end batch ------------------------------------------------------------------------------------------------

        # Scheduler
        scheduler.step()

        # DDP process 0 or single-GPU
        # mAP
        if ema is not None:
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride'])
        final_epoch = epoch + 1 == epochs
        # end epoch ----------------------------------------------------------------------------------------------------
    # end training

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov4-p5.pdparams', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='data.yaml path')
    parser.add_argument('--hyp', type=str, default='', help='hyperparameters path, i.e. data/hyp.scratch.yaml')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='train,test sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const='get_last', default=False,
                        help='resume from given path/last.pdparams, or most recent run if blank')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    parser.add_argument('--adam', action='store_true', help='use paddle.optimizer.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--logdir', type=str, default='runs/', help='logging directory')
    opt = parser.parse_args()

    # train(hyp, opt, tb_writer)
    dist.spawn(train, args=(opt,), nprocs=4)
