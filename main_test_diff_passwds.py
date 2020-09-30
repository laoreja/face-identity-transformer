# system libraries
import os, sys
import os.path as osp
import time
import numpy as np
from PIL import Image
import gc
from collections import OrderedDict

import torch
import torchvision.transforms as transforms

# libraries within this package
from cmd_args import parse_args
from utils.visualizer import Visualizer
from utils.util import generate_code
import datasets
import models

TEST_CODE_NUM = 10


def main():
    # parse args
    global args
    args = parse_args(sys.argv[1])
    args.during_training = False

    args.gpu_ids = list(range(len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))))
    args.device = torch.device('cuda:0')
    args.test_size = args.batch_size // 4 * len(args.gpu_ids)

    # add timestamp to ckpt_dir
    args.timestamp = time.strftime('%m%d%H%M%S', time.localtime())
    args.ckpt_dir += '_' + args.timestamp


    # -------------------- init ckpt_dir, logging --------------------
    os.makedirs(args.ckpt_dir, mode=0o777, exist_ok=True)

    # -------------------- init visu --------------------
    visualizer = Visualizer(args)

    visualizer.logger.log('sys.argv:\n' + ' '.join(sys.argv))
    for arg in sorted(vars(args)):
        visualizer.logger.log('{:20s} {}'.format(arg, getattr(args, arg)))
    visualizer.logger.log('')

    # -------------------- dataset & loader --------------------
    test_dataset = datasets.__dict__[args.dataset](
        train=False,
        transform=transforms.Compose([
            transforms.Resize(args.imageSize, Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))
        ]),
        args=args
    )

    visualizer.logger.log('test_dataset: ' + str(test_dataset))

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.test_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32))
    )

    # -------------------- create model --------------------
    model_dict = {}

    G_input_nc = args.input_nc + args.passwd_length
    model_dict['G'] = models.define_G(G_input_nc, args.output_nc,
                                      args.ngf, args.which_model_netG, args.n_downsample_G,
                                      args.normG, args.dropout,
                                      args.init_type, args.init_gain,
                                      args.passwd_length,
                                      use_leaky=args.use_leakyG,
                                      use_resize_conv=args.use_resize_conv,
                                      padding_type=args.padding_type)
    model_dict['G_nets'] = [model_dict['G']]

    print('model_dict')
    for k, v in model_dict.items():
        print(k + ':')
        if isinstance(v, list):
            print('list, len:', len(v))
            print('')
        else:
            print(v)

    # -------------------- resume --------------------
    if args.resume:
        if osp.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint['epoch'] + 1

            name = 'G'
            net = model_dict[name]
            if isinstance(net, torch.nn.DataParallel):
                net = net.module
            net.load_state_dict(checkpoint['state_dict_' + name])

            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
        gc.collect()
        torch.cuda.empty_cache()

    test(test_loader, model_dict, visualizer, args)


def test(test_loader, model_dict, visualizer, args, iter=0):
    model_dict['G'].train()

    with torch.no_grad():
        for dis_idx in range(TEST_CODE_NUM):
            z, dis_target, \
            rand_z, rand_dis_target, \
            inv_z, inv_dis_target, \
            rand_inv_z, rand_inv_dis_target, _, _ = generate_code(args.passwd_length,
                                                                  args.batch_size,
                                                                  args.device,
                                                                  inv=True,
                                                                  use_minus_one=args.use_minus_one,
                                                                  gen_random_WR=False)

            # all the test images use the same passwords
            for i, (img, label, landmarks, img_path) in enumerate(test_loader):
                fake = model_dict['G'](img, z.cpu())
                recon = model_dict['G'](fake, inv_z)
                rand_recon = model_dict['G'](fake, rand_inv_z)

                current_visuals = OrderedDict()
                current_visuals['real'] = img
                current_visuals['fake'] = fake
                current_visuals['recon'] = recon
                current_visuals['rand_recon'] = rand_recon

                visualizer.display_test_results_vertical_html(current_visuals, img_path, dis_idx, iter, use_real=True,
                                                              add_padding=False, refresh=-1)

if __name__ == '__main__':
    main()
