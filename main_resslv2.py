import argparse
import os
import time

import torch.backends.cudnn as cudnn
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.transforms as T

# from models.densecl import DenseCL
# from models.proposal import Proposal
from models.resslv2 import ReSSLv2
from models.loader import ContrastiveDatasetWeakAugmentation

from transforms import *

from backbones import resnet
from models import necks

model_names = sorted(name for name in resnet.__all__ if name.islower() and callable(resnet.__dict__[name]))
neck_names = sorted(name for name in necks.__all__ if name.islower() and callable(necks.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch COCO Pre-Training...')

parser.add_argument('--lr', type=float, default=0.3, help='base lr')

parser.add_argument('--optimizer', type=str, choices=['sgd', 'lars'], default='sgd',
                    help='for optimizer choice.')
parser.add_argument('--wd', type=float, default=1e-4, help='weight decay')
parser.add_argument('--m', type=float, default=0.9, help='momentum for SGD')

parser.add_argument('--batch_size', type=int, default=64, help='batch_size for single gpu')

parser.add_argument('--crop', type=float, default=0.2)
parser.add_argument('--size', type=int, default=224, help='image crop size')

parser.add_argument('--queue_len', type=int, default=65536, help='')
parser.add_argument('--feature_dim', type=int, default=128, help='feature dimension')
parser.add_argument('--momentum', type=float, default=0.999,
                    help='momentume parameter used in MoCo and InstDisc')

# loss
parser.add_argument('--loss_lambda', type=float, default=0.5, help='')

parser.add_argument('--temperature', type=float, default=0.1, help='temperature value for student')
parser.add_argument('--temperature_momentum', type=float, default=0.04, help='temperature value for teacher')

parser.add_argument('--backbone', type=str, default='resnet50', choices=model_names, help="backbone architecture")
parser.add_argument('--neck', type=str, default='densecl_neck', choices=neck_names, help="neck architecture")

parser.add_argument('--in_channels', type=int, default=2048, help='feature dimension')
parser.add_argument('--hid_channels', type=int, default=2048, help='feature dimension')
parser.add_argument('--out_channels', type=int, default=128, help='feature dimension')

parser.add_argument('--lr_scheduler', type=str, default='cosine', choices=["step", "cosine"],  # densecl coco pre-train
                    help="learning rate scheduler")

parser.add_argument('--start_epoch', type=int, default=0, help='used for resume')
parser.add_argument('--epochs', type=int, default=800, help='number of training epochs')

parser.add_argument('--path', type=str,
                    default='/users/timoteos.ozcelik/selfsup/detection/datasets/data/coco/train2017/',
                    help='dataset director')
parser.add_argument('--output_dir', type=str, default='/users/timoteos.ozcelik', help='output director')

parser.add_argument('--save_every', type=int, default=100, help='save in each save_every epoch')

parser.add_argument('--print_freq', type=int, default=100, help='')

def main():
    args = parser.parse_args()

    ngpus_per_node = torch.cuda.device_count()
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))

def main_worker(rank, world_size, args):
    print("Use GPU: {} for training".format(rank))

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12381"

    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size
    )

    # create model
    print("=> creating model '{}'".format(args.backbone))

    # updated version
    model = ReSSLv2(
        # resnet.__dict__[args.backbone](),
        resnet.__dict__[args.backbone],
        # necks.__dict__[args.neck](args.in_channels, args.hid_channels, args.out_channels),
        necks.__dict__[args.neck],
        args.queue_len, args.feature_dim, args.momentum, args.temperature, args.temperature_momentum
    )

    torch.cuda.set_device(rank)
    model.cuda(rank)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.wd,
        momentum=args.m,
    )

    cudnn.benchmark = True  # added

    weak_aug = T.Compose([
        T.RandomResizedCrop(size=args.size, scale=(args.crop, 1.)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    strong_aug = T.Compose([
        T.RandomResizedCrop(size=args.size, scale=(args.crop, 1.)),
        T.RandomApply([
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
        )], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.RandomApply([GaussianBlur(sigma=[0.1, 2.0])], p=0.5),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = ContrastiveDatasetWeakAugmentation(args.path, weak_aug, strong_aug)

    sampler = torch.utils.data.distributed.DistributedSampler(dataset)

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=(sampler is None),
        num_workers=4, pin_memory=True, sampler=sampler, drop_last=True)

    for epoch in range(args.start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        train(train_loader, model, optimizer, epoch, args, rank)

        if rank == 0 and (epoch + 1) % args.save_every == 0:
            save_checkpoint(args, epoch, model, optimizer)

    dist.destroy_process_group()


def train(train_loader, model, optimizer, epoch, args, rank):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')

    # top1 = AverageMeter('Acc@1', ':6.2f')
    # top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), [batch_time, data_time, losses],
                             prefix="Epoch: [{}]".format(epoch))

    # amp scalar
    scaler = amp.GradScaler()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (im_q, im_k) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        im_q = im_q.cuda(rank, non_blocking=True)
        im_k = im_k.cuda(rank, non_blocking=True)

        with amp.autocast():
            loss = model(im_q, im_k)

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        with torch.no_grad():
            # logits, labels = extra['logits'], extra['labels']
            # acc1, acc5 = accuracy(logits, labels, topk=(1, 5))
            losses.update(loss.item(), im_q.size(0))

            # losses.update(loss.item(), im_q.size(0))
            # top1.update(acc1[0], im_q.size(0))
            # top5.update(acc5[0], im_q.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        #  loss.backward()
        #  optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and rank == 0:
            progress.display(i)

def save_checkpoint(args, epoch, model, optimizer):
    state = {
        'opt': args,
        'state_dict': model.module.backbone.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }

    PATH = os.path.join(args.output_dir, f'ckpt_epoch_{epoch}.pth')
    torch.save(state, PATH)
    print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    lr *= 0.5 * (1.0 + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = output.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()

