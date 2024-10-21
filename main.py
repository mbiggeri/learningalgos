"""
Adapted from
https://github.com/Laborieux-Axel/Equilibrium-Propagation/blob/master/main.py
"""
import torch.utils.data
import torchvision
import argparse
from models import (make_pools, P_MLP, VF_MLP, P_CNN, VF_CNN, RON,
                    my_init, my_sigmoid, my_hard_sig, ctrd_hard_sig, hard_sigmoid, train_epoch, evaluate)


parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='/home/gibberi/Desktop/Tesi/Datasets')
parser.add_argument('--model', type=str, default='MLP', metavar='m', help='model', choices=['MLP', 'VFMLP', 'CNN', 'VFCNN', 'RON'])
parser.add_argument('--task', type=str, default='MNIST', metavar='t', help='task', choices=['MNIST', 'CIFAR10'])

parser.add_argument('--pools', type=str, default='mm', metavar='p', help='pooling')
parser.add_argument('--archi', nargs='+', type=int, default=[784, 512, 10], metavar='A',
                    help='architecture of the network')
parser.add_argument('--channels', nargs='+', type=int, default=[32, 64], metavar='C', help='channels of the convnet')
parser.add_argument('--kernels', nargs='+', type=int, default=[5, 5], metavar='K', help='kernels sizes of the convnet')
parser.add_argument('--strides', nargs='+', type=int, default=[1, 1], metavar='S', help='strides of the convnet')
parser.add_argument('--paddings', nargs='+', type=int, default=[0, 0], metavar='P', help='paddings of the conv layers')
parser.add_argument('--fc', nargs='+', type=int, default=[10], metavar='S', help='linear classifier of the convnet')

parser.add_argument('--act', type=str, default='mysig', metavar='a', help='activation function')
parser.add_argument('--optim', type=str, default='sgd', metavar='opt', help='optimizer for training')
parser.add_argument('--lrs', nargs='+', type=float, default=[], metavar='l', help='layer wise lr')
parser.add_argument('--wds', nargs='+', type=float, default=None, metavar='l', help='layer weight decays')
parser.add_argument('--mmt', type=float, default=0.0, metavar='mmt', help='Momentum for sgd')
parser.add_argument('--loss', type=str, default='mse', metavar='lss', help='loss for training')
parser.add_argument('--alg', type=str, default='EP', metavar='al', help='EP or BPTT or CEP')
parser.add_argument('--mbs', type=int, default=20, metavar='M', help='minibatch size')
parser.add_argument('--T1', type=int, default=20, metavar='T1', help='Time of first phase')
parser.add_argument('--T2', type=int, default=4, metavar='T2', help='Time of second phase')
parser.add_argument('--betas', nargs='+', type=float, default=[0.0, 0.01], metavar='Bs', help='Betas')
parser.add_argument('--epochs', type=int, default=1, metavar='EPT', help='Number of epochs per tasks')
parser.add_argument('--random-sign', default=False, action='store_true', help='randomly switch beta_2 sign')
parser.add_argument('--data-aug', default=False, action='store_true', help='enabling data augmentation for cifar10')
parser.add_argument('--lr-decay', default=False, action='store_true', help='enabling learning rate decay')
parser.add_argument('--scale', type=float, default=None, metavar='g', help='scal factor for weight init')
parser.add_argument('--seed', type=int, default=None, metavar='s', help='random seed')
parser.add_argument('--thirdphase', default=False, action='store_true',
                    help='add third phase for higher order evaluation of the gradient (default: False)')
parser.add_argument('--softmax', default=False, action='store_true',
                    help='softmax loss with parameters (default: False)')
parser.add_argument('--same-update', default=False, action='store_true',
                    help='same update is applied for VFCNN back and forward')
parser.add_argument('--cep-debug', default=False, action='store_true', help='debug cep')
parser.add_argument('--use_test', action='store_true', help='evaluate on test set instead of validation')

parser.add_argument('--eps_min', type=float, default=1.0)
parser.add_argument('--eps_max', type=float, default=2.0)
parser.add_argument('--gamma_min', type=float, default=1.0)
parser.add_argument('--gamma_max', type=float, default=2.0)
parser.add_argument('--tau', type=float, default=0.1)
parser.add_argument('--learn_oscillators', action='store_true')

parser.add_argument('--weight-decay', type=float, default=0.0, metavar='wd',
                    help='Weight decay (L2 regularization) factor (default: 0.0)')
parser.add_argument('--use-weight-decay', default=False, action='store_true',
                    help='Enable L2 regularization (default: False)')


args = parser.parse_args()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


mbs = args.mbs
if args.seed is not None:
    torch.manual_seed(args.seed)

if args.task == 'MNIST':
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize(mean=(0.0,), std=(1.0,))])

    mnist_dset_train = torchvision.datasets.MNIST(root=args.data_root, train=True, transform=transform,
                                                  target_transform=None, download=True)
    if not args.use_test:
        mnist_dset_train, mnist_dset_valid = torch.utils.data.random_split(mnist_dset_train, [45000, 15000])
        valid_loader = torch.utils.data.DataLoader(mnist_dset_valid, batch_size=200, shuffle=False, num_workers=0)

    train_loader = torch.utils.data.DataLoader(mnist_dset_train, batch_size=mbs, shuffle=True, num_workers=0)
    mnist_dset_test = torchvision.datasets.MNIST(root=args.data_root, train=False, transform=transform,
                                                 target_transform=None, download=True)
    test_loader = torch.utils.data.DataLoader(mnist_dset_test, batch_size=200, shuffle=False, num_workers=0)

elif args.task == 'CIFAR10':
    if args.data_aug:
        transform_train = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(0.5),
                                                          torchvision.transforms.RandomCrop(size=[32, 32], padding=4,
                                                                                            padding_mode='edge'),
                                                          torchvision.transforms.ToTensor(),
                                                          torchvision.transforms.Normalize(
                                                              mean=(0.4914, 0.4822, 0.4465),
                                                              std=(3 * 0.2023, 3 * 0.1994, 3 * 0.2010))])
    else:
        transform_train = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                          torchvision.transforms.Normalize(
                                                              mean=(0.4914, 0.4822, 0.4465),
                                                              std=(3 * 0.2023, 3 * 0.1994, 3 * 0.2010))])

    transform_test = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                     torchvision.transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                                                                      std=(3 * 0.2023, 3 * 0.1994,
                                                                                           3 * 0.2010))])

    cifar10_train_dset = torchvision.datasets.CIFAR10(root=args.data_root, train=True, transform=transform_train,
                                                      download=True)
    if not args.use_test:
        cifar_train_size = int(0.7*len(cifar10_train_dset))
        cifar10_train_dset, cifar10_valid_dset = torch.utils.data.random_split(
            cifar10_train_dset, [cifar_train_size, len(cifar10_train_dset) - cifar_train_size])
        valid_loader = torch.utils.data.DataLoader(cifar10_valid_dset, batch_size=200, shuffle=False, num_workers=1)

    cifar10_test_dset = torchvision.datasets.CIFAR10(root=args.data_root, train=False, transform=transform_test,
                                                     download=True)
    train_loader = torch.utils.data.DataLoader(cifar10_train_dset, batch_size=mbs, shuffle=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(cifar10_test_dset, batch_size=200, shuffle=False, num_workers=1)

if args.act == 'mysig':
    activation = my_sigmoid
elif args.act == 'sigmoid':
    activation = torch.sigmoid
elif args.act == 'tanh':
    activation = torch.tanh
elif args.act == 'hard_sigmoid':
    activation = hard_sigmoid
elif args.act == 'my_hard_sig':
    activation = my_hard_sig
elif args.act == 'ctrd_hard_sig':
    activation = ctrd_hard_sig

if args.loss == 'mse':
    criterion = torch.nn.MSELoss(reduction='none').to(device)
elif args.loss == 'cel':
    criterion = torch.nn.CrossEntropyLoss(reduction='none').to(device)
print('loss =', criterion, '\n')



if args.model == 'MLP':
    model = P_MLP(args.archi, activation=activation)

elif args.model == 'VFMLP':
    model = VF_MLP(args.archi, activation=activation)
elif args.model == 'RON':
    model = RON(args.archi, device=device, activation=activation, epsilon_min=args.eps_min, epsilon_max=args.eps_max,
                gamma_max=args.gamma_max, gamma_min=args.gamma_min, tau=args.tau, learn_oscillators=args.learn_oscillators)
elif args.model.find('CNN') != -1:

    if args.task == 'MNIST':
        pools = make_pools(args.pools)
        channels = [1] + args.channels
        if args.model == 'CNN':
            model = P_CNN(28, channels, args.kernels, args.strides, args.fc, pools, args.paddings,
                          activation=activation, softmax=args.softmax)
        elif args.model == 'VFCNN':
            model = VF_CNN(28, channels, args.kernels, args.strides, args.fc, pools, args.paddings,
                           activation=activation, softmax=args.softmax, same_update=args.same_update)

    elif args.task == 'CIFAR10':
        pools = make_pools(args.pools)
        channels = [3] + args.channels
        if args.model == 'CNN':
            model = P_CNN(32, channels, args.kernels, args.strides, args.fc, pools, args.paddings,
                          activation=activation, softmax=args.softmax)
        elif args.model == 'VFCNN':
            model = VF_CNN(32, channels, args.kernels, args.strides, args.fc, pools, args.paddings,
                           activation=activation, softmax=args.softmax, same_update=args.same_update)
    else:
        raise ValueError('Task not recognized')

    print('\n')
    print('Poolings =', model.pools)

if args.scale is not None:
    model.apply(my_init(args.scale))

model.to(device)
print(model)

betas = args.betas[0], args.betas[1]

assert (len(args.lrs) == len(model.synapses))

# Constructing the optimizer
optim_params = []
if (args.alg == 'CEP' and args.wds) and not (args.cep_debug):
    for idx in range(len(model.synapses)):
        args.wds[idx] = (1 - (1 - args.wds[idx] * 0.1) ** (1 / args.T2)) / args.lrs[idx]

for idx in range(len(model.synapses)):
    if args.wds is None:
        optim_params.append({'params': model.synapses[idx].parameters(), 'lr': args.lrs[idx]})
    else:
        optim_params.append(
            {'params': model.synapses[idx].parameters(), 'lr': args.lrs[idx], 'weight_decay': args.wds[idx]})
if hasattr(model, 'B_syn'):
    for idx in range(len(model.B_syn)):
        if args.wds is None:
            optim_params.append({'params': model.B_syn[idx].parameters(), 'lr': args.lrs[idx + 1]})
        else:
            optim_params.append({'params': model.B_syn[idx].parameters(), 'lr': args.lrs[idx + 1],
                                 'weight_decay': args.wds[idx + 1]})

if args.optim == 'sgd':
    optimizer = torch.optim.SGD(
        optim_params, 
        momentum=args.mmt,
        weight_decay=args.weight_decay if args.use_weight_decay else 0.0)
elif args.optim == 'adam':
    optimizer = torch.optim.Adam(
        optim_params,
        weight_decay=args.weight_decay if args.use_weight_decay else 0.0)

# Constructing the scheduler
if args.lr_decay:
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40,80,120], gamma=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100, eta_min=1e-5)
else:
    scheduler = None

print(optimizer)
print('\ntraining algorithm : ', args.alg, '\n')

eval_loader = test_loader if args.use_test else valid_loader

for epoch in range(args.epochs):
    train_epoch(model, optimizer, epoch, train_loader, args.T1, args.T2, betas, device,
                criterion, alg=args.alg, random_sign=args.random_sign, thirdphase=args.thirdphase, cep_debug=args.cep_debug,
                ron=(args.model == 'RON'))

    if scheduler is not None:  # learning rate decay step
        if epoch < scheduler.T_max:
            scheduler.step()

    test_acc = evaluate(model, eval_loader, args.T1, device, ron=(args.model == 'RON'))
    print('\nTest accuracy :', round(test_acc, 2))

training_acc = evaluate(model, train_loader, args.T1, device, ron=(args.model == 'RON'))
print('\nTrain accuracy :', round(training_acc, 2))
test_acc = evaluate(model, eval_loader, args.T1, device, ron=(args.model == 'RON'))
print('\nTest accuracy :', round(test_acc, 2))
