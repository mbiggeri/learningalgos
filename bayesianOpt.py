# Not finished, it works but some things need to be reviewed and it needs to be made more intuitive to use

import json
import threading
import torch.utils.data
import torchvision
import argparse
import optuna
from models import (make_pools, P_MLP, VF_MLP, P_CNN, VF_CNN, RON,
                    my_init, my_sigmoid, my_hard_sig, ctrd_hard_sig, hard_sigmoid, train_epoch, evaluate)

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='/Users/michaelbiggeri/Desktop/Tesi/Codice/datasets')	# folder where the datasets will be downloaded
parser.add_argument('--model', type=str, default='MLP', metavar='m', help='model', choices=['MLP', 'VFMLP', 'CNN', 'VFCNN', 'RON'])
parser.add_argument('--task', type=str, default='MNIST', metavar='t', help='task', choices=['MNIST', 'CIFAR10'])

parser.add_argument('--pools', type=str, default='mm', metavar='p', help='pooling')
parser.add_argument('--channels', nargs='+', type=int, default=[32, 64], metavar='C', help='channels of the convnet')
parser.add_argument('--kernels', nargs='+', type=int, default=[5, 5], metavar='K', help='kernels sizes of the convnet')
parser.add_argument('--strides', nargs='+', type=int, default=[1, 1], metavar='S', help='strides of the convnet')
parser.add_argument('--paddings', nargs='+', type=int, default=[0, 0], metavar='P', help='paddings of the conv layers')
parser.add_argument('--fc', nargs='+', type=int, default=[10], metavar='S', help='linear classifier of the convnet')

parser.add_argument('--act', type=str, default='mysig', metavar='a', help='activation function')
parser.add_argument('--wds', nargs='+', type=float, default=None, metavar='l', help='layer weight decays')
parser.add_argument('--loss', type=str, default='mse', metavar='lss', help='loss for training')
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

parser.add_argument('--learn_oscillators', action='store_true')

# Script di utilizzo: python bayesianOpt.py \ --data_root "/percorso/ai/tuoi/dataset" \ --epochs (number) \ --model (model) \ --task (task)
# esempio: "python bayesianOpt.py --model RON --task CIFAR10 --epochs 5"


args = parser.parse_args()

# device = ('cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Utilizzo del dispositivo:', device)

# Caricamento del dataset
if args.task == 'MNIST':
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.0,), std=(1.0,))
    ])
    dataset = torchvision.datasets.MNIST(root=args.data_root, train=True, transform=transform, download=True)
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])
    test_dataset = torchvision.datasets.MNIST(root=args.data_root, train=False, transform=transform, download=True)
elif args.task == 'CIFAR10':
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                         std=(0.2023, 0.1994, 0.2010))
    ])
    dataset = torchvision.datasets.CIFAR10(root=args.data_root, train=True, transform=transform, download=True)
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])
    test_dataset = torchvision.datasets.CIFAR10(root=args.data_root, train=False, transform=transform, download=True)

def objective(trial):
    # Definiamo due dizionari, uno con gli iperparametri fissati e l'altro con gli iperparametri da ottimizzare, poi uniamo i dizionari.
    ######################################################
    # Dizionario degli iperparametri fissi
    fixed_params = {
        'architecture': [3072, 512, 512, 10],
        'optimizer': 'sgd',
        'lrs': [0.01, 0.01, 0.01],  # Due valori per due layer
        'activation': 'my_hard_sig',
        'T1': 100,
        'T2': 20,
        'batch_size': 128,  # mbs
        'alg': 'EP',
        'betas': (0.0, 0.5),
        'loss': 'mse',

        #---RON---
        # 'eps_min': 1.0,
        # 'eps_max': 2.0,
        # 'gamma_min': 1.0,
        # 'gamma_max': 2.0,
        # 'tau': 0.1,
    }

    # Dizionario degli iperparametri da ottimizzare
    opt_params = {
        # 'architecture': trial.suggest_categorical('architecture', [[784, 512, 10], [784, 256, 10]]),
        # 'optimizer': trial.suggest_categorical('optimizer', ['sgd', 'adam']),
        # 'activation': trial.suggest_categorical('activation', ['my_hard_sig', 'mysig', 'sigmoid', 'tanh']),
        # 'T1': trial.suggest_int('T1', 20, 50, step=5),
        # 'T2': trial.suggest_int('T2', 5, 25, step=2),
        # 'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
        # 'alg': trial.suggest_categorical('alg', ['EP', 'BPTT', 'CEP']),
        # 'betas': (0.0, trial.suggest_float('beta2', 0.0, 1.0, step=0.1)),
        # 'loss': trial.suggest_categorical('loss', ['mse', 'cel'])

        #---RON---
        'eps_min': trial.suggest_float('eps_min', 0.1, 2.0, step=0.2),
        'gamma_min': trial.suggest_float('gamma_min', 0.1, 2.0, step=0.2),
        
        'tau': trial.suggest_float('tau', 0.1, 1.0, step=0.1),
    }

    # Devono essere inizializzati dopo eps_min e gamma_min se li vogliamo usare come parametri
    opt_params['eps_max'] = trial.suggest_float('eps_max', opt_params['eps_min'], 3.0, step=0.1)
    opt_params['gamma_max'] = trial.suggest_float('gamma_max', opt_params['gamma_min'], 3.0, step=0.1)

    # Unisci i due dizionari
    params = {**fixed_params, **opt_params}
    
    # Dopo aver suggerito 'optimizer', suggeriamo 'momentum' se 'optimizer' è 'sgd'
    if params['optimizer'] == 'sgd':
        params['momentum'] = 0.9
        # opt_params['momentum'] = trial.suggest_float('momentum', 0.0, 0.9)
    else:
        params['momentum'] = 0.0  # Impostiamo 'momentum' a 0.0 se non usiamo SGD
    ######################################################

    #------------------------------------------------------
    # TODO: Aggiungi possibilità di ottimizzare o meno lrs:
    #------------------------------------------------------
    
    
    # Gestione di lrs per ogni layer
    archi = params['architecture']
    '''
    num_layers = len(archi) - 1  # Numero di layer (input non conta come layer)
    lrs = []
    for i in range(num_layers):
        lr_i = trial.suggest_float(f'lr_layer{i+1}', 1e-5, 1e-1, log=True)
        lrs.append(lr_i)
    params['lrs'] = lrs
    '''

    # Stampa i parametri utilizzati per questo trial
    print('Trial hyperparameters:', params)


    # Preparazione dei data loader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=params['batch_size'], shuffle=False)

    # Selezione della funzione di attivazione
    activation_map = {
        'mysig': my_sigmoid,
        'sigmoid': torch.sigmoid,
        'tanh': torch.tanh,
        'my_hard_sig': my_hard_sig
    }
    activation = activation_map[params['activation']]

    # Definizione dell'architettura del modello
    if args.model == 'MLP':
        model = P_MLP(archi, activation=activation)
    elif args.model == 'VFMLP':
        model = VF_MLP(archi, activation=activation)
    elif args.model == 'RON':
        model = RON(archi, device=device, activation=activation, epsilon_min=params['eps_min'], epsilon_max=params['eps_max'],
                gamma_max=params['gamma_max'], gamma_min=params['gamma_min'], tau=params['tau'], learn_oscillators=args.learn_oscillators)
    # Aggiungi altri modelli se necessario

    model.to(device)

    # Definizione dell'ottimizzatore per ogni layer
    optim_params = []
    for idx, synapse in enumerate(model.synapses):
        optim_params.append({'params': synapse.parameters(), 'lr': params['lrs'][idx]})

    if params['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(optim_params, momentum=params['momentum'])
    elif params['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(optim_params)

    # Selezione della loss function
    if params['loss'] == 'mse':
        criterion = torch.nn.MSELoss(reduction='none').to(device)
    elif params['loss'] == 'cel':
        criterion = torch.nn.CrossEntropyLoss(reduction='none').to(device)

    # Constructing the scheduler
    if args.lr_decay:
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40,80,120], gamma=0.1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100, eta_min=1e-5)
    else:
        scheduler = None

    # Ciclo di training
    for epoch in range(args.epochs):
        train_epoch(model, optimizer, epoch, train_loader, params['T1'], params['T2'], params['betas'], device,
                criterion, params['alg'], random_sign=args.random_sign, thirdphase=args.thirdphase, cep_debug=args.cep_debug,
                ron=(args.model == 'RON'), id=trial.number)

        # Learning rate decay step
        if scheduler is not None:
            if epoch < scheduler.T_max:
                scheduler.step()

        test_acc = evaluate(model, valid_loader, params['T1'], device, ron=(args.model == 'RON')) 
        print('Trial ', trial.id, '\nTest accuracy :', round(test_acc, 2))
        
        # Report intermedi a Optuna
        trial.report(test_acc, epoch)
        
        # Interrompi il trial se Hyperband decide di prunare
        if trial.should_prune():
            raise optuna.TrialPruned()

    # Validazione
    training_acc = evaluate(model, train_loader, params['T1'], device, ron=(args.model == 'RON'))
    return test_acc

# Definizione di un pruner Hyperband
pruner = optuna.pruners.HyperbandPruner(
    min_resource=1,        # Minimo numero di epoche
    max_resource=5,        # Massimo numero di epoche
    reduction_factor=3     # Fattore di riduzione (si tagliano 1/3 dei rami)
)

# Esecuzione dello studio Optuna
study = optuna.create_study(direction='maximize', pruner=pruner)
study.optimize(objective, n_trials=60, n_jobs=-1)    # n_jobs=-1 utilizza tutti i core disponibili

# Mostra i 5 migliori set di iperparametri trovati
print('\nTop 5 Best Trials:')
top_trials = sorted(study.trials, key=lambda t: t.value, reverse=True)[:5]
for i, trial in enumerate(top_trials):
    print(f"Rank {i+1}:")
    print(f"  Value: {trial.value}")
    print(f"  Params: {trial.params}")
    
# Salvare i risultati su un JSON
with open('optuna_results.json', 'w') as f:
    json.dump([t.params for t in study.trials], f)
