# MLP MNIST 95%
CUDA_VISIBLE_DEVICES=5 python main.py --model MLP --task MNIST --archi 784 512 10 --optim sgd --lrs 0.01 0.01 --mmt 0.9 --epochs 10 --act my_hard_sig --T1 250 --T2 30 --mbs 128 --alg EP --betas 0.0 0.5 --loss mse

# RON MNIST 98% OSCILLATORS NOT LEARNED
CUDA_VISIBLE_DEVICES=1 python main.py --model RON --task MNIST --archi 784 512 10 --optim sgd --lrs 0.1 0.1 --mmt 0.9 --epochs 10 --act my_hard_sig --T1 250 --T2 30 --mbs 128 --alg EP --betas 0.0 0.5 --loss mse --gamma_min 0.2 --gamma_max 0.7 --eps_min 0.2 --eps_max 0.7 --tau 0.8 --use_test

# MLP CIFAR10 48%
 CUDA_VISIBLE_DEVICES=2 python main.py --model MLP --task CIFAR10 --archi 3072 512 512 10 --optim sgd --lrs 0.01 0.01 0.01 --mmt 0.9 --epochs 20 --act my_hard_sig --T1 250 --T2 30 --mbs 128 --alg EP --betas 0.0 0.5 --loss mse

# RON CIFAR10 48% OSCILLATORS NOT LEARNED
 CUDA_VISIBLE_DEVICES=2 python main.py --model RON --task CIFAR10 --archi 3072 512 512 10 --optim sgd --lrs 0.1 0.1 0.1 --mmt 0.9 --epochs 20 --act my_hard_sig --T1 250 --T2 30 --mbs 128 --alg EP --betas 0.0 0.5 --loss mse --gamma_min 1 --gamma_max 2 --eps_min 1 --eps_max 2 --tau 0.7
