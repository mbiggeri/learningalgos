"""
Adapted from
https://github.com/Laborieux-Axel/Equilibrium-Propagation/blob/master/model_utils.py
"""


import torch
import numpy as np
import torch.nn.functional as F
import math

def my_sigmoid(x):
    return 1 / (1 + torch.exp(-4 * (x - 0.5)))


def hard_sigmoid(x):
    return (1 + F.hardtanh(2 * x - 1)) * 0.5


def ctrd_hard_sig(x):
    return (F.hardtanh(2 * x)) * 0.5


def my_hard_sig(x):
    return (1 + F.hardtanh(x - 1)) * 0.5


def copy(neurons):
    copy = []
    for n in neurons:
        copy.append(torch.empty_like(n).copy_(n.data).requires_grad_())
    return copy


def make_pools(letters):
    pools = []
    for p in range(len(letters)):
        if letters[p] == 'm':
            pools.append(torch.nn.MaxPool2d(2, stride=2))
        elif letters[p] == 'a':
            pools.append(torch.nn.AvgPool2d(2, stride=2))
        elif letters[p] == 'i':
            pools.append(torch.nn.Identity())
    return pools


def my_init(scale):
    def my_scaled_init(m):
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_uniform_(m.weight, math.sqrt(5))
            m.weight.data.mul_(scale)
            if m.bias is not None:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                torch.nn.init.uniform_(m.bias, -bound, bound)
                m.bias.data.mul_(scale)
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight, math.sqrt(5))
            m.weight.data.mul_(scale)
            if m.bias is not None:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                torch.nn.init.uniform_(m.bias, -bound, bound)
                m.bias.data.mul_(scale)

    return my_scaled_init


# Multi-Layer Perceptron

class P_MLP(torch.nn.Module):
    def __init__(self, archi, activation=torch.tanh):
        super(P_MLP, self).__init__()

        self.activation = activation
        self.archi = archi
        self.softmax = False  # Softmax readout is only defined for CNN and VFCNN
        self.nc = self.archi[-1]

        # Synapses
        self.synapses = torch.nn.ModuleList()
        for idx in range(len(archi) - 1):
            self.synapses.append(torch.nn.Linear(archi[idx], archi[idx + 1], bias=True))

    def Phi(self, x, y, neurons, beta, criterion):
        # Computes the primitive function given static input x, label y, neurons is the sequence of hidden layers neurons
        # criterion is the loss
        x = x.view(x.size(0), -1)  # flattening the input

        layers = [x] + neurons  # concatenate the input to other layers

        # Primitive function computation
        phi = 0.0
        for idx in range(len(self.synapses)):
            phi += torch.sum(self.synapses[idx](layers[idx]) * layers[idx + 1],
                             dim=1).squeeze()  # Scalar product s_n.W.s_n-1

        if beta != 0.0:  # Nudging the output layer when beta is non zero
            if criterion.__class__.__name__.find('MSE') != -1:
                y = F.one_hot(y, num_classes=self.nc)
                L = 0.5 * criterion(layers[-1].float(), y.float()).sum(dim=1).squeeze()
            else:
                L = criterion(layers[-1].float(), y).squeeze()
            phi -= beta * L

        return phi

    def forward(self, x, y, neurons, T, beta=0.0, criterion=torch.nn.MSELoss(reduction='none')):
        # Run T steps of the dynamics for static input x, label y, neurons and nudging factor beta.
        not_mse = (criterion.__class__.__name__.find('MSE') == -1)
        mbs = x.size(0)
        device = x.device

        for t in range(T):
            phi = self.Phi(x, y, neurons, beta, criterion)  # Computing Phi
            init_grads = torch.tensor([1 for i in range(mbs)], dtype=torch.float, device=device,
                                      requires_grad=True)  # Initializing gradients
            grads = torch.autograd.grad(phi, neurons, grad_outputs=init_grads)  # dPhi/ds

            for idx in range(len(neurons) - 1):
                neurons[idx] = self.activation(grads[idx])  # s_(t+1) = sigma( dPhi/ds )
                neurons[idx].requires_grad = True

            if not_mse:
                neurons[-1] = grads[-1]
            else:
                neurons[-1] = self.activation(grads[-1])

            neurons[-1].requires_grad = True

        return neurons

    def init_neurons(self, mbs, device):
        # Initializing the neurons
        neurons = []
        append = neurons.append
        for size in self.archi[1:]:
            append(torch.zeros((mbs, size), requires_grad=True, device=device))
        return neurons

    def compute_syn_grads(self, x, y, neurons_1, neurons_2, betas, criterion):
        # Computing the EP update given two steady states neurons_1 and neurons_2, static input x, label y
        beta_1, beta_2 = betas

        self.zero_grad()  # p.grad is zero
        phi_1 = self.Phi(x, y, neurons_1, beta_1, criterion)
        phi_1 = phi_1.mean()

        phi_2 = self.Phi(x, y, neurons_2, beta_2, criterion)
        phi_2 = phi_2.mean()

        delta_phi = (phi_2 - phi_1) / (beta_1 - beta_2)
        delta_phi.backward()  # p.grad = -(d_Phi_2/dp - d_Phi_1/dp)/(beta_2 - beta_1) ----> dL/dp  by the theorem


class RON(torch.nn.Module):
    def __init__(self, archi, device, activation=torch.tanh, tau=1, epsilon_min=0, epsilon_max=1, gamma_min=0, gamma_max=1, learn_oscillators=True):
        super(RON, self).__init__()

        self.activation = activation
        self.archi = archi
        self.softmax = False
        self.same_update = False
        self.nc = self.archi[-1]
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.tau = tau
        print("learn oscillator = ", learn_oscillators)
        self.learn_oscillators = learn_oscillators
        self.device = device

        self.gamma = torch.rand(archi[1], device=device) * (gamma_max - gamma_min) + gamma_min
        self.epsilon = torch.rand(archi[1], device=device) * (epsilon_max - epsilon_min) + epsilon_min
        self.gamma = torch.nn.Parameter(self.gamma, requires_grad=learn_oscillators)
        self.epsilon = torch.nn.Parameter(self.epsilon, requires_grad=learn_oscillators)
        assert len(archi) > 2, "The architecture must have at least 1 hidden layer"
        assert all([archi[1] == a for a in archi[2:-1]]), "The hidden layers must have the same number of neurons"

        # Synapses
        self.synapses = torch.nn.ModuleList()
        for idx in range(len(archi) - 1):
            self.synapses.append(torch.nn.Linear(archi[idx], archi[idx + 1], bias=True))

    def Phi_statez(self, x, y, neuronsy, beta, criterion):
        x = x.view(x.size(0), -1)

        layersy = [x] + neuronsy

        phi = 0.0
        for idx in range(len(self.synapses)):
            phi += torch.sum(self.synapses[idx](layersy[idx]) * layersy[idx + 1],
                             dim=1).squeeze()

        if beta != 0.0:
            if criterion.__class__.__name__.find('MSE') != -1:
                y = F.one_hot(y, num_classes=self.nc)
                L = 0.5 * criterion(layersy[-1].float(), y.float()).sum(dim=1).squeeze()
            else:
                L = criterion(layersy[-1].float(), y).squeeze()
            phi -= beta * L

        return phi

    def Phi_statey(self, neuronsz, neuronsy):
        phi = 0.0
        for idx in range(len(neuronsz)):
            phi += 0.5 * (torch.einsum('ij,ij->i', neuronsy[idx], neuronsy[idx]) +
                          self.tau * torch.einsum('ij,ij->i', neuronsz[idx], neuronsz[idx]))
        return phi

    def Phi(self, x, y, neuronsz, neuronsy, beta, criterion):
        x = x.view(x.size(0), -1)
        layersz = [x] + neuronsz
        layersy = [x] + neuronsy

        phi = torch.sum(0.5 * self.tau * self.synapses[0](x) * layersy[1], dim=1).squeeze()
        for idx in range(1, len(self.synapses) - 1):
            phiz = ((-0.5 * torch.einsum('ij,ij->i', torch.einsum('ij,jj->ij', layersz[idx], torch.diag(self.epsilon).to(self.device)), layersz[idx]))
                    + (-0.5 * torch.einsum('ij,ij->i', torch.einsum('ij,jj->ij', layersy[idx], torch.diag(self.gamma).to(self.device)), layersy[idx]))
                    + (0.5 * torch.einsum('ij,ij->i', layersz[idx], layersz[idx]))
                    + (self.tau * torch.sum(self.synapses[idx](layersy[idx]) * layersy[idx+1], dim=1).squeeze()))

            phi += 0.5 * (torch.einsum('ij,ij->i', layersy[idx], layersy[idx]) + self.tau * phiz)
        phi += torch.sum(0.5 * self.tau * self.synapses[-1](layersy[-2]) * layersy[-1], dim=1).squeeze()

        if beta != 0.0:
            if criterion.__class__.__name__.find('MSE') != -1:
                y = F.one_hot(y, num_classes=self.nc)
                L = 0.5 * criterion(layersy[-1].float(), y.float()).sum(dim=1).squeeze()
            else:
                L = criterion(layersy[-1].float(), y).squeeze()
            phi -= beta * L

        return phi

    def forward(self, x, y, neuronsz, neuronsy, T, beta=0.0, criterion=torch.nn.MSELoss(reduction='none')):
        # Run T steps of the dynamics for static input x, label y, neurons and nudging factor beta.
        not_mse = (criterion.__class__.__name__.find('MSE') == -1)
        mbs = x.size(0)
        device = x.device

        for t in range(T):
            phi = self.Phi_statez(x, y, neuronsy, beta, criterion)
            init_grads = torch.tensor([1 for i in range(mbs)], dtype=torch.float, device=device,
                                      requires_grad=True)
            grads = torch.autograd.grad(phi, neuronsy, grad_outputs=init_grads)

            for idx in range(len(neuronsz)):
                oscillator = neuronsz[idx] - self.tau * self.epsilon * neuronsz[idx] - self.tau * self.gamma * neuronsy[idx]
                neuronsz[idx] = (self.activation(grads[idx]) * self.tau + oscillator).detach()
                neuronsz[idx].requires_grad = True

            if not_mse:
                neuronsy[-1] = grads[-1]
            else:
                neuronsy[-1] = self.activation(grads[-1])
            neuronsy[-1].requires_grad = True

            phi = self.Phi_statey(neuronsz, neuronsy)
            init_grads = torch.tensor([1 for i in range(mbs)], dtype=torch.float, device=device,
                                      requires_grad=True)
            gradsz = torch.autograd.grad(phi, neuronsz, grad_outputs=init_grads, retain_graph=True)
            gradsy = torch.autograd.grad(phi, neuronsy[:-1], grad_outputs=init_grads)
            grads = [gz + gy for gz, gy in zip(gradsz, gradsy)]

            for idx in range(len(neuronsy) - 1):
                neuronsy[idx] = grads[idx]
                neuronsy[idx].requires_grad = True
        return neuronsz, neuronsy

    def init_neurons(self, mbs, device):
        # Initializing the neurons
        neuronsz, neuronsy = [], []
        for size in self.archi[1:-1]:
            neuronsz.append(torch.zeros(mbs, size, requires_grad=True, device=device))
            neuronsy.append(torch.zeros(mbs, size, requires_grad=True, device=device))
        neuronsy.append(torch.zeros(mbs, self.archi[-1], requires_grad=True, device=device))
        return neuronsz, neuronsy

    def compute_syn_grads(self, x, y, neurons_1, neurons_2, betas, criterion):
        # Computing the EP update given two steady states neurons_1 and neurons_2, static input x, label y
        beta_1, beta_2 = betas
        neurons_1z, neurons_1y = neurons_1
        neurons_2z, neurons_2y = neurons_2

        self.zero_grad()  # p.grad is zero
        phi_1 = self.Phi(x, y, neurons_1z, neurons_1y, beta_1, criterion)
        phi_1 = phi_1.mean()

        phi_2 = self.Phi(x, y, neurons_2z, neurons_2y, beta_2, criterion)
        phi_2 = phi_2.mean()

        delta_phi = (phi_2 - phi_1) / (beta_1 - beta_2)
        delta_phi.backward()  # p.grad = -(d_Phi_2/dp - d_Phi_1/dp)/(beta_2 - beta_1) ----> dL/dp  by the theorem


# Vector Field Multi-Layer Perceptron

class VF_MLP(torch.nn.Module):
    def __init__(self, archi, activation=torch.tanh):
        super(VF_MLP, self).__init__()

        self.activation = activation
        self.archi = archi
        self.softmax = False
        self.nc = self.archi[-1]

        # Forward synapses
        self.synapses = torch.nn.ModuleList()
        for idx in range(len(archi) - 1):
            self.synapses.append(torch.nn.Linear(archi[idx], archi[idx + 1], bias=True))

        # Backward synapses
        self.B_syn = torch.nn.ModuleList()
        for idx in range(1, len(archi) - 1):
            self.B_syn.append(torch.nn.Linear(archi[idx + 1], archi[idx], bias=False))

    def Phi(self, x, y, neurons, beta, criterion, neurons_2=None):
        # For assymetric connections each layer has its own phi
        x = x.view(x.size(0), -1)

        layers = [x] + neurons

        phis = []

        if neurons_2 is None:  # Standard case for the dynamics
            for idx in range(len(self.synapses) - 1):
                phi = torch.sum(self.synapses[idx](layers[idx]) * layers[idx + 1], dim=1).squeeze()
                phi += torch.sum(self.B_syn[idx](layers[idx + 2]) * layers[idx + 1], dim=1).squeeze()
                phis.append(phi)

            phi = torch.sum(self.synapses[-1](layers[-2]) * layers[-1], dim=1).squeeze()
            if beta != 0.0:
                if criterion.__class__.__name__.find('MSE') != -1:
                    y = F.one_hot(y, num_classes=self.nc)
                    L = 0.5 * criterion(layers[-1].float(), y.float()).sum(dim=1).squeeze()
                else:
                    L = criterion(layers[-1].float(), y).squeeze()
                phi -= beta * L

            phis.append(phi)

        else:  # Used only for computing the vector field EP update
            layers_2 = [x] + neurons_2
            for idx in range(len(self.synapses) - 1):
                phi = torch.sum(self.synapses[idx](layers[idx]) * layers_2[idx + 1], dim=1).squeeze()
                phi += torch.sum(self.B_syn[idx](layers[idx + 2]) * layers_2[idx + 1], dim=1).squeeze()
                phis.append(phi)

            phi = torch.sum(self.synapses[-1](layers[-2]) * layers_2[-1], dim=1).squeeze()
            if beta != 0.0:
                if criterion.__class__.__name__.find('MSE') != -1:
                    y = F.one_hot(y, num_classes=self.nc)
                    L = 0.5 * criterion(layers_2[-1].float(), y.float()).sum(dim=1).squeeze()
                else:
                    L = criterion(layers_2[-1].float(), y).squeeze()
                phi -= beta * L

            phis.append(phi)

        return phis

    def forward(self, x, y, neurons, T, beta=0.0, criterion=torch.nn.MSELoss(reduction='none')):

        not_mse = (criterion.__class__.__name__.find('MSE') == -1)
        mbs = x.size(0)
        device = x.device

        for t in range(T):
            phis = self.Phi(x, y, neurons, beta, criterion)
            for idx in range(len(neurons) - 1):
                init_grad = torch.tensor([1 for i in range(mbs)], dtype=torch.float, device=device, requires_grad=True)
                grad = torch.autograd.grad(phis[idx], neurons[idx], grad_outputs=init_grad)

                neurons[idx] = self.activation(grad[0])
                neurons[idx].requires_grad = True

            init_grad = torch.tensor([1 for i in range(mbs)], dtype=torch.float, device=device, requires_grad=True)
            grad = torch.autograd.grad(phis[-1], neurons[-1], grad_outputs=init_grad)
            if not_mse:
                neurons[-1] = grad[0]
            else:
                neurons[-1] = self.activation(grad[0])

            neurons[-1].requires_grad = True

        return neurons

    def init_neurons(self, mbs, device):

        neurons = []
        append = neurons.append
        for size in self.archi[1:]:
            append(torch.zeros((mbs, size), requires_grad=True, device=device))
        return neurons

    def compute_syn_grads(self, x, y, neurons_1, neurons_2, betas, criterion):

        beta_1, beta_2 = betas

        self.zero_grad()  # p.grad is zero
        phis_1 = self.Phi(x, y, neurons_1, beta_1, criterion)

        phis_2 = self.Phi(x, y, neurons_1, beta_2, criterion, neurons_2=neurons_2)

        for idx in range(len(neurons_1)):
            phi_1 = phis_1[idx].mean()
            phi_2 = phis_2[idx].mean()
            delta_phi = (phi_2 - phi_1) / (beta_1 - beta_2)
            delta_phi.backward()  # p.grad = -(d_Phi_2/dp - d_Phi_1/dp)/(beta_2 - beta_1)


# Convolutional Neural Network

class P_CNN(torch.nn.Module):
    def __init__(self, in_size, channels, kernels, strides, fc, pools, paddings, activation=hard_sigmoid,
                 softmax=False):
        super(P_CNN, self).__init__()

        # Dimensions used to initialize neurons
        self.in_size = in_size
        self.channels = channels
        self.kernels = kernels
        self.strides = strides
        self.paddings = paddings
        self.fc = fc
        self.nc = fc[-1]

        self.activation = activation
        self.pools = pools

        self.synapses = torch.nn.ModuleList()

        self.softmax = softmax  # whether to use softmax readout or not

        size = in_size  # size of the input : 32 for cifar10

        for idx in range(len(channels) - 1):
            self.synapses.append(torch.nn.Conv2d(channels[idx], channels[idx + 1], kernels[idx],
                                                 stride=strides[idx], padding=paddings[idx], bias=True))

            size = int((size + 2 * paddings[idx] - kernels[idx]) / strides[idx] + 1)  # size after conv
            if self.pools[idx].__class__.__name__.find('Pool') != -1:
                size = int((size - pools[idx].kernel_size) / pools[idx].stride + 1)  # size after Pool

        size = size * size * channels[-1]
        fc_layers = [size] + fc

        for idx in range(len(fc)):
            self.synapses.append(torch.nn.Linear(fc_layers[idx], fc_layers[idx + 1], bias=True))

    def Phi(self, x, y, neurons, beta, criterion):

        mbs = x.size(0)
        conv_len = len(self.kernels)
        tot_len = len(self.synapses)

        layers = [x] + neurons
        phi = 0.0

        # Phi computation changes depending on softmax == True or not
        if not self.softmax:
            for idx in range(conv_len):
                phi += torch.sum(self.pools[idx](self.synapses[idx](layers[idx])) * layers[idx + 1],
                                 dim=(1, 2, 3)).squeeze()
            for idx in range(conv_len, tot_len):
                phi += torch.sum(self.synapses[idx](layers[idx].view(mbs, -1)) * layers[idx + 1], dim=1).squeeze()

            if beta != 0.0:
                if criterion.__class__.__name__.find('MSE') != -1:
                    y = F.one_hot(y, num_classes=self.nc)
                    L = 0.5 * criterion(layers[-1].float(), y.float()).sum(dim=1).squeeze()
                else:
                    L = criterion(layers[-1].float(), y).squeeze()
                phi -= beta * L

        else:
            # the output layer used for the prediction is no longer part of the system ! Summing until len(self.synapses) - 1 only
            for idx in range(conv_len):
                phi += torch.sum(self.pools[idx](self.synapses[idx](layers[idx])) * layers[idx + 1],
                                 dim=(1, 2, 3)).squeeze()
            for idx in range(conv_len, tot_len - 1):
                phi += torch.sum(self.synapses[idx](layers[idx].view(mbs, -1)) * layers[idx + 1], dim=1).squeeze()

            # the prediction is made with softmax[last weights[penultimate layer]]
            if beta != 0.0:
                L = criterion(self.synapses[-1](layers[-1].view(mbs, -1)).float(), y).squeeze()
                phi -= beta * L

        return phi

    def forward(self, x, y, neurons, T, beta=0.0, criterion=torch.nn.MSELoss(reduction='none')):

        not_mse = (criterion.__class__.__name__.find('MSE') == -1)
        mbs = x.size(0)
        device = x.device

        for t in range(T):
            phi = self.Phi(x, y, neurons, beta, criterion)
            init_grads = torch.tensor([1 for i in range(mbs)], dtype=torch.float, device=device, requires_grad=True)
            grads = torch.autograd.grad(phi, neurons, grad_outputs=init_grads, create_graph=False)

            for idx in range(len(neurons) - 1):
                neurons[idx] = self.activation(grads[idx])
                neurons[idx].requires_grad = True

            if not_mse and not (self.softmax):
                neurons[-1] = grads[-1]
            else:
                neurons[-1] = self.activation(grads[-1])

            neurons[-1].requires_grad = True

        return neurons

    def init_neurons(self, mbs, device):

        neurons = []
        append = neurons.append
        size = self.in_size
        for idx in range(len(self.channels) - 1):
            size = int((size + 2 * self.paddings[idx] - self.kernels[idx]) / self.strides[idx] + 1)  # size after conv
            if self.pools[idx].__class__.__name__.find('Pool') != -1:
                size = int((size - self.pools[idx].kernel_size) / self.pools[idx].stride + 1)  # size after Pool
            append(torch.zeros((mbs, self.channels[idx + 1], size, size), requires_grad=True, device=device))

        size = size * size * self.channels[-1]

        if not self.softmax:
            for idx in range(len(self.fc)):
                append(torch.zeros((mbs, self.fc[idx]), requires_grad=True, device=device))
        else:
            # we *REMOVE* the output layer from the system
            for idx in range(len(self.fc) - 1):
                append(torch.zeros((mbs, self.fc[idx]), requires_grad=True, device=device))

        return neurons

    def compute_syn_grads(self, x, y, neurons_1, neurons_2, betas, criterion):

        beta_1, beta_2 = betas

        self.zero_grad()  # p.grad is zero
        phi_1 = self.Phi(x, y, neurons_1, beta_1, criterion)
        phi_1 = phi_1.mean()

        phi_2 = self.Phi(x, y, neurons_2, beta_2, criterion)
        phi_2 = phi_2.mean()

        delta_phi = (phi_2 - phi_1) / (beta_1 - beta_2)
        delta_phi.backward()  # p.grad = -(d_Phi_2/dp - d_Phi_1/dp)/(beta_2 - beta_1) ----> dL/dp  by the theorem


# Vector Field Convolutional Neural Network

class VF_CNN(torch.nn.Module):
    def __init__(self, in_size, channels, kernels, strides, fc, pools, paddings, activation=hard_sigmoid, softmax=False,
                 same_update=False):
        super(VF_CNN, self).__init__()

        # Dimensions used to initialize neurons
        self.in_size = in_size
        self.channels = channels
        self.kernels = kernels
        self.strides = strides
        self.paddings = paddings
        self.fc = fc
        self.nc = fc[-1]

        self.activation = activation
        self.pools = pools

        self.synapses = torch.nn.ModuleList()  # forward connections
        self.B_syn = torch.nn.ModuleList()  # backward connections

        self.same_update = same_update  # whether to use the same update for forward ans backward connections
        self.softmax = softmax  # whether to use softmax readout

        size = in_size

        for idx in range(len(channels) - 1):
            self.synapses.append(torch.nn.Conv2d(channels[idx], channels[idx + 1], kernels[idx],
                                                 stride=strides[idx], padding=paddings[idx], bias=True))

            if idx > 0:  # backward synapses except for first layer
                self.B_syn.append(torch.nn.Conv2d(channels[idx], channels[idx + 1], kernels[idx],
                                                  stride=strides[idx], padding=paddings[idx], bias=False))

            size = int((size + 2 * paddings[idx] - kernels[idx]) / strides[idx] + 1)  # size after conv
            if self.pools[idx].__class__.__name__.find('Pool') != -1:
                size = int((size - pools[idx].kernel_size) / pools[idx].stride + 1)  # size after Pool

        size = size * size * channels[-1]

        fc_layers = [size] + fc

        for idx in range(len(fc)):
            self.synapses.append(torch.nn.Linear(fc_layers[idx], fc_layers[idx + 1], bias=True))
            if not (self.softmax and (idx == (len(fc) - 1))):
                self.B_syn.append(torch.nn.Linear(fc_layers[idx], fc_layers[idx + 1], bias=False))

    def angle(self):  # computes the angle between forward and backward weights
        angles = []
        with torch.no_grad():
            for idx in range(len(self.B_syn)):
                fnorm = self.synapses[idx + 1].weight.data.pow(2).sum().pow(0.5).item()
                bnorm = self.B_syn[idx].weight.data.pow(2).sum().pow(0.5).item()
                cos = self.B_syn[idx].weight.data.mul(self.synapses[idx + 1].weight.data).sum().div(fnorm * bnorm)
                angle = torch.acos(cos).item() * (180 / (math.pi))
                angles.append(angle)
        return angles

    def Phi(self, x, y, neurons, beta, criterion, neurons_2=None):

        mbs = x.size(0)
        conv_len = len(self.kernels)
        tot_len = len(self.synapses)
        bck_len = len(self.B_syn)

        layers = [x] + neurons
        phis = []

        if neurons_2 is None:  # neurons 2 is not None only when computing the EP update for different updates between forward and backward
            # Phi computation changes depending on softmax == True or not
            if not self.softmax:

                for idx in range(conv_len - 1):
                    phi = torch.sum(self.pools[idx](self.synapses[idx](layers[idx])) * layers[idx + 1],
                                    dim=(1, 2, 3)).squeeze()
                    phi += torch.sum(self.pools[idx + 1](self.B_syn[idx](layers[idx + 1])) * layers[idx + 2],
                                     dim=(1, 2, 3)).squeeze()
                    phis.append(phi)

                phi = torch.sum(
                    self.pools[conv_len - 1](self.synapses[conv_len - 1](layers[conv_len - 1])) * layers[conv_len],
                    dim=(1, 2, 3)).squeeze()
                phi += torch.sum(self.B_syn[conv_len - 1](layers[conv_len].view(mbs, -1)) * layers[conv_len + 1],
                                 dim=1).squeeze()
                phis.append(phi)

                for idx in range(conv_len + 1, tot_len - 1):
                    phi = torch.sum(self.synapses[idx](layers[idx].view(mbs, -1)) * layers[idx + 1], dim=1).squeeze()
                    phi += torch.sum(self.B_syn[idx](layers[idx + 1].view(mbs, -1)) * layers[idx + 2], dim=1).squeeze()
                    phis.append(phi)

                phi = torch.sum(self.synapses[-1](layers[-2].view(mbs, -1)) * layers[-1], dim=1).squeeze()
                if beta != 0.0:
                    if criterion.__class__.__name__.find('MSE') != -1:
                        y = F.one_hot(y, num_classes=self.nc)
                        L = 0.5 * criterion(layers[-1].float(), y.float()).sum(dim=1).squeeze()
                    else:
                        L = criterion(layers[-1].float(), y).squeeze()
                    phi -= beta * L
                phis.append(phi)

            else:
                # the output layer used for the prediction is no longer part of the system ! Summing until len(self.synapses) - 1 only
                for idx in range(conv_len - 1):
                    phi = torch.sum(self.pools[idx](self.synapses[idx](layers[idx])) * layers[idx + 1],
                                    dim=(1, 2, 3)).squeeze()
                    phi += torch.sum(self.pools[idx + 1](self.B_syn[idx](layers[idx + 1])) * layers[idx + 2],
                                     dim=(1, 2, 3)).squeeze()
                    phis.append(phi)

                phi = torch.sum(
                    self.pools[conv_len - 1](self.synapses[conv_len - 1](layers[conv_len - 1])) * layers[conv_len],
                    dim=(1, 2, 3)).squeeze()
                if bck_len >= conv_len:
                    phi += torch.sum(self.B_syn[conv_len - 1](layers[conv_len].view(mbs, -1)) * layers[conv_len + 1],
                                     dim=1).squeeze()
                    phis.append(phi)

                    for idx in range(conv_len, tot_len - 2):
                        phi = torch.sum(self.synapses[idx](layers[idx].view(mbs, -1)) * layers[idx + 1],
                                        dim=1).squeeze()
                        phi += torch.sum(self.B_syn[idx](layers[idx + 1].view(mbs, -1)) * layers[idx + 2],
                                         dim=1).squeeze()
                        phis.append(phi)

                    phi = torch.sum(self.synapses[-2](layers[-2].view(mbs, -1)) * layers[-1], dim=1).squeeze()
                    if beta != 0.0:
                        L = criterion(self.synapses[-1](layers[-1].view(mbs, -1)).float(), y).squeeze()
                        phi -= beta * L
                    phis.append(phi)

                    # the prediction is made with softmax[last weights[penultimate layer]]
                elif beta != 0.0:
                    L = criterion(self.synapses[-1](layers[-1].view(mbs, -1)).float(), y).squeeze()
                    phi -= beta * L
                    phis.append(phi)
                else:
                    phis.append(phi)

        else:
            layers_2 = [x] + neurons_2
            # Phi computation changes depending on softmax == True or not
            if not self.softmax:

                for idx in range(conv_len - 1):
                    phi = torch.sum(self.pools[idx](self.synapses[idx](layers[idx])) * layers_2[idx + 1],
                                    dim=(1, 2, 3)).squeeze()
                    phi += torch.sum(self.pools[idx + 1](self.B_syn[idx](layers_2[idx + 1])) * layers[idx + 2],
                                     dim=(1, 2, 3)).squeeze()
                    phis.append(phi)

                phi = torch.sum(
                    self.pools[conv_len - 1](self.synapses[conv_len - 1](layers[conv_len - 1])) * layers_2[conv_len],
                    dim=(1, 2, 3)).squeeze()
                phi += torch.sum(self.B_syn[conv_len - 1](layers_2[conv_len].view(mbs, -1)) * layers[conv_len + 1],
                                 dim=1).squeeze()
                phis.append(phi)

                for idx in range(conv_len + 1, tot_len - 1):
                    phi = torch.sum(self.synapses[idx](layers[idx].view(mbs, -1)) * layers_2[idx + 1], dim=1).squeeze()
                    phi += torch.sum(self.B_syn[idx](layers_2[idx + 1].view(mbs, -1)) * layers[idx + 2],
                                     dim=1).squeeze()
                    phis.append(phi)

                phi = torch.sum(self.synapses[-1](layers[-2].view(mbs, -1)) * layers_2[-1], dim=1).squeeze()
                if beta != 0.0:
                    if criterion.__class__.__name__.find('MSE') != -1:
                        y = F.one_hot(y, num_classes=self.nc)
                        L = 0.5 * criterion(layers_2[-1].float(), y.float()).sum(dim=1).squeeze()
                    else:
                        L = criterion(layers_2[-1].float(), y).squeeze()
                    phi -= beta * L
                phis.append(phi)

            else:
                # the output layer used for the prediction is no longer part of the system ! Summing until len(self.synapses) - 1 only
                for idx in range(conv_len - 1):
                    phi = torch.sum(self.pools[idx](self.synapses[idx](layers[idx])) * layers_2[idx + 1],
                                    dim=(1, 2, 3)).squeeze()
                    phi += torch.sum(self.pools[idx + 1](self.B_syn[idx](layers_2[idx + 1])) * layers[idx + 2],
                                     dim=(1, 2, 3)).squeeze()
                    phis.append(phi)

                phi = torch.sum(
                    self.pools[conv_len - 1](self.synapses[conv_len - 1](layers[conv_len - 1])) * layers_2[conv_len],
                    dim=(1, 2, 3)).squeeze()
                if bck_len >= conv_len:
                    phi += torch.sum(self.B_syn[conv_len - 1](layers_2[conv_len].view(mbs, -1)) * layers[conv_len + 1],
                                     dim=1).squeeze()
                    phis.append(phi)

                    for idx in range(conv_len, tot_len - 2):
                        phi = torch.sum(self.synapses[idx](layers[idx].view(mbs, -1)) * layers_2[idx + 1],
                                        dim=1).squeeze()
                        phi += torch.sum(self.B_syn[idx](layers_2[idx + 1].view(mbs, -1)) * layers[idx + 2],
                                         dim=1).squeeze()
                        phis.append(phi)

                    phi = torch.sum(self.synapses[-2](layers[-2].view(mbs, -1)) * layers_2[-1], dim=1).squeeze()
                    if beta != 0.0:
                        L = criterion(self.synapses[-1](layers_2[-1].view(mbs, -1)).float(), y).squeeze()
                        phi -= beta * L
                    phis.append(phi)

                    # the prediction is made with softmax[last weights[penultimate layer]]
                elif beta != 0.0:
                    L = criterion(self.synapses[-1](layers_2[-1].view(mbs, -1)).float(), y).squeeze()
                    phi -= beta * L
                    phis.append(phi)
                else:
                    phis.append(phi)
        return phis

    def forward(self, x, y, neurons, T, beta=0.0, criterion=torch.nn.MSELoss(reduction='none')):

        not_mse = (criterion.__class__.__name__.find('MSE') == -1)
        mbs = x.size(0)
        device = x.device

        for t in range(T):
            phis = self.Phi(x, y, neurons, beta, criterion)
            for idx in range(len(neurons) - 1):
                init_grad = torch.tensor([1 for i in range(mbs)], dtype=torch.float, device=device,
                                         requires_grad=True)
                grad = torch.autograd.grad(phis[idx], neurons[idx], grad_outputs=init_grad, create_graph=False)

                neurons[idx] = self.activation(grad[0])
                neurons[idx].requires_grad = True

            init_grad = torch.tensor([1 for i in range(mbs)], dtype=torch.float, device=device, requires_grad=False)
            grad = torch.autograd.grad(phis[-1], neurons[-1], grad_outputs=init_grad, create_graph=False)
            if not_mse and not (self.softmax):
                neurons[-1] = grad[0]
            else:
                neurons[-1] = self.activation(grad[0])

            neurons[-1].requires_grad = True

        return neurons

    def init_neurons(self, mbs, device):

        neurons = []
        append = neurons.append
        size = self.in_size
        for idx in range(len(self.channels) - 1):
            size = int((size + 2 * self.paddings[idx] - self.kernels[idx]) / self.strides[idx] + 1)  # size after conv
            if self.pools[idx].__class__.__name__.find('Pool') != -1:
                size = int((size - self.pools[idx].kernel_size) / self.pools[idx].stride + 1)  # size after Pool
            append(torch.zeros((mbs, self.channels[idx + 1], size, size), requires_grad=True, device=device))

        size = size * size * self.channels[-1]

        if not self.softmax:
            for idx in range(len(self.fc)):
                append(torch.zeros((mbs, self.fc[idx]), requires_grad=True, device=device))
        else:
            # we remove the output layer from the system
            for idx in range(len(self.fc) - 1):
                append(torch.zeros((mbs, self.fc[idx]), requires_grad=True, device=device))

        return neurons

    def compute_syn_grads(self, x, y, neurons_1, neurons_2, betas, criterion, neurons_3=None):

        beta_1, beta_2 = betas

        self.zero_grad()  # p.grad is zero
        if neurons_3 is None:  # neurons_3 is not None only when doing thirdphase with old VF (same update False)
            phis_1 = self.Phi(x, y, neurons_1, beta_1, criterion)  # phi will habe the form s_* W s_*
        else:
            phis_1 = self.Phi(x, y, neurons_1, beta_1, criterion,
                              neurons_2=neurons_2)  # phi will have the form s_* W s_beta

        if self.same_update:
            phis_2 = self.Phi(x, y, neurons_2, beta_2, criterion)  # Phi = s_beta W s_beta
        else:
            if neurons_3 is None:
                phis_2 = self.Phi(x, y, neurons_1, beta_2, criterion, neurons_2=neurons_2)  # phi = s_* W s_beta
            else:
                phis_2 = self.Phi(x, y, neurons_1, beta_2, criterion, neurons_2=neurons_3)  # phi = s_* W s_-beta

        for idx in range(len(neurons_1)):
            phi_1 = phis_1[idx].mean()
            phi_2 = phis_2[idx].mean()
            delta_phi = ((phi_2 - phi_1) / (beta_1 - beta_2))
            delta_phi.backward()  # p.grad = -(d_Phi_2/dp - d_Phi_1/dp)/(beta_2 - beta_1)
        if self.same_update:
            with torch.no_grad():
                for idx in range(len(self.B_syn)):
                    common_update = 0.5 * (self.B_syn[idx].weight.grad.data + self.synapses[idx + 1].weight.grad.data)
                    self.B_syn[idx].weight.grad.data.copy_(common_update)
                    self.synapses[idx + 1].weight.grad.data.copy_(common_update)


def train_epoch(model, optimizer, epoch_number, train_loader, T1, T2, betas, device, criterion, alg='EP',
          random_sign=False, thirdphase=False, cep_debug=False, ron=False, id=None):
    mbs = train_loader.batch_size
    iter_per_epochs = math.ceil(len(train_loader.dataset) / mbs)
    beta_1, beta_2 = betas

    run_correct = 0
    run_total = 0
    model.train()

    for idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        # if alg=='CEP' and cep_debug:
        #    x = x.double()

        if ron:
            neuronsz, neuronsy = model.init_neurons(x.size(0), device)
        else:
            neurons = model.init_neurons(x.size(0), device)
        if alg == 'EP' or alg == 'CEP':
            # First phase
            if ron:
                neuronsz, neuronsy = model(x, y, neuronsz, neuronsy, T1, beta=beta_1, criterion=criterion)
                neurons_1 = (copy(neuronsz), copy(neuronsy))
                neurons = neuronsy
            else:
                neurons = model(x, y, neurons, T1, beta=beta_1, criterion=criterion)
                neurons_1 = copy(neurons)
        elif alg == 'BPTT':
            assert not ron, "RON not implemented for BPTT"
            neurons = model(x, y, neurons, T1 - T2, beta=0.0, criterion=criterion)
            # detach data and neurons from the graph
            x = x.detach()
            x.requires_grad = True
            for k in range(len(neurons)):
                neurons[k] = neurons[k].detach()
                neurons[k].requires_grad = True

            neurons = model(x, y, neurons, T2, beta=0.0, criterion=criterion)  # T2 time step

        # Predictions for running accuracy
        with torch.no_grad():
            if not model.softmax:
                pred = torch.argmax(neurons[-1], dim=1).squeeze()
            else:
                # WATCH OUT: prediction is different when softmax == True
                pred = torch.argmax(F.softmax(model.synapses[-1](neurons[-1].view(x.size(0), -1)), dim=1),
                                    dim=1).squeeze()

            run_correct += (y == pred).sum().item()
            run_total += x.size(0)

        if alg == 'EP':
            # Second phase
            if random_sign and (beta_1 == 0.0):
                rnd_sgn = 2 * np.random.randint(2) - 1
                betas = beta_1, rnd_sgn * beta_2
                beta_1, beta_2 = betas

            if ron:
                neuronsz, neuronsy = model(x, y, neuronsz, neuronsy, T2, beta=beta_2, criterion=criterion)
                neurons_2 = (copy(neuronsz), copy(neuronsy))
            else:
                neurons = model(x, y, neurons, T2, beta=beta_2, criterion=criterion)
                neurons_2 = copy(neurons)

            # Third phase (if we approximate f' as f'(x) = (f(x+h) - f(x-h))/2h)
            if thirdphase:
                if ron:
                    neuronsz, neuronsy = copy(neurons_1[0]), copy(neurons_1[1])
                    neuronsz, neuronsy = model(x, y, neuronsz, neuronsy, T2, beta=- beta_2, criterion=criterion)
                    neurons_3 = (copy(neuronsz), copy(neuronsy))
                else:
                # come back to the first equilibrium
                    neurons = copy(neurons_1)
                    neurons = model(x, y, neurons, T2, beta=- beta_2, criterion=criterion)
                    neurons_3 = copy(neurons)
                if not (isinstance(model, VF_CNN)):
                    model.compute_syn_grads(x, y, neurons_2, neurons_3, (beta_2, - beta_2), criterion)
                else:
                    if model.same_update:
                        model.compute_syn_grads(x, y, neurons_2, neurons_3, (beta_2, - beta_2), criterion)
                    else:
                        model.compute_syn_grads(x, y, neurons_1, neurons_2, (beta_2, - beta_2), criterion,
                                                neurons_3=neurons_3)
            else:
                model.compute_syn_grads(x, y, neurons_1, neurons_2, betas, criterion)

            optimizer.step()

        elif alg == 'CEP':
            if random_sign and (beta_1 == 0.0):
                rnd_sgn = 2 * np.random.randint(2) - 1
                betas = beta_1, rnd_sgn * beta_2
                beta_1, beta_2 = betas

            # second phase
            if cep_debug:
                prev_p = {}
                for (n, p) in model.named_parameters():
                    prev_p[n] = p.clone().detach()
                for i in range(len(model.synapses)):
                    prev_p['lrs' + str(i)] = optimizer.param_groups[i]['lr']
                    prev_p['wds' + str(i)] = optimizer.param_groups[i]['weight_decay']
                    optimizer.param_groups[i]['lr'] *= 6e-5
                    # optimizer.param_groups[i]['weight_decay'] = 0.0

            for k in range(T2):
                neurons = model(x, y, neurons, 1, beta=beta_2, criterion=criterion)  # one step
                neurons_2 = copy(neurons)
                model.compute_syn_grads(x, y, neurons_1, neurons_2, betas,
                                        criterion)  # compute cep update between 2 consecutive steps
                for (n, p) in model.named_parameters():
                    p.grad.data.div_((1 - optimizer.param_groups[int(n[9])]['lr'] *
                                      optimizer.param_groups[int(n[9])]['weight_decay']) ** (T2 - 1 - k))
                optimizer.step()  # update weights
                neurons_1 = copy(neurons)

            if thirdphase:
                neurons = model(x, y, neurons, T2, beta=0.0, criterion=criterion)  # come back to s*
                neurons_2 = copy(neurons)
                for k in range(T2):
                    neurons = model(x, y, neurons, 1, beta=-beta_2, criterion=criterion)
                    neurons_3 = copy(neurons)
                    model.compute_syn_grads(x, y, neurons_2, neurons_3, (beta_2, -beta_2), criterion)
                    optimizer.step()
                    neurons_2 = copy(neurons)

        elif alg == 'BPTT':
            assert not ron, "RON not implemented for BPTT"
            # final loss
            if criterion.__class__.__name__.find('MSE') != -1:
                loss = 0.5 * criterion(neurons[-1].float(), F.one_hot(y, num_classes=model.nc).float()).sum(
                    dim=1).mean().squeeze()
            else:
                if not model.softmax:
                    loss = criterion(neurons[-1].float(), y).mean().squeeze()
                else:
                    loss = criterion(model.synapses[-1](neurons[-1].view(x.size(0), -1)).float(),
                                     y).mean().squeeze()
            # setting gradients field to zero before backward
            model.zero_grad()

            # Backpropagation through time
            loss.backward()
            optimizer.step()
        if ((idx % (iter_per_epochs // 10) == 0) or (idx == iter_per_epochs - 1)):
            run_acc = run_correct / run_total
            if (id != None):
                # print("Trial ", id, '-> Epoch :', round(epoch_number + (idx / iter_per_epochs), 2),
                #   '\tRun train acc :', round(run_acc, 3), '\t(' + str(run_correct) + '/' + str(run_total) + ')')
                pass
            else:
                print('Epoch :', round(epoch_number + (idx / iter_per_epochs), 2),
                  '\tRun train acc :', round(run_acc, 3), '\t(' + str(run_correct) + '/' + str(run_total) + ')')


def evaluate(model, loader, T, device, ron=False):
    # Evaluate the model on a dataloader with T steps for the dynamics
    model.eval()
    correct = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if ron:
            neuronsz, neuronsy = model.init_neurons(x.size(0), device)
            neuronsz, neuronsy = model(x, y, neuronsz, neuronsy, T)  # dynamics for T time steps
            neurons = neuronsy
        else:
            neurons = model.init_neurons(x.size(0), device)
            neurons = model(x, y, neurons, T)  # dynamics for T time steps

        if not model.softmax:
            pred = torch.argmax(neurons[-1],
                                dim=1).squeeze()  # in this cas prediction is done directly on the last (output) layer of neurons
        else:  # prediction is done as a readout of the penultimate layer (output is not part of the system)
            pred = torch.argmax(F.softmax(model.synapses[-1](neurons[-1].view(x.size(0), -1)), dim=1), dim=1).squeeze()

        correct += (y == pred).sum().item()

    acc = correct / len(loader.dataset)
    return acc
