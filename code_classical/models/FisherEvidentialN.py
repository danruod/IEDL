import numpy as np
import torch
from torch import nn

from architectures.linear_sequential import linear_sequential
from architectures.convolution_linear_sequential import convolution_linear_sequential
from architectures.vgg_sequential import vgg16_bn

class FisherEvidentialN(nn.Module):
    def __init__(self,
                 input_dims,  # Input dimension. list of ints
                 output_dim,  # Output dimension. int
                 hidden_dims=[64, 64, 64],  # Hidden dimensions. list of ints
                 kernel_dim=None,  # Kernel dimension if conv architecture. int
                 architecture='linear',  # Encoder architecture name. int
                 k_lipschitz=None,  # Lipschitz constant. float or None (if no lipschitz)
                 batch_size=64,  # Batch size. int
                 lr=1e-3,  # Learning rate. float
                 loss='IEDL',  # Loss name. string
                 clf_type='softplus',
                 # target_con=1.0,  # target
                 # kl_c=-1.0,
                 fisher_c=1.0,
                 seed=123):  # Random seed for init. int
        super().__init__()

        torch.cuda.manual_seed(seed)
        torch.set_default_tensor_type(torch.FloatTensor)

        # Architecture parameters
        self.input_dims, self.output_dim, self.hidden_dims, self.kernel_dim = input_dims, output_dim, hidden_dims, kernel_dim
        self.k_lipschitz = k_lipschitz
        # Training parameters
        self.batch_size, self.lr = batch_size, lr
        self.loss = loss

        # self.target_con = target_con
        # self.kl_c = kl_c
        self.target_con = 1.0
        self.kl_c = -1.0
        self.fisher_c = fisher_c

        self.loss_mse_ = torch.tensor(0.0)
        self.loss_var_ = torch.tensor(0.0)
        self.loss_kl_ = torch.tensor(0.0)
        self.loss_fisher_ = torch.tensor(0.0)

        # Feature selection
        if architecture == 'linear':
            self.sequential = linear_sequential(input_dims=self.input_dims,
                                                hidden_dims=self.hidden_dims,
                                                output_dim=self.output_dim,
                                                k_lipschitz=self.k_lipschitz)
        elif architecture == 'conv':
            assert len(input_dims) == 3
            self.sequential = convolution_linear_sequential(input_dims=self.input_dims,
                                                            linear_hidden_dims=self.hidden_dims,
                                                            conv_hidden_dims=[64, 64, 64],
                                                            output_dim=self.output_dim,
                                                            kernel_dim=self.kernel_dim,
                                                            k_lipschitz=self.k_lipschitz)
        elif architecture == 'vgg':
            assert len(input_dims) == 3
            self.sequential = vgg16_bn(output_dim=self.output_dim, k_lipschitz=self.k_lipschitz)
        else:
            raise NotImplementedError

        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()
        self.clf_type = clf_type

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)


    def forward(self, input, labels_=None, return_output='alpha', compute_loss=False, epoch=10.):
        assert not (labels_ is None and compute_loss)

        # Forward
        if self.clf_type == "exp":
            logits = self.sequential(input)
            evi_alp_ = torch.exp(logits) + 1.0
        elif self.clf_type == "softplus":
            logits = self.sequential(input)
            evi_alp_ = self.softplus(logits) + 1.0
        else:
            raise NotImplementedError

        # Calculate loss
        if compute_loss:
            labels_1hot_ = torch.zeros_like(logits).scatter_(-1, labels_.unsqueeze(-1), 1)
            if self.loss == 'IEDL':
                # IEDL -> fisher_mse
                self.loss_mse_, self.loss_var_, self.loss_fisher_ = self.compute_fisher_mse(labels_1hot_, evi_alp_)
            elif self.loss == 'EDL':
                # EDL -> mse
                self.loss_mse_, self.loss_var_ = self.compute_mse(labels_1hot_, evi_alp_)
            elif self.loss == 'DEDL':
                self.loss_mse_, self.loss_var_ = self.compute_mse(labels_1hot_, evi_alp_)
                _, _, self.loss_fisher_ = self.compute_fisher_mse(labels_1hot_, evi_alp_)
            else:
                raise NotImplementedError

            evi_alp_ = (evi_alp_ - self.target_con) * (1 - labels_1hot_) + self.target_con
            self.loss_kl_ = self.compute_kl_loss(evi_alp_, labels_, self.target_con)

            if self.kl_c == -1:
                regr = np.minimum(1.0, epoch / 10.)
                self.grad_loss = self.loss_mse_ + self.loss_var_ + self.fisher_c * self.loss_fisher_ + regr * self.loss_kl_
            else:
                self.grad_loss = self.loss_mse_ + self.loss_var_ + self.fisher_c * self.loss_fisher_ + self.kl_c * self.loss_kl_

        if return_output == 'hard':
            # return max(logits)
            return self.predict(logits)
        elif return_output == 'soft':
            # return softmax(logits)
            return self.softmax(logits)
        elif return_output == 'alpha':
            return evi_alp_
        else:
            raise AssertionError


    def compute_mse(self, labels_1hot_, evi_alp_):
        evi_alp0_ = torch.sum(evi_alp_, dim=-1, keepdim=True)

        loss_mse_ = (labels_1hot_ - evi_alp_ / evi_alp0_).pow(2).sum(-1).mean()
        loss_var_ = (evi_alp_ * (evi_alp0_ - evi_alp_) / (evi_alp0_ * evi_alp0_ * (evi_alp0_ + 1))).sum(
            -1).mean()

        return loss_mse_, loss_var_

    def compute_fisher_mse(self, labels_1hot_, evi_alp_):
        evi_alp0_ = torch.sum(evi_alp_, dim=-1, keepdim=True)

        gamma1_alp = torch.polygamma(1, evi_alp_)
        gamma1_alp0 = torch.polygamma(1, evi_alp0_)

        gap = labels_1hot_ - evi_alp_ / evi_alp0_

        loss_mse_ = (gap.pow(2) * gamma1_alp).sum(-1).mean()

        loss_var_ = (evi_alp_ * (evi_alp0_ - evi_alp_) * gamma1_alp / (evi_alp0_ * evi_alp0_ * (evi_alp0_ + 1))).sum(-1).mean()

        loss_det_fisher_ = - (torch.log(gamma1_alp).sum(-1) + torch.log(1.0 - (gamma1_alp0 / gamma1_alp).sum(-1))).mean()

        return loss_mse_, loss_var_, loss_det_fisher_

    def compute_kl_loss(self, alphas, labels, target_concentration, concentration=1.0, epsilon=1e-8):
        # TODO: Need to make sure this actually works right...
        # todo: so that concentration is either fixed, or on a per-example setup

        # Create array of target (desired) concentration parameters
        if target_concentration < 1.0:
            concentration = target_concentration

        target_alphas = torch.ones_like(alphas) * concentration
        target_alphas += torch.zeros_like(alphas).scatter_(-1, labels.unsqueeze(-1), target_concentration - 1)

        alp0 = torch.sum(alphas, dim=-1, keepdim=True)
        target_alp0 = torch.sum(target_alphas, dim=-1, keepdim=True)

        alp0_term = torch.lgamma(alp0 + epsilon) - torch.lgamma(target_alp0 + epsilon)
        alp0_term = torch.where(torch.isfinite(alp0_term), alp0_term, torch.zeros_like(alp0_term))
        assert torch.all(torch.isfinite(alp0_term)).item()

        alphas_term = torch.sum(torch.lgamma(target_alphas + epsilon) - torch.lgamma(alphas + epsilon)
                                + (alphas - target_alphas) * (torch.digamma(alphas + epsilon) -
                                                              torch.digamma(alp0 + epsilon)), dim=-1, keepdim=True)
        alphas_term = torch.where(torch.isfinite(alphas_term), alphas_term, torch.zeros_like(alphas_term))
        assert torch.all(torch.isfinite(alphas_term)).item()

        loss = torch.squeeze(alp0_term + alphas_term).mean()

        return loss

    def step(self):
        self.optimizer.zero_grad()
        self.grad_loss.backward()
        self.optimizer.step()

    def predict(self, p):
        output_pred = torch.max(p, dim=-1)[1]
        return output_pred
