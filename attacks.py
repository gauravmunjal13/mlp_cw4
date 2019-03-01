import numpy as np
import torch

from scipy.ndimage import rotate, shift

class FGSMAttack:
    def __init__(self, args, model):
        self.args = args
        self.eps = float(args.attack_args)
        self.model = model

    def __call__(self, x_batch, y_batch):
        x_batch.requires_grad = True
        pred = self.model(x_batch)
        _, pred_ind = pred.max(1)
        is_correct = pred_ind.eq(y_batch)
        loss = torch.nn.CrossEntropyLoss()(pred, y_batch)
        self.model.zero_grad()
        loss.backward()
        x_grad = x_batch.grad.data
        grad_sign = x_grad.sign()
        x_adv = x_batch.detach()
        x_adv[is_correct] += self.eps * grad_sign[is_correct]
        return x_adv

class SpatialAttack:
    def __init__(self, args, test_data, model):
        self.args = args
        self.rng = np.random.RandomState(args.seed)
        self.k = int(args.attack_args)
        self.test_data = test_data
        self.model = model

    def perturb(self, x_entry, y_entry):
        # Set up rotation and translation grid
        rot_degrees_range = np.linspace(-30, 30, 31)
        h_trans_range = np.linspace(-3, 3, 5)
        v_trans_range = np.linspace(-3, 3, 5)
        choices = []
        # Sample k tuples
        for _ in range(self.k):
            rot_degrees = self.rng.choice(rot_degrees_range, 1)
            h_trans = self.rng.choice(h_trans_range, 1)
            v_trans = self.rng.choice(v_trans_range, 1)
            choices.append((rot_degrees, h_trans, v_trans))
        worst_x_adv = None
        worst_loss = -np.inf
        # Find the tuple with worst loss, and return the perturbed input
        for rot_degrees, h_trans, v_trans in choices:
            x_adv = rotate(x_entry, rot_degrees, axes=(-2, -1), reshape=False)
            x_adv = shift(x_adv, (0, h_trans, v_trans))
            pred = self.model(torch.tensor(x_adv[np.newaxis]).cuda())
            loss = torch.nn.CrossEntropyLoss()(pred, torch.tensor(y_entry[np.newaxis]).cuda())
            if loss > worst_loss:
                worst_loss = loss
                worst_x_adv = x_adv
        return worst_x_adv[np.newaxis]

    def __call__(self, x_batch, y_batch):
        pred = self.model(x_batch)
        _, pred_ind = pred.max(1)
        is_correct = pred_ind.eq(y_batch)
        x_adv_batch = []
        y_adv_batch = []
        # Iterate one entry at a time, and perturb correct entries
        for i, (x_entry, y_entry) in enumerate(zip(x_batch, y_batch)):
            if is_correct[i] == 0:
                continue
            x_entry, y_entry = x_entry.cpu().numpy(), y_entry.cpu().numpy()
            x_adv_batch.append(self.perturb(x_entry, y_entry))
            y_adv_batch.append(y_entry[np.newaxis])
        x_adv_batch = torch.tensor(np.concatenate(x_adv_batch)).cuda()
        y_adv_batch = torch.tensor(np.concatenate(y_adv_batch)).cuda()
        return x_adv_batch, y_adv_batch