import numpy as np
import torch

from scipy.ndimage import rotate, shift

class FGSMAttack:
    def __init__(self, args, test_data, model):
        self.args = args
        self.eps = float(args.attack_args)
        self.test_data = test_data
        self.model = model

    def perturb(self, x_batch, x_grads, inds):
        grad_sign = x_grads.sign()
        x_adv = x_batch.detach()
        x_adv[inds] += self.eps * grad_sign[inds]
        return x_adv

    def __call__(self):
        num_correct = num_examples = 0
        for x_batch, y_batch in self.test_data:
            x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
            x_batch.requires_grad = True
            pred = self.model(x_batch)
            _, pred_ind = pred.max(1)
            is_correct = pred_ind.eq(y_batch)
            loss = torch.nn.CrossEntropyLoss()(pred, y_batch)
            self.model.zero_grad()
            loss.backward()
            x_grad = x_batch.grad.data
            # Only perturb correct entries
            x_adv = self.perturb(x_batch, x_grad, is_correct)
            pred = self.model(x_adv)
            _, pred_ind = pred.max(1)
            is_correct = pred_ind.eq(y_batch)
            num_correct += int(is_correct.sum())
            num_examples += y_batch.size(0)
        acc = num_correct / num_examples
        save_dir = f'train/{self.args.exp_name}_{self.args.dataset_name}_{self.args.seed}/'
        with open(save_dir + 'result.txt', 'w') as f:
            f.write(f'{self.eps},{acc:.3f}')
        print(f'eps {self.eps}, acc {acc:.3f}')

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

    def __call__(self):
        num_correct = num_examples = 0
        for x_batch, y_batch in self.test_data:
            x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
            pred = self.model(x_batch)
            _, pred_ind = pred.max(1)
            is_correct = pred_ind.eq(y_batch)
            x_adv = []
            y_adv = []
            # Iterate one entry at a time, and perturb correct entries
            for i, (x_entry, y_entry) in enumerate(zip(x_batch, y_batch)):
                if is_correct[i] == 0:
                    continue
                x_entry, y_entry = x_entry.cpu().numpy(), y_entry.cpu().numpy()
                x_adv.append(self.perturb(x_entry, y_entry))
                y_adv.append(y_entry[np.newaxis])
            x_adv = torch.tensor(np.concatenate(x_adv)).cuda()
            y_adv = torch.tensor(np.concatenate(y_adv)).cuda()
            pred = self.model(x_adv)
            _, pred_ind = pred.max(1)
            is_correct = pred_ind.eq(y_adv)
            num_correct += int(is_correct.sum())
            num_examples += y_batch.size(0)
        acc = num_correct / num_examples
        save_dir = f'train/{self.args.exp_name}_{self.args.dataset_name}_{self.args.seed}/'
        with open(save_dir + 'result.txt', 'w') as f:
            f.write(f'{self.k},{acc:.3f}')
        print(f'k {self.k}, acc {acc:.3f}')