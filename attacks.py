import numpy as np
import torch

from scipy.ndimage import rotate, shift

def fgsm_attack(x_batch, x_grads, epsilon, is_correct):
    grad_sign = x_grads.sign()
    inds = (is_correct == 1).astype('int')
    x_adv = x_batch.detach()
    x_adv[inds] += epsilon * grad_sign[inds]
    return x_adv

def spatial_attack(model, rng, k, x_entry, y_entry):
    rot_degrees_range = np.linspace(-30, 30, 31)
    h_trans_range = np.linspace(-3, 3, 5)
    v_trans_range = np.linspace(-3, 3, 5)
    choices = []
    for _ in range(k):
        rot_degrees = rng.choice(rot_degrees_range, 1)
        h_trans = rng.choice(h_trans_range, 1)
        v_trans = rng.choice(v_trans_range, 1)
        choices.append((rot_degrees, h_trans, v_trans))
    worst_x_adv = None
    worst_loss = -np.inf
    for rot_degrees, h_trans, v_trans in choices:
        x_adv = rotate(x_entry, rot_degrees, axes=(-2, -1), reshape=False)
        x_adv = shift(x_adv, (0, h_trans, v_trans))
        pred = model(x_adv)
        loss = torch.nn.CrossEntropyLoss()(pred, y_entry)
        if loss > worst_loss:
            worst_loss = loss
            worst_x_adv = x_adv
    return worst_x_adv[np.newaxis]