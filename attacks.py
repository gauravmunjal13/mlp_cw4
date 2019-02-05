def fgsm_attack(x_batch, x_grads, epsilon, is_correct):
    grad_sign = x_grads.sign()
    inds = (is_correct == 1).astype('int')
    x_adv = x_batch.detach()
    x_adv[inds] += epsilon * grad_sign[inds]
    return x_adv