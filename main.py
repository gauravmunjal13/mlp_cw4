import numpy as np
import os
import torch

from argparse import ArgumentParser
from models import *
from preprocess import get_data
from torch.utils.data import DataLoader, TensorDataset

def main(args):
    print(args)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    rng = np.random.RandomState(args.seed)
    save_dir = 'save/{}/'.format(f'{args.exp_name}')
    os.makedirs(save_dir, exist_ok=True)
    x_train, y_train, x_test, y_test = get_data(args)
    train_loader = DataLoader(TensorDataset(torch.tensor(x_train), torch.tensor(y_train).long()), batch_size=args.batch_size,
        shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(x_test), torch.tensor(y_test).long()), batch_size=args.batch_size)
    model = SimpleCNN(args).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_init, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs, eta_min=args.lr_final)
    stats = np.empty((args.num_epochs, 2))
    for cur_epoch in range(args.num_epochs):
        scheduler.step(cur_epoch)
        model.train()
        num_correct_train = 0
        num_total_train = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
            optimizer.zero_grad()
            pred = model(x_batch)
            loss = torch.nn.CrossEntropyLoss()(pred, y_batch)
            loss.backward()
            optimizer.step()
            _, pred_ind = pred.max(1)
            num_correct_train += pred_ind.eq(y_batch).sum().item()
            num_total_train += y_batch.size(0)
        model.eval()
        num_correct_test = 0
        num_total_test = 0
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
            pred = model(x_batch)
            _, pred_ind = pred.max(1)
            num_correct_test += pred_ind.eq(y_batch).sum().item()
            num_total_test += y_batch.size(0)
        train_acc = num_correct_train / num_total_train
        val_acc = num_correct_test / num_total_test
        stats[cur_epoch, 0] = train_acc
        stats[cur_epoch, 1] = val_acc
        np.savetxt(save_dir + 'stats.txt', stats, delimiter=',')
        torch.save(model.state_dict(), save_dir + 'model.pt')
        print('epoch {}, train_acc {:.3f}, val_acc {:.3f}'.format(
            cur_epoch,
            train_acc,
            val_acc
        ))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--exp_name', dest='exp_name', type=str, default='exp_name')
    parser.add_argument('--dataset_name', dest='dataset_name', type=str, default='cifar10')
    parser.add_argument('--attack_type', dest='attack_type', type=str, default=None)
    parser.add_argument('--seed', dest='seed', type=int, default=0)
    parser.add_argument('--num_classes', dest='num_classes', type=int, default=10)
    parser.add_argument('--num_epochs', dest='num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=100)
    parser.add_argument('--lr_init', dest='lr_init', type=float, default=0.01)
    parser.add_argument('--lr_final', dest='lr_final', type=float, default=1e-4)
    parser.add_argument('--momentum', dest='momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=5e-4)
    args = parser.parse_args()
    main(args)