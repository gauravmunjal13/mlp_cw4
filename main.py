import os
import torchvision.models

from argparse import ArgumentParser
from attacks import *
from models import *
from preprocess import get_data
from torch.utils.data import DataLoader, TensorDataset

def train(args):
    print(args)
    # Make experiments reproducible
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # Setup data, model, optimizer, and lr scheduler
    save_dir = f'save/{args.exp_name}_{args.dataset_name}_{args.seed}/'
    os.makedirs(save_dir, exist_ok=True)
    train_data, test_data = get_data(args)
    model = SimpleCNN(args).cuda()
    # model = torchvision.models.resnet18({'num_classes': args.num_classes}).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_init, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs, eta_min=args.lr_final)
    if args.attack_type is None:
        attack = None
    elif args.attack_type == 'fgsm':
        attack = FGSMAttack(args, model)
    else:
        raise ValueError
    train_acc = np.empty(args.num_epochs)
    for cur_epoch in range(args.num_epochs):
        scheduler.step(cur_epoch)
        # Train iter
        model.train()
        num_correct_train = 0
        num_total_train = 0
        for x_batch, y_batch in train_data:
            x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
            optimizer.zero_grad()
            if attack is None:
                pred = model(x_batch)
            else:
                # Adversarial training
                x_adv_batch = attack(x_batch, y_batch)
                pred = model(x_adv_batch)
            loss = torch.nn.CrossEntropyLoss()(pred, y_batch)
            loss.backward()
            optimizer.step()
            _, pred_ind = pred.max(1)
            num_correct_train += pred_ind.eq(y_batch).sum().item()
            num_total_train += y_batch.size(0)
        train_acc_epoch = num_correct_train / num_total_train
        train_acc[cur_epoch] = train_acc_epoch
        np.savetxt(save_dir + 'train_acc.txt', train_acc, delimiter=',')
        torch.save(model.state_dict(), save_dir + 'model.pt')
        print(f'epoch {cur_epoch}, train_acc {train_acc_epoch:.3f}')

def get_adv_test_data(args):
    save_dir = f'save/{args.exp_name}_{args.dataset_name}_{args.seed}/'
    _, test_data = get_data(args)
    model = SimpleCNN(args).cuda()
    weights_path = save_dir + 'model.pt'
    model.load_state_dict(torch.load(weights_path))
    if args.attack_type == 'fgsm':
        attack = FGSMAttack(args, model)
    else:
        raise ValueError
    x_adv = []
    y_adv = []
    for x_batch, y_batch in test_data:
        x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
        x_adv_batch = attack(x_batch, y_batch)
        x_adv.append(x_adv_batch)
        y_adv.append(y_batch)
    x_adv = torch.cat(x_adv)
    y_adv = torch.cat(y_adv)
    adv_test_data = DataLoader(TensorDataset(x_adv, y_adv), batch_size=args.batch_size)
    return adv_test_data

def test(args):
    print(args)
    save_dir = f'save/{args.exp_name}_{args.dataset_name}_{args.seed}/'
    model = SimpleCNN(args).cuda()
    weights_path = save_dir + 'model.pt'
    model.load_state_dict(torch.load(weights_path))
    if args.attack_type is None:
        _, test_data = get_data(args)
    else:
        test_data = get_adv_test_data(args)
    model.eval()
    num_correct_test = 0
    num_total_test = 0
    for x_batch, y_batch in test_data:
        x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
        pred = model(x_batch)
        _, pred_ind = pred.max(1)
        num_correct_test += pred_ind.eq(y_batch).sum().item()
        num_total_test += y_batch.size(0)
    test_acc = num_correct_test / num_total_test
    with open(save_dir + f'{args.attack_type}_{args.attack_args}.txt', 'w') as f:
        f.write(f'{test_acc:.3f}')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--exp_type', dest='exp_type', type=str, required=True)
    parser.add_argument('--exp_name', dest='exp_name', type=str, required=True)
    parser.add_argument('--dataset_name', dest='dataset_name', type=str, default='cifar10')
    parser.add_argument('--num_classes', dest='num_classes', type=int, default=10)
    parser.add_argument('--attack_type', dest='attack_type', type=str, default=None)
    parser.add_argument('--attack_args', dest='attack_args', type=str, default=None, help='comma delimited list of args')
    parser.add_argument('--seed', dest='seed', type=int, default=0)
    parser.add_argument('--num_epochs', dest='num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=100)
    parser.add_argument('--lr_init', dest='lr_init', type=float, default=0.01)
    parser.add_argument('--lr_final', dest='lr_final', type=float, default=1e-4)
    parser.add_argument('--momentum', dest='momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=5e-4)
    args = parser.parse_args()
    if args.exp_type == 'train':
        train(args)
    elif args.exp_type == 'test':
        test(args)