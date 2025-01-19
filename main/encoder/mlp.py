# 1️⃣ This code trains a small classifier on embedding conditions to predict levels/returns.
# 2️⃣ To use:
#   - Firstly, place the condition data in the `data/` directory.
#   - The conditions should have different levels to ensure proper training.
#   - If predicting return, manually edit the data path in the `return_dataset` function.
# 3️⃣ After running the code:
#   - Training accuracy logs will be saved in `result_classification/`.
#   - The confusion matrix will be stored in `plot/prediction/`.

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"
import psutil
p = psutil.Process()
p.cpu_affinity(range(0,32))
print(f'CPU pool num after assignment: {len(p.cpu_affinity())}')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

from sklearn import metrics
from sklearn import decomposition
from sklearn import manifold
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
import numpy as np

import copy
import random
import time
import argparse
import logging
import pickle
from torch.nn.functional import normalize
LOGGER = logging.getLogger(__name__)

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.input_fc = nn.Linear(input_dim, 250)
        self.hidden_fc = nn.Linear(250, 100)
        self.output_fc = nn.Linear(100, output_dim)

    def forward(self, x):

        # x = [batch size, height, width]

        batch_size = x.shape[0]

        x = x.view(batch_size, -1)

        # x = [batch size, height * width]

        h_1 = F.relu(self.input_fc(x))

        # h_1 = [batch size, 250]

        h_2 = F.relu(self.hidden_fc(h_1))

        # h_2 = [batch size, 100]

        y_pred = self.output_fc(h_2)

        # y_pred = [batch size, output dim]

        return y_pred, h_2

class RelativeErrorLoss(nn.Module):
    def __init__(self):
        super(RelativeErrorLoss, self).__init__()

    def forward(self, y_pred, y_true):
        # Avoid division by zero by adding a small epsilon
        epsilon = 1e-8
        relative_error = torch.abs(y_pred - y_true) / (torch.abs(y_true) + epsilon)
        return torch.mean(relative_error)

def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc

def train(model, iterator, optimizer, criterion, device):

    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for (x, y) in tqdm(iterator, desc="Training", leave=False):

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        y_pred, _ = model(x)

        loss = criterion(y_pred, y)

        acc = calculate_accuracy(y_pred, y)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion, device):

    epoch_loss = 0
    epoch_acc = 0
    total_samples = 0
    model.eval()
    with torch.no_grad():

        for (x, y) in tqdm(iterator, desc="Evaluating", leave=False):

            x = x.to(device)
            y = y.to(device)

            current_batch_size = x.size(0)  # Number of samples in the current batch
            total_samples += current_batch_size

            y_pred, _ = model(x)

            loss = criterion(y_pred, y)

            acc = calculate_accuracy(y_pred, y)

            epoch_loss += loss.item() * current_batch_size
            epoch_acc += acc.item() * current_batch_size

    return epoch_loss / total_samples, epoch_acc / total_samples

def evaluate_return(model, iterator, criterion, device, mean, std):

    total_loss = 0
    percent_loss = 0

    model.eval()

    with torch.no_grad():

        for (x, y) in tqdm(iterator, desc="Evaluating", leave=False):

            x = x.to(device)
            y = y.to(device)

            y_pred, _ = model(x)
            y_pred = y_pred * std + mean
            y_pred = torch.clamp(y_pred, min=0, max=20000)
            y = y
            loss = criterion(y_pred, y)
            # y_pred = y_pred.detach().cpu().numpy() * std + mean
            # loss = np.sqrt(((np.squeeze(y) - np.squeeze(y_pred))**2).mean())

            total_loss += loss.item()
            

    return total_loss, 0

def evaluate_return_wo_batch(model, iterator, criterion, device, mean, std):

    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():

        # x = np.array([item[0] for item in iterator])        
        # y = np.array([item[1] for item in iterator])
        # y_pred, _ = model(x)

        # y_pred = y_pred * std + mean
        # y = y
        # # loss = criterion(y_pred, y.detach().cpu().numpy())
        # loss = np.sqrt(((np.squeeze(y) - np.squeeze(y_pred))**2).mean())

        # epoch_loss += loss.item()

        for (x, y) in tqdm(iterator, desc="Evaluating", leave=False):

            x = x.to(device)
            y = y.to(device)

            y_pred, _ = model(x)

            y_pred = y_pred.detach().cpu().numpy() * std + mean
            y = y.detach().cpu().numpy()
            # loss = criterion(y_pred, y.detach().cpu().numpy())
            loss = ((np.squeeze(y) - np.squeeze(y_pred))**2).mean()

            epoch_loss += loss.item()
            break
            

    return epoch_loss/len(iterator), 0

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def get_predictions(model, iterator, device):

    model.eval()

    images = []
    labels = []
    probs = []

    with torch.no_grad():

        for (x, y) in iterator:

            x = x.to(device)

            y_pred, _ = model(x)

            y_prob = F.softmax(y_pred, dim=-1)

            images.append(x.cpu())
            labels.append(y.cpu())
            probs.append(y_prob.cpu())

    images = torch.cat(images, dim=0)
    labels = torch.cat(labels, dim=0)
    probs = torch.cat(probs, dim=0)

    return images, labels, probs

def plot_confusion_matrix(labels, pred_labels, exp_name, args):

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    cm = metrics.confusion_matrix(labels, pred_labels)
    cm = metrics.ConfusionMatrixDisplay(cm, display_labels=range(3))
    cm.plot(values_format='d', cmap='Blues', ax=ax)
    # datadir = "plot/prediction"
    datadir = os.path.join("plot/prediction", args.datadir)
    savepic = os.path.join(datadir, f"{exp_name}_confusion.png")
    plt.savefig(savepic)
    print("Plot saved at {}".format(savepic))

def plot_loss(train_loss_list, name, args):
    from matplotlib import pyplot as plt
    plt.figure()
    plt.plot(train_loss_list, label=name)
    plt.legend()
    plt.show()
    savefile = os.path.join("plot/prediction", args.datadir, name + ".png")
    plt.savefig(savefile)

# normalize train and test separately
# def normalizebyi(data, index, dim=1, outdtype=torch.float32):
#     emb = [x[index] for x in data]
#     emb = torch.tensor(emb, dtype=torch.float32)
#     emb = normalize(emb, dim=dim)
#     for i in range(len(data)): 
#         data[i][index] = torch.tensor(emb[i],dtype=outdtype)
#     return data

# def normalizedata(data):
#     data = normalizebyi(normalizebyi(data, 0, dim=1), 1, dim=0, outdtype=torch.float32)
#     data = [[data[i][0], torch.tensor([data[i][1]])] for i in range(len(data))]
#     return data

# normalize train and test by train mean and std
def normalizebyi(data, index, dim=0, mean=None, std=None, type="train"):
    # normalize input or output
    # index = 0 for input, index = 1 for output
    if type=="train":
        mean = torch.mean(torch.stack([torch.tensor(x[index]) for x in data]), dim=dim).detach().cpu().numpy()
        std = torch.std(torch.stack([torch.tensor(x[index]) for x in data]), dim=dim).detach().cpu().numpy()
    for i in range(len(data)):
        data[i][index] = (data[i][index] - mean) / std
    return data, mean, std

def return_dataset(args):
    print(f"Fetching dataset for return...")
    datadir = os.path.join("data", args.datadir)
    datafile = os.path.join(datadir,args.exp_name + ".pkl")
    if os.path.isfile(datafile):
        with open(datafile, 'rb') as f:
            emb_list = pickle.load(f)
        print("Loaded embedding list")
    else:
        print("Data not found:", datafile)
    # LOGGER.info(">>> Num of skills in one traj: {}~{}, Average {}".format(min(emb_list["num_z"]), 
    #                                                                   max(emb_list["num_z"]), 
    #                                                                   sum(emb_list["num_z"])/len(emb_list["num_z"])))
    emb_list["emb"] = [np.squeeze(i) for i in emb_list["emb"]]
    length = len(emb_list["emb"])

    # load rewards
    mycwd = os.getcwd()
    os.chdir("/home/zichang/proj/IQ-Learn/iq_learn/encoder")
    # HACK change this for return dataset 
    # check if args.exp_name contains "Cheetah", "Hopper", "Walker" etc., and match it with the corresponding dataset_paths
    if "cheetah" in args.exp_name.lower():
        dataset_paths = ['/home/zichang/proj/IQ-Learn/iq_learn/experts/cheetah/HalfCheetah-v2_300_6317r.pkl',
                            '/home/zichang/proj/IQ-Learn/iq_learn/experts/cheetah/HalfCheetah-v2_300_10851r.pkl',
                            '/home/zichang/proj/IQ-Learn/iq_learn/experts/cheetah/HalfCheetah-v2_300_12794r.pkl']
    elif "hopper" in args.exp_name.lower():
        dataset_paths = ['/home/zichang/proj/IQ-Learn/iq_learn/experts/hopper/Hopper-v2_300_409r.pkl',
                            '/home/zichang/proj/IQ-Learn/iq_learn/experts/hopper/Hopper-v2_300_879r.pkl',
                            '/home/zichang/proj/IQ-Learn/iq_learn/experts/hopper/Hopper-v2_300_3205r.pkl']
    elif "walker" in args.exp_name.lower():
        dataset_paths = ['/home/zichang/proj/IQ-Learn/iq_learn/experts/walker/Walker2d-v2_300_3073r.pkl',
                            '/home/zichang/proj/IQ-Learn/iq_learn/experts/walker/Walker2d-v2_300_4366r.pkl',
                            '/home/zichang/proj/IQ-Learn/iq_learn/experts/walker/Walker2d-v2_300_5909r.pkl']
    elif "ant" in args.exp_name.lower():
        dataset_paths = ['/home/zichang/proj/IQ-Learn/iq_learn/experts/ant/Ant-v2_300_2964r.pkl',
                        '/home/zichang/proj/IQ-Learn/iq_learn/experts/ant/Ant-v2_300_4872r.pkl',
                        '/home/zichang/proj/IQ-Learn/iq_learn/experts/ant/Ant-v2_300_5652r.pkl']
    # elif "humanoid" in args.exp_name.lower():
    #     dataset_paths = ['/home/zichang/proj/IQ-Learn/iq_learn/experts/humanoid/Humanoid-v2_300_2907r.pkl',
    #                     '/home/zichang/proj/IQ-Learn/iq_learn/experts/humanoid/Humanoid-v2_300_4997r.pkl',
    #                     '/home/zichang/proj/IQ-Learn/iq_learn/experts/humanoid/Humanoid-v2_300_6206r.pkl']
    elif "pusher" in args.exp_name.lower():
        dataset_paths = ['/home/zichang/proj/IQ-Learn/iq_learn/experts/pusher/Pusher-v2_300_-164r.pkl',
                         '/home/zichang/proj/IQ-Learn/iq_learn/experts/pusher/Pusher-v2_300_-104r.pkl',
                         '/home/zichang/proj/IQ-Learn/iq_learn/experts/pusher/Pusher-v2_300_-65r.pkl',]
    elif "humanoidstandup" in args.exp_name.lower() or "humanoid_standup" in args.exp_name.lower():
        dataset_paths = ['/home/zichang/proj/IQ-Learn/iq_learn/experts/humanoid_standup/HumanoidStandup-v2_300_45888r.pkl',
                         '/home/zichang/proj/IQ-Learn/iq_learn/experts/humanoid_standup/HumanoidStandup-v2_300_68520r.pkl',
                         '/home/zichang/proj/IQ-Learn/iq_learn/experts/humanoid_standup/HumanoidStandup-v2_300_82237r.pkl',]
    else:
        print("Name does not contain available env name: Cheetah, Walker, Hopper, Ant, HumanoidStandup")
    
    # if "Cheetah" in args.exp_name:
    #     dataset_paths = ['/home/zichang/proj/IQ-Learn/iq_learn/experts/cheetah/HalfCheetah-v2_300_6317r.pkl',
    #                         '/home/zichang/proj/IQ-Learn/iq_learn/experts/cheetah/HalfCheetah-v2_300_10851r.pkl',
    #                         '/home/zichang/proj/IQ-Learn/iq_learn/experts/cheetah/HalfCheetah-v2_300_12794r.pkl']
    # elif "Hopper" in args.exp_name:
    #     dataset_paths = ['/home/zichang/proj/IQ-Learn/iq_learn/experts/hopper/Hopper-v2_300_409r.pkl',
    #                         '/home/zichang/proj/IQ-Learn/iq_learn/experts/hopper/Hopper-v2_300_879r.pkl',
    #                         '/home/zichang/proj/IQ-Learn/iq_learn/experts/hopper/Hopper-v2_300_3205r.pkl']
    # elif "Walker" in args.exp_name:
    #     dataset_paths = ['/home/zichang/proj/IQ-Learn/iq_learn/experts/walker/Walker2d-v2_300_3073r.pkl',
    #                         '/home/zichang/proj/IQ-Learn/iq_learn/experts/walker/Walker2d-v2_300_4366r.pkl',
    #                         '/home/zichang/proj/IQ-Learn/iq_learn/experts/walker/Walker2d-v2_300_5909r.pkl']
    # elif "Ant" in args.exp_name:
    #     dataset_paths = ['/home/zichang/proj/IQ-Learn/iq_learn/experts/ant/Ant-v2_300_2964r.pkl',
    #                     '/home/zichang/proj/IQ-Learn/iq_learn/experts/ant/Ant-v2_300_4872r.pkl',
    #                     '/home/zichang/proj/IQ-Learn/iq_learn/experts/ant/Ant-v2_300_5652r.pkl']
    # elif "Humanoid" in args.exp_name:
    #     dataset_paths=['/home/zichang/proj/IQ-Learn/iq_learn/experts/humanoid/Humanoid-v2_300_2907r.pkl',
    #                     '/home/zichang/proj/IQ-Learn/iq_learn/experts/humanoid/Humanoid-v2_300_4997r.pkl',
    #                     '/home/zichang/proj/IQ-Learn/iq_learn/experts/humanoid/Humanoid-v2_300_6206r.pkl']
    rewards = []
    for idx, dataset_path in enumerate(dataset_paths):
        with open(dataset_path, 'rb') as f:
            trajectories = pickle.load(f)
        rewards.extend([np.array(x, dtype=type) for x in trajectories['rewards']])
    os.chdir(mycwd) 
    rewards = [[float(sum(x))] for x in rewards]
    data = [[emb_list["emb"][i], torch.tensor([rewards[i]])]for i in range(length)]
    random.shuffle(data)
    num_heldout = 82
    train_data = data[:-num_heldout]
    test_data = data[-num_heldout:]

    train_data, x_mean, x_std = normalizebyi(train_data, 0, dim=0, type="train")
    train_data = [[np.nan_to_num(i[0], nan=1e-8), i[1]] for i in train_data]
    train_data, y_mean, y_std = normalizebyi(train_data, 1, dim=0,  type="train")

    train_data = [[train_data[i][0], torch.tensor([train_data[i][1]])] for i in range(len(train_data))]
    
    test_data, _, _ = normalizebyi(test_data, 0, dim=0, mean=x_mean, std=x_std, type="test")
    test_data = [[np.nan_to_num(i[0], nan=1e-8), i[1]] for i in test_data]
    # save meanstd for evaluation use
    datafile_meanstd_return = os.path.join(datadir,f"{args.exp_name}_predict_return_meanstd" + ".pkl")
    with open(datafile_meanstd_return, 'wb') as f:
        pickle.dump([test_data, y_mean, y_std], f)
    test_data, _, _ = normalizebyi(test_data, 1, dim=0, mean=y_mean, std=y_std, type="test")
    test_data = [[test_data[i][0], torch.tensor([test_data[i][1]])] for i in range(len(test_data))]

    datafile_train = os.path.join(datadir,f"{args.exp_name}_predict_return_train" + ".pkl")
    datafile_test = os.path.join(datadir,f"{args.exp_name}_predict_return_test" + ".pkl")
    with open(datafile_train, 'wb') as f:
        pickle.dump(train_data, f)
    with open(datafile_test, 'wb') as f:
        pickle.dump(test_data, f)
    return train_data, test_data

def level_dataset(args):
    print(f"Fetching data for level...")
    datadir = os.path.join("data", args.datadir)
    datafile = os.path.join(datadir,args.exp_name + ".pkl")
    if os.path.isfile(datafile):
        with open(datafile, 'rb') as f:
            emb_list = pickle.load(f)
        print("Loaded previous data")
    else:
        print("Data not found:", datafile)
    # LOGGER.info(">>> Num of skills in one traj: {}~{}, Average {}".format(min(emb_list["num_z"]), 
    #                                                                   max(emb_list["num_z"]), 
    #                                                                   sum(emb_list["num_z"])/len(emb_list["num_z"])))
    emb_list["emb"] = [np.squeeze(i) for i in emb_list["emb"]]
    length = len(emb_list["emb"])
    data = [[emb_list["emb"][i], emb_list["level"][i]]for i in range(length)]
    random.shuffle(data)
    num_heldout = 82

    train_data =  data[:-num_heldout]
    test_data = data[-num_heldout:]
    train_data, x_mean, x_std = normalizebyi(train_data, 0, dim=0, type="train")
    test_data, _, _ = normalizebyi(test_data, 0, dim=0, mean=x_mean, std=x_std, type="test")

    datafile_train = os.path.join(datadir,f"{args.exp_name}_predict_level_train" + ".pkl")
    train_data = [[train_data[i][0], torch.tensor(train_data[i][1])] for i in range(len(train_data))]
    datafile_test = os.path.join(datadir,f"{args.exp_name}_predict_level_test" + ".pkl")
    test_data = [[test_data[i][0], torch.tensor(test_data[i][1])] for i in range(len(test_data))]
    with open(datafile_train, 'wb') as f:
        pickle.dump(train_data, f)
    with open(datafile_test, 'wb') as f:
        pickle.dump(test_data, f)
    return train_data, test_data

def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("exp_name", help="name of the experiment to run")
    arg_parser.add_argument("--emb_size", type=int, default=256)
    arg_parser.add_argument("--epoch", type=int, default=10)
    arg_parser.add_argument(
        "--label", type=str, default="level", choices=["level", "return"]
    )
    arg_parser.add_argument(
        "--datadir", type=str, default="baseline", choices=["ours", "baseline","gcpc_wo_r", "z_logit", "gcpc"]
    )
    arg_parser.add_argument(
            "-s", "--seed", default=0, help="random seed to use.", type=int)
    
    args = arg_parser.parse_args()

    logname = os.path.join("result_classification", args.datadir, f"{args.exp_name}_{args.label}.log")
    logging.basicConfig(filename=logname,
                        filemode='w',
                        # format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        format='%(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.label == "level":
        datadir = os.path.join("data", args.datadir)
        datafile_train = os.path.join(datadir,f"{args.exp_name}_predict_level_train" + ".pkl")
        datafile_test = os.path.join(datadir,f"{args.exp_name}_predict_level_test" + ".pkl")
        if os.path.isfile(datafile_train):
            with open(datafile_train, 'rb') as f:
                train_data = pickle.load(f)
            print("Loaded train data")
            with open(datafile_test, 'rb') as f:
                test_data = pickle.load(f)
            print("Loaded test data")
        else:
            train_data, test_data = level_dataset(args)
            print("Saved data")
        OUTPUT_DIM = 3
        criterion = nn.CrossEntropyLoss()
    else:
        datadir = os.path.join("data", args.datadir)
        datafile_train = os.path.join(datadir,f"{args.exp_name}_predict_return_train" + ".pkl")
        datafile_test = os.path.join(datadir,f"{args.exp_name}_predict_return_test" + ".pkl")
        if os.path.isfile(datafile_train):
            with open(datafile_train, 'rb') as f:
                train_data = pickle.load(f)
            print("Loaded train data")
            with open(datafile_test, 'rb') as f:
                test_data = pickle.load(f)
            print("Loaded test data")
        else:
            train_data, test_data = return_dataset(args)
            print("Saved data")        
        OUTPUT_DIM = 1
        criterion = RelativeErrorLoss()
        # criterion = nn.MSELoss()


    VALID_RATIO = 0.9

    n_train_examples = int(len(train_data) * VALID_RATIO)
    n_valid_examples = len(train_data) - n_train_examples

    train_data, valid_data = data.random_split(train_data,
                                            [n_train_examples, n_valid_examples])

    LOGGER.info(f'Number of training examples: {len(train_data)}')
    LOGGER.info(f'Number of validation examples: {len(valid_data)}')
    LOGGER.info(f'Number of testing examples: {len(test_data)}')

    valid_data = copy.deepcopy(valid_data)

    BATCH_SIZE = 64

    train_iterator = data.DataLoader(train_data,
                                    shuffle=True,
                                    batch_size=BATCH_SIZE)

    valid_iterator = data.DataLoader(valid_data,
                                    batch_size=BATCH_SIZE)

    test_iterator = data.DataLoader(test_data,
                                    batch_size=BATCH_SIZE)

    INPUT_DIM = args.emb_size

    model = MLP(INPUT_DIM, OUTPUT_DIM)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'The model has {count_parameters(model):,} trainable parameters')

    optimizer = optim.Adam(model.parameters())
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = criterion.to(device)
    EPOCHS = args.epoch

    best_valid_loss = float('inf')

    train_loss_list = []
    val_loss_list = []
    save_flag = False # in case no model is saved
    for epoch in trange(EPOCHS):

        start_time = time.monotonic()

        train_loss, train_acc = train(model, train_iterator, optimizer, criterion, device)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion, device)
        train_loss_list.append(train_loss)
        val_loss_list.append(valid_loss)
        if valid_loss < best_valid_loss:
            save_flag = True
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), f'experiments/mlp/{args.exp_name}_model.pt')

        end_time = time.monotonic()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        if args.label=="level":
            LOGGER.info(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            LOGGER.info(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
            LOGGER.info(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
        else:
            LOGGER.info(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            LOGGER.info(f'\tTrain Loss: {train_loss:.10f}')
            LOGGER.info(f'\t Val. Loss: {valid_loss:.10f}')
    if save_flag==False:
        torch.save(model.state_dict(), f'experiments/mlp/{args.exp_name}_model.pt')
    model.load_state_dict(torch.load(f'experiments/mlp/{args.exp_name}_model.pt'))
   
    if args.label=="level":
        test_loss, test_acc = evaluate(model, test_iterator, criterion, device)
        LOGGER.info(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
        images, labels, probs = get_predictions(model, test_iterator, device)
        pred_labels = torch.argmax(probs, 1)
        plot_confusion_matrix(labels, pred_labels,args.exp_name, args)
    else:
        datafile_meanstd_return = os.path.join(datadir,f"{args.exp_name}_predict_return_meanstd" + ".pkl")
        with open(datafile_meanstd_return, 'rb') as f:
            test_data, mean, std = pickle.load(f)
            print("Loaded test data and train meanstd")
        test_iterator = data.DataLoader(test_data,
                                    batch_size=BATCH_SIZE)
        test_loss, test_acc = evaluate_return(model, test_iterator, criterion, device, mean, std)
        # test_loss, test_acc = evaluate_return_wo_batch(model, test_iterator, criterion, device, mean, std)
        LOGGER.info(f'Test Loss: {test_loss:.5f}')
        LOGGER.info(f'Test Loss per traj: {100*test_loss/len(test_data):.2f}%')
        plot_loss(train_loss_list, "train_loss", args)
        plot_loss(val_loss_list, "val_loss", args)
    
    print("Log saved at {}".format(logname))


if __name__ == '__main__':
    main()