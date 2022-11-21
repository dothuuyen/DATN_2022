import argparse
from torch.utils.data import Dataset
import numpy as np
import random
import cv2
import glob
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize
import torch.optim as optim
from sklearn import metrics

import os
import time

import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

import torch.nn.functional as F

from models import build_model
from label_probagation import label_probagation_GCN, label_probagation_GTN

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def arg_parser():
    parser = argparse.ArgumentParser(description='PyTorch Action recognition Training')

    # model definition
    parser.add_argument('--backbone_net', default='resnet', type=str, help='backbone network')
    parser.add_argument('-d', '--depth', default=50, type=int, metavar='N',
                        help='depth of resnet (default: 18)', choices=[18, 34, 50, 101, 152])
    parser.add_argument('--dropout', default=0.5, type=float,
                        help='dropout ratio before the final layer')
    parser.add_argument('--groups', default=8, type=int, help='number of frames')
    parser.add_argument('--frames_per_group', default=1, type=int,
                        help='[uniform sampling] number of frames per group; '
                             '[dense sampling]: sampling frequency')
    parser.add_argument('--without_t_stride', dest='without_t_stride', action='store_true',
                        help='skip the temporal pooling in the model')
    parser.add_argument('--pooling_method', default='max',
                        choices=['avg', 'max'], help='method for temporal pooling method')
    parser.add_argument('--dw_t_conv', dest='dw_t_conv', action='store_true',
                        help='[S3D model] only enable depth-wise conv for temporal modeling')
    # model definition: temporal model for 2D models
    parser.add_argument('--temporal_module_name', default='TAM', type=str,
                        help='[2D model] which temporal aggregation module to use. None == TSN',
                        choices=[None, 'TSN', 'TAM'])
    parser.add_argument('--blending_frames', default=3, type=int, help='For TAM only.')
    parser.add_argument('--blending_method', default='sum',
                        choices=['sum', 'max'], help='method for blending channels in TAM')
    parser.add_argument('--no_dw_conv', dest='dw_conv', action='store_false',
                        help='[2D model] disable depth-wise conv for TAM')

    # training setting
    parser.add_argument('--use_model', default='none', type=str,
                        help='Filtering model',
                        choices=['baseline', 'none', 'gcn', 'gtn'])

    parser.add_argument('-b', '--batch-size', default=8, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--lr_scheduler', default='cosine', type=str,
                        help='learning rate scheduler',
                        choices=['step', 'multisteps', 'cosine', 'plateau'])
    parser.add_argument('--lr_steps', default=[15, 30, 45], type=float, nargs="+",
                        metavar='LRSteps', help='[step]: use a single value: the periodto decay '
                                                'learning rate by 10. '
                                                '[multisteps] epochs to decay learning rate by 10')
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--pretrained', dest='pretrained', type=str, metavar='PATH',
                        help='use pre-trained model')
    parser.add_argument('--clip_gradient', '--cg', default=None, type=float,
                        help='clip the total norm of gradient before update parameter')
    parser.add_argument('--no_imagenet_pretrained', dest='imagenet_pretrained',
                        action='store_false',
                        help='disable to load imagenet model')
    parser.add_argument('--seed', default=2022, type=int,
                        help='Training seed')

    # data-related
    parser.add_argument('--train_folder', metavar='DIR', help='path to dataset file list')
    parser.add_argument('--test_folder', metavar='DIR', help='path to dataset file list')
    parser.add_argument('--dataset', default='st2stv2')
    parser.add_argument('--threed_data', action='store_true',
                        help='format data to 5D for 3D onv.')
    parser.add_argument('--disable_scaleup', action='store_true',
                        help='do not scale up and then crop a small region, '
                             'directly crop the input_size from center.')
    parser.add_argument('--modality', default='rgb', type=str, help='rgb or flow',
                        choices=['rgb', 'flow'])

    # logging
    parser.add_argument('--logdir', default='./', type=str, help='log path')
    parser.add_argument('--print-freq', default=100, type=int,
                        help='frequency to print the log during the training')

    parser.add_argument('--n_filters', default=1, type=int,
                        help='Number of filters')
    # for testing and validation

    return parser


# In[4]:
def get_train_files(input_folder, n_frames, step=None, prefix='Train'):

    folder_level1 = sorted(os.listdir(input_folder))
    output = []
    output_test = []
    for folder_name in folder_level1:
        if prefix in folder_name:
            _i = np.random.randint(5)

            filenames = sorted(glob.glob("%s/%s/*.tif" % (input_folder, folder_name)))
            basename = [os.path.basename(f) for f in filenames]

            labels = [0 for _ in range(len(filenames))]
            
            start = 0
            while start + n_frames < len(filenames):
                list_files = filenames[start:start + n_frames]
                if _i >= 0:
                    output += [{'list_files': list_files,
                           'start_frame_idx': start,
                           'end_frame_idx': start+ n_frames - 1,
                           'prefix': prefix,
                           'label': 0,
                           'video_name': folder_name
                           }]
                else:
                    output_test += [{'list_files': list_files,
                           'start_frame_idx': start,
                           'end_frame_idx': start+ n_frames - 1,
                           'prefix': prefix,
                           'label': 0,
                           'video_name': folder_name
                           }]
                start += step if step is not None else n_frames
            
            
    return output, output_test

def get_gt_label(filenames, threshold = 0.2):
    cnt = 0
    
    for i, fname in enumerate(filenames):
        img = np.float32(cv2.imread(fname, 0))/255.
        if np.sum(img) > 0.1:
            cnt += 1
            
    return int(cnt >= 0.2*len(filenames))


def get_test_files(input_folder, n_frames, step=None, prefix='Test', train_split=None):

    folder_level1 = sorted(os.listdir(input_folder))
    output_trains, output_tests = [], []
    
    for folder_name in folder_level1:
        if (prefix in folder_name) and ('gt' not in folder_name):
            test_id = int(folder_name[-3:])
            _i = np.random.randint(3)
            print(test_id)
            filenames = sorted(glob.glob("%s/%s/*.tif" % (input_folder, folder_name)))
            gt_filenames = sorted(glob.glob("%s/%s_gt/*.bmp" % (input_folder, folder_name)))
            basename = [os.path.basename(f) for f in filenames]

            labels = [0 for _ in range(len(filenames))]
            
            start = 0
            while start + n_frames < len(filenames):
                list_files = filenames[start:start + n_frames]
                data = {'list_files': list_files,
                           'start_frame_idx': start,
                           'end_frame_idx': start+ n_frames - 1,
                           'prefix': prefix,
                           'label': get_gt_label(gt_filenames[start:start + n_frames]),
                           'video_name': folder_name
                           }
                start += step if step is not None else n_frames
                
                if _i >= 2:
                    output_trains += [data]
                else:
                    output_tests += [data]
            
            
    return output_trains, output_tests


def load_images_and_label(filenames, image_size, local_transform, normalize_transform):
    """Loads an image into a tensor. Also returns its label."""
    imgs = []

    for i, fname in enumerate(filenames):
        img = cv2.imread(fname)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #if local_transform is not None:
            #res = local_transform(image=img)
            #img = res['image']

        img = cv2.resize(img, (image_size, image_size))
        img = torch.tensor(img).permute((2, 0, 1)).float().div(255)
        img = normalize_transform(img)
        imgs.append(img)

    return torch.stack(imgs, dim=0).reshape((-1, image_size, image_size))


class VideoDataset(Dataset):
    def __init__(self, output_files, labels, n_frames, image_size,
                 device,
                 split,
                 seed=10):

        assert len(output_files) == len(labels)
        self.image_size = image_size
        self.device = device
        self.split = split

        self.video_imgs = output_files
        self.list_labels = labels
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        normalize_transform = Normalize(mean, std)

        self.transform = None
        self.normalize_transform = normalize_transform

    def __getitem__(self, index):
        filenames = self.video_imgs[index]['list_files']
        label = self.list_labels[index]
        imgs = load_images_and_label(filenames,
                                     self.image_size,
                                     self.transform,
                                     self.normalize_transform)

        return imgs, label

    def __len__(self):
        return len(self.video_imgs)

    def get_labels(self):
        return self.list_labels


# Always get features prediction
def get_predictions(model, data_loader, device):
    predicts = []
    features = []
    
    model = model.eval()
    for batch_idx, data in enumerate(data_loader):
        batch_size = data[0].shape[0]
        
        x = data[0].to(device)
        y_true = data[1].to(device).long()

        y_pred, fts = model(x, features=True)

        predicts += y_pred.detach().cpu().numpy().tolist()
        features += fts.detach().cpu().numpy().tolist()
        
    return np.array(predicts), np.array(features)


def train_loop_fn(net, loader, optimizer, scheduler, device):
    net = net.train()
    log_steps = 100
    loss_fn = nn.CrossEntropyLoss()

    sum_loss = 0
    n_iter = 0

    start_time = time.time()
    total_steps = len(loader)
    for x, data in enumerate(loader):
        optimizer.zero_grad()
        output = net(data[0].to(device))

        # print(data[0].shape)
        # print(output)
        # print(data[1])

        loss = loss_fn(output, data[1].to(device).long())
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()

        n_iter += 1
        if (x + 1) % log_steps == 0:
            print("[x = %i] loss = %.6f, speed = %.6f" % (x + 1, loss.item(), (time.time() - start_time) / n_iter))

        sum_loss += loss.item()

        if scheduler is not None:
            scheduler.step()

    return sum_loss

def evaluate_fn(net, data_loader, device):
    bce_loss = 0
    total_examples = 0
    n_iter = 0
    loss_fn = nn.CrossEntropyLoss()
    net.eval()
    outputs = []
    output_fts = []
    labels = []

    for batch_idx, data in enumerate(data_loader):
        batch_size = data[0].shape[0]
        x = data[0].to(device)
        y_true = data[1].to(device).long()

        y_pred, fts = net(x, True)
        #print(np.mean(y_pred.detach().cpu().numpy()))        

        bce_loss += loss_fn(y_pred, y_true).item()
        y_pred = torch.nn.functional.softmax(y_pred, dim=-1)

        total_examples += batch_size
        n_iter += 1

        outputs += y_pred.detach().cpu().numpy().tolist()
        output_fts += fts.detach().cpu().numpy().tolist()
        labels += data[1].numpy().tolist()

    bce_loss /= n_iter

    print("\t Valid BCE: %.4f" % (bce_loss))
    return bce_loss, labels, np.array(outputs), np.array(output_fts)

def train_model(model, train_files, train_labels, device, epochs=10):
    train_dataset = VideoDataset(train_files,
                                 train_labels,
                                 8,
                                 224,
                                 'cpu',
                                 split='test',
                                 seed=2022)

    train_loader = DataLoader(train_dataset, batch_size=4,
                              shuffle=True,
                              num_workers=2)

    optimizer = optim.SGD(model.parameters(), lr=1e-4)
    scheduler = None

    for ii in range(epochs):
        print("Epochs: {}".format(ii))
        start_time = time.time()
        train_score = train_loop_fn(model,
                                    train_loader,
                                    optimizer,
                                    scheduler,
                                    device)
        print(train_score)
        print("Training one epoch in: {}".format(time.time() - start_time))

def test_model(model, test_files, device):
    test_labels = [a['label'] for a in test_files]

    test_dataset = VideoDataset(test_files,
                                test_labels,
                                8,
                                224,
                                'cpu',
                                split='test', seed=2022)
    test_loader = DataLoader(test_dataset, batch_size=2,
                             shuffle=False,
                             num_workers=2)

    test_loss, labels, outputs, _ = evaluate_fn(model, test_loader, device)
    y = np.array([p[1] for p in outputs])
    print("Test loss: ", test_loss)
    fpr, tpr, thresholds = metrics.roc_curve(labels, y, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    print("AUC: ", auc)
    best_C = None
    best_f1_score = -1
    best_t = 0.5

    for t in range(50, 51, 1):
        output_y = np.zeros_like(y)
        output_y[y >= t*0.01] = 1
        f1 = metrics.f1_score(labels, output_y)

        if f1 > best_f1_score:
            best_f1_score = f1
            best_C = metrics.confusion_matrix(labels, output_y)

    print('F1-Score = ', best_f1_score)
    print('Accuracy = ', 1.0 * (best_C[0][0] + best_C[1][1]) / np.sum(best_C))

    output_y = np.zeros_like(y)
    output_y[y >= best_t] = 1

    video_names = []
    start_frames = []
    labels = []
    predicted_labels = []
    for i in range(len(test_files)):
        video_names += [test_files[i]['video_name']]
        start_frames += [test_files[i]['start_frame_idx']]
        labels += [test_files[i]['label']]
        predicted_labels += [output_y[i]]

    df = pd.DataFrame({
        'video_name': video_names,
        'start_frame': start_frames,
        'labels': labels,
        'predicted_label': predicted_labels
    })

    df.to_csv('predicted.csv', index=False)


def create_graph(file_names, output_proba, outputs_fts, model_gnn):
    N = len(file_names)
    video_name = file_names[0]['video_name']
    position = [(a['start_frame_idx'], a['end_frame_idx']) for a in file_names]
    low_p = np.percentile(output_proba[:, 1], 20)
    high_p = np.percentile(output_proba[:, 1], 95)

    # Create label and train mask
    labels = []
    is_train = []
    for i in range(len(output_proba)):
        if output_proba[i][1] < low_p:
            is_train += [1]
            labels += [0]
        elif (low_p < output_proba[i][1]) and (output_proba[i][1] < high_p):
            is_train += [0]
            labels += [0]
        else:
            is_train += [1]
            labels += [1]

    # Create adj-matrix
    A = np.zeros((N, N), dtype=np.float32)
    for i in range(len(outputs_fts)):
        for j in range(len(outputs_fts)):
            A[i][j] = np.dot(outputs_fts[i] - outputs_fts[j], outputs_fts[i] - outputs_fts[j])

    A[i][j] = np.sqrt(A[i][j])
    amax = np.max(A, axis=1)
    A /= amax

    edge_list = []
    for i in range(N):
        for j in range(N):
            if A[i][j] < 0.1:
                edge_list += [(i, j)]
                edge_list += [(j, i)]

    # edge_list = np.unique(edge_list)
    print("Process: ", video_name)
    print("\tGet {} train, {} test, {} edges".format(sum(is_train), len(is_train) - sum(is_train), len(edge_list)))

    np.savez('%s.npz' % video_name,
             edge_list=np.array(edge_list),
             position=np.array(position),
             is_train=np.array(is_train),
             labels=np.array(labels),
             fts=outputs_fts)

    print('Filtering = ', video_name)
    if model_gnn == 'gtn':
        gtn_out = label_probagation_GTN(np.array(edge_list),
                                        np.array(outputs_fts),
                                        np.array(position),
                                        np.array(labels),
                                        np.array(is_train))
        return gtn_out
    elif model_gnn == 'gcn':
        gcn_out = label_probagation_GCN(np.array(edge_list),
                                        np.array(outputs_fts),
                                        np.array(position),
                                        np.array(labels),
                                        np.array(is_train))
        return gcn_out
    else:
        return np.array([0 for _ in range(len(labels))])


def create_filtering_graph(train_files, outputs, outputs_fts, model_gnn):
    list_video_names = np.array([a['video_name'] for a in train_files])

    start_id = 0
    probagation_label = []
    while start_id < len(train_files):
        end_id = start_id
        while end_id < len(train_files):
            if list_video_names[start_id] != list_video_names[end_id]:
                break
            end_id += 1

        if 'Train' in list_video_names[start_id]:
            probagation_label += [0 for _ in range(start_id, end_id)]
            start_id = end_id
            continue

        label_output = create_graph(train_files[start_id:end_id],
                                    outputs[start_id:end_id],
                                    outputs_fts[start_id:end_id], model_gnn)
        probagation_label += label_output.tolist()
        start_id = end_id

    return np.array(probagation_label)


def filtering_week_supervised(model, train_files, test_files, device, percentile=0.8, model_gnn=None):
    # Step 1: Create base dataset (train with no shuffle)
    train_labels = [a['label'] for a in train_files]
    print("Train labels: Got %i 0-labels, %i 1-label" % (len(train_labels) - sum(train_labels),
                                           sum(train_labels)))
    train_dataset = VideoDataset(train_files,
                                 train_labels,
                                 8,
                                 224,
                                 'cpu',
                                 split='test',
                                 seed=2022)

    raw_train_loader = DataLoader(train_dataset, batch_size=2,
                                  shuffle=False,
                                  num_workers=2)

    # Step 2: Get raw output
    print("Get raw train outputs ")
    _, labels, outputs, outputs_fts = evaluate_fn(model, raw_train_loader, device)
    probagation_labels = create_filtering_graph(train_files, outputs, outputs_fts, model_gnn)
    print(outputs.shape, probagation_labels.shape)
    print('Probagation labels = ', np.min(probagation_labels), np.mean(probagation_labels), np.max(probagation_labels))
    print('Raw labels = ', np.min(outputs[:, 1]), np.max(outputs[:, 1]))

    raw_p = 0.5*outputs[:, 1] + 0.5*probagation_labels
    threshold = np.percentile(raw_p, percentile)
    print("Threshold = ", threshold)
    print(np.mean(outputs[:, 1]), np.mean(probagation_labels))

    # Step 3: Create noisy labels
    noisy_labels = []
    for i in range(len(train_files)):
        data = train_files[i]
        video_name = data['video_name']

        if 'Train' in video_name:
            tmp_label = 0
        else:
            if raw_p[i] > threshold:
                tmp_label = 1
            else:
                tmp_label = 0

        noisy_labels += [tmp_label]

    #print("Got %i 0-labels, %i 1-label" % (len(noisy_labels) - sum(noisy_labels),
    #                                       sum(noisy_labels)))

    #print('Confusion matrix on train data: ')
    #print(metrics.confusion_matrix(train_labels, noisy_labels))

    # Step 4: Run training with noisy labels
    train_model(model, train_files, noisy_labels, device, epochs=2)
    test_model(model, test_files, device)


def run_baseline(model, train_files, test_files, device):
    # Step 1: Create base dataset (train with no shuffle)
    train_labels = [a['label'] for a in train_files]
    print("Train labels: Got %i 0-labels, %i 1-label" % (len(train_labels) - sum(train_labels),
                                           sum(train_labels)))
    train_dataset = VideoDataset(train_files,
                                 train_labels,
                                 8,
                                 224,
                                 'cpu',
                                 split='test',
                                 seed=2022)

    raw_train_loader = DataLoader(train_dataset, batch_size=2,
                                  shuffle=False,
                                  num_workers=2)

    # Step 2: Get raw output
    print("Get raw train outputs ")
    _, labels, outputs, outputs_fts = evaluate_fn(model, raw_train_loader, device)

    # Step 3: Create noisy labels
    noisy_labels = []
    for i in range(len(train_files)):
        data = train_files[i]
        video_name = data['video_name']

        if 'Train' in video_name:
            tmp_label = 0
        else:
            tmp_label = 1

        noisy_labels += [tmp_label]

    # Step 4: Run training with noisy labels
    train_model(model, train_files, noisy_labels, device, epochs=2)
    test_model(model, test_files, device)


# In[15]:


def train_with_real_labels():
    train_labels = [a['label'] for a in train_files]
    device = 'cuda'
    train_model(model, train_files, train_labels, device, epochs=1)
    test_model(model, test_files, device)


if __name__ == "__main__":
    parser = arg_parser()
    #args = parser.parse_args(['--backbone_net', 'resnet', '-d', '50', '--temporal_module_name', 'TAM',
    #                      '--groups', '8', '--no_imagenet_pretrained'])

    args = parser.parse_args()
    seed_everything(args.seed)

    TRAIN_FOLDER = args.train_folder
    TEST_FOLDER = args.test_folder

    os.listdir(TRAIN_FOLDER)

    print("Load dataset")
    train_files, train_test_files = get_train_files(TRAIN_FOLDER, 8, 2)
    test_train_files, test_files = get_test_files(TEST_FOLDER, 8, 2)
    train_files += test_train_files
    test_files += train_test_files

    print("Got %i train files, %i test files" % (len(train_files), len(test_files)))

    args.num_classes = 2

    model, arch_name = build_model(args, test_mode=True)

    checkpoint = torch.load('./K400-TAM-ResNet-50-f32.pth', map_location='cpu')

    pretrained = checkpoint['state_dict']
    model_dict = model.state_dict()
    for k, v in model_dict.items():
        if not (k == 'fc.weight' or k == 'fc.bias'):
            model_dict[k] = pretrained[k]

    model.load_state_dict(model_dict)
    model = model.to('cuda')

    device = 'cuda'
    use_model = args.use_model
    if use_model == 'baseline':
        run_baseline(model, train_files, test_files, device)
    else:
        for t in range(args.n_filters):
            filtering_week_supervised(model, train_files, test_files, device, 0.5, use_model)
