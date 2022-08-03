import numpy as np
import json
import pickle
import random
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.optim as optim

from pose_network import Network

class CustomDataset(Dataset):
    """ Adjusted to load data according to selected storage concept.
        Incorporates pipeline to transform points into distance inbetween all of
        them and reduces dimensionality via PCA. """
    def __init__(self, paths, num_points=8, normalize=True):
        self.paths = paths
        self.num_points = num_points
        self.normalize = normalize
        distances, self.labels, self.class_map = self.dataset_setup(paths,\
                                                                    normalize)
        self.pca = PCA(n_components=num_points)
        self.distances = self.pca.fit_transform(distances)

    def __len__(self):
        return self.distances.shape[0]

    def __getitem__(self, idx):
        x = self.distances[idx]
        y = self.labels[idx]
        x_tensor = torch.from_numpy(x).type(torch.FloatTensor)
        return x_tensor, torch.tensor(y)

    def return_pca_param(self):
        return self.pca.get_params()

    def return_pca(self):
        return self.pca

    def get_distances(self, points):
        dist_matrix = cdist(points, points)
        dist_non_zero = dist_matrix[dist_matrix != 0].flatten()
        return dist_non_zero

    def dataset_setup(self, paths, normalize=True):
        all_points = []
        all_labels = []

        data_names = ['rock_pose_rh.json', 'rock_pose_lh.json', 'paper_pose_rh.json',\
                        'paper_pose_lh.json', 'scissors_pose_rh.json',\
                        'scissors_pose_lh.json']
        data_class = ['rock', 'rock', 'paper', 'paper', 'scissors', 'scissors']
        class_map = {0: 'rock', 1: 'paper', 2: 'scissors'}
        inv_class_map = {v: k for k, v in class_map.items()}

        for path in paths:
            for i, data_name in enumerate(data_names):
                with open(path+data_name, 'r') as f:
                    points = json.load(f)
                for point in points:
                    all_points.append(point)
                    all_labels.append(inv_class_map[data_class[i]])
        # points to distances
        all_distances = []
        for points in all_points:
            all_distances.append(self.get_distances(points))
        # normalize
        if normalize:
            norm = np.linalg.norm(all_distances, axis=1)
            all_distances = all_distances/norm[:,None]
        return all_distances, all_labels, class_map

def train_model(model, criterion, optimizer, dataloaders,batch_size, num_epochs=5):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase])
            epoch_acc = running_corrects.double() / len(dataloaders[phase])

            print('{} loss: {:.4f}, acc: {:.4f}'.format(phase, epoch_loss,\
                                                        epoch_acc/batch_size))
    return model


if __name__ == "__main__":
    # setup data loaders
    batch_size = 30
    save_path = '/home/niklas/Ablage/Own_projects/Hand_RPS/trained_model/'
    paths = ['/home/niklas/Ablage/Own_projects/Hand_RPS/1/',\
            '/home/niklas/Ablage/Own_projects/Hand_RPS/2/']

    dataset = CustomDataset(paths)
    # save pca parameters
    pca = dataset.return_pca()
    pickle.dump(pca, open(save_path+'pca_params.pkl',"wb"))
    # train-test-split
    lengths = [int(dataset.__len__()*0.8), int(dataset.__len__()*0.2)]
    data_train, data_test = torch.utils.data.random_split(dataset, lengths)
    data_loaders = {'train': torch.utils.data.DataLoader(data_train,\
                                                            batch_size=batch_size,\
                                                            shuffle=True),
                    'validation': torch.utils.data.DataLoader(data_test,\
                                                            batch_size=batch_size,\
                                                            shuffle=True)}
    # load neural Network
    model = Network().float()
    # train network
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    model_trained = train_model(model, criterion, optimizer, data_loaders, batch_size)
    # save model
    torch.save(model.state_dict(), save_path+'trained_model.pth')
