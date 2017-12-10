import random

import numpy as np

class CrossValidationLoader(object):
    def __init__(self, dataset, n_folds=5, shuffle=False, group_by=None,
                 random_seed=42, num_workers=4, transform=None):

        self.dataset = dataset
        self.n_folds = n_folds
        self.shuffle = shuffle
        self.group_by = group_by
        self.random_seed = random_seed
        self.num_workers = num_workers
        self.transform = transform

        if shuffle:
            random.seed(random_seed)
            random.shuffle(self.dataset)

        self.current_fold = -1
        self.folds = list()
        self.next_fold()

        self.train_batch_pos = 0
        self.valid_batch_pos = 0

    def next_fold(self):
        if self.current_fold > self.n_folds - 1:
            return None

        if len(self.folds) == 0:
            all_indices = range(len(self.dataset))
            fold_size = len(self.dataset) / self.n_folds
            for fi in range(self.n_folds):
                if fi != self.n_folds - 1:
                    fstart = fi*fold_size
                    fend = fstart + fold_size
                    self.folds.append(all_indices[fstart:fend])
                else:
                    fstart = fi*fold_size
                    self.folds.append(all_indices[fstart:])

        self.current_fold += 1
        self.valid_indices = self.folds[self.current_fold]
        self.train_indices = list()
        for i in range(len(self.folds)):
            if i != self.current_fold:
                self.train_indices.extend(self.folds[i])

        self.train_batch_pos = 0
        self.valid_batch_pos = 0

    def fold(self):
        done = False
        while self.current_fold < self.n_folds and not done:
            yield self
            if self.current_fold != self.n_folds - 1:
                self.next_fold()
            else:
                done = True

    def __len__(self):
        return len(self.dataset)

    def has_next_train_batch(self):
        return self.train_batch_pos < len(self.train_indices)

    def next_train_batch(self, batch_size):
        start = self.train_batch_pos
        end = self.train_batch_pos + batch_size
        self.train_batch_pos += batch_size

        batch_indices = self.train_indices[start:end]
        batch = [self.dataset[i] for i in batch_indices]

        return batch

    def has_next_valid_batch(self):
        return self.valid_batch_pos < len(self.valid_indices)

    def next_valid_batch(self, batch_size):
        start = self.valid_batch_pos
        end = self.valid_batch_pos + batch_size
        self.valid_batch_pos += batch_size

        batch_indices = self.valid_indices[start:end]
        batch = [self.dataset[i] for i in batch_indices]

        return batch

    def train(self, batch_size):
        while self.has_next_train_batch():
            yield self.next_train_batch(batch_size)

    def valid(self, batch_size):
        while self.has_next_valid_batch():
            yield self.next_valid_batch(batch_size)

if __name__ == '__main__':
    l = range(10)

    cv_loader = CrossValidationLoader(l, shuffle=True)

    for fold in cv_loader.fold():
        print('Train fold {}'.format(fold.current_fold))
        print('train')
        for batch in fold.train(4):
            print(batch)
        print('valid')
        for batch in fold.valid(4):
            print(batch)

    from sklearn.datasets import load_iris

    iris = load_iris()
    iris_list = [(iris['target'][i], iris['data'][i]) for i in range(150)]
    cv_loader = CrossValidationLoader(iris_list, shuffle=True)

    for fold in cv_loader.fold():
        for batch in fold.train(4):
            print(batch)
        for batch in fold.valid(4):
            print(batch)
