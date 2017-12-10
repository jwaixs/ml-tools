import sklearn

import numpy as np

class Stack(object):
    '''Stack, the simple stacker function for classifiers.

    Args:
        classifiers (list of classifiers): List of classifiers to stack.
        n_splits (optional, int): Number of splits for cross validation.
            Defaults to 5.
        shuffle (optional, boolean): Shuffle dataset. Defaults to False.
        random_state (optional, int): Random seed. Defaults to 42.

    '''
    def __init__(self, classifiers, n_splits = 5, shuffle = False, random_state = 42):
        self.classifiers = classifiers
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def fit(self, X, y):
        folds = list(sklearn.model_selection.KFold(
            n_splits = self.n_splits,
            shuffle = True,
            random_state = self.random_state
        ).split(X, y))

        R = np.zeros((X.shape[0], len(self.classifiers)))

        for i, clf in enumerate(self.classifiers):
            for j, (train_idx, test_idx) in enumerate(folds):
                X_fold_train = X[train_idx]
                y_fold_train = y[train_idx]
                X_fold_test = X[test_idx]
                y_fold_test = y[test_idx]

                clf.fit(X_fold_train, y_fold_train)
                y_fold_pred = clf.predict(X_fold_test)[:]

                R[test_idx, i] = y_fold_pred

        return R

if __name__ == '__main__':
    from sklearn.linear_model import SGDClassifier
    from sklearn.svm import SVC, LinearSVC
    from sklearn.datasets import load_iris

    iris = load_iris()

    classifiers = [
        SVC(),
        SVC(decision_function_shape = 'ovo'),
        LinearSVC(),
        SGDClassifier()
    ]

    stack1 = Stack(classifiers, shuffle = True)
    X1 = stack1.fit(iris.data, iris.target)
    X1 = np.concatenate((X1, iris.data), axis = 1)
    print(X1)

    stack2 = Stack(classifiers, shuffle = True)
    X2 = stack2.fit(X1, iris.target)
    X2 = np.concatenate((X2, X1, iris.data), axis = 1)
    print(X2)

    stack3 = Stack([SVC()], shuffle = True)
    X3 = stack3.fit(X2, iris.target)
    print(X3)


    def check(prediction):
        correct = 0
        for i in range(150):
            if prediction[i] == iris.target[i]:
                correct += 1

        print('Correct: {}'.format(correct))
        print('Wrong: {}'.format(150 - correct))

    check(X3)
