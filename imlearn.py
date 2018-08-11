import random
import numpy as np
import copy
import pandas as pd

from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns

##############

def custom_train_test_split(df, split_on, test_size=0.3, random_seed=None):
    """
    Splits df into train and test sets along a specific attribute

    Note: It is possible, for instance, for movement 1 to land in train set
    and movement 2 of the same work in test set. Reasonable people might disagree
    whether to split on movements or entire works, but my opinion is that this the way to go.

    Note #2: Test_size is the size split for attribute values, which may differ from
    the actual train_test split. There is no way to ensure the actual split also matches
    without running a reasonably thoughtful search routine, which is would necessarily
    break the whole randomness thing.
    """

    # Find values to split on
    split_on_values = df[split_on].unique().tolist()

    # Shuffle
    random.seed(random_seed)
    random.shuffle(split_on_values)

    # Divvy up the values
    n_test_values = int(len(split_on_values) * test_size)
    test_values = split_on_values[:n_test_values]
    train_values = split_on_values[n_test_values:]

    # Divide into train and test dfs
    train_df = df[df[split_on].apply(lambda x: x in train_values)]
    test_df = df[df[split_on].apply(lambda x: x in test_values)]

    return train_df, test_df

def customfold_split(df, split_on, n_folds=5, random_seed=None):
    """
    Divide a dataframe into n_folds number of folds based on split_on attribute.
    """
    # Find values to split on
    split_on_values = df[split_on].unique().tolist()

    # Shuffle
    random.seed(random_seed)
    random.shuffle(split_on_values)

    # Create n_folds empty lists and assign split_on's values into lists
    folds = [[] for fold_i in range(n_folds)]
    while split_on_values:
        for fold in folds:
            if split_on_values: fold.append(split_on_values.pop())

    # Split full_train by folded values
    folded_dfs = []
    for fold in folds:
        new_df = df[df[split_on].apply(lambda x: x in fold)].copy()
        folded_dfs.append(new_df)

    return folded_dfs

class CustomCV():
    """
    A class for running cross-validation by splitting folds along on a particular feature
    """

    def __init__(self, clf, fold_on, scorer=accuracy_score, standard_scale=True):
        self.clf = clf
        self.fold_on = fold_on
        self.scorer = accuracy_score
        self.standard_scale = standard_scale

    def fit_score(self, train, y_feat, cv=5,
                omit_mask=None, display=False, random_seed=None):
        """
        Train and test cv number of folds using custom split
        """

        # Collect a list of dfs split on attribute fold_on
        folds = customfold_split(train, self.fold_on, n_folds=cv, random_seed=random_seed)

        # Make list of (train,val) tuples to evaluate on
        kf = k_folds(folds)

        # Evaluate on each fold
        all_results = []
        all_predictions = []
        all_actuals = []

        for index, (train, val) in enumerate(kf):

            # X_y split
            X_train, y_train = X_y_split(train, y_feat)
            X_val, y_val = X_y_split(val, y_feat)

            # Apply omit_masks
            X_train = masked(X_train, omit_mask)
            y_train = masked(y_train, omit_mask)
            X_val = masked(X_val, omit_mask)
            y_val = masked(y_val, omit_mask)

            # Scale (if applicable)
            if self.standard_scale:
                scaler = StandardScaler()
                scaler.fit(X_train)
                scaler.transform(X_train)
                scaler.transform(X_val)

            # Fit
            self.clf.fit(X_train, y_train)

            # Predict
            predictions_train = self.clf.predict(X_train)
            predictions_val = self.clf.predict(X_val)

            # Score
            train_score = self.scorer(y_train, predictions_train)
            val_score = self.scorer(y_val, predictions_val)

            # Store
            all_results.append((train_score, val_score))
            all_predictions += (list(predictions_val))
            all_actuals += y_val[y_feat].tolist()

            if display: print("Fold",index+1,'train_score:',train_score,'|','val_score:',val_score)

        # Return average validation score
        self.score_ = np.mean([val_score for train_score, val_score in all_results])
        self.train_score_ = np.mean([train_score for train_score, val_score in all_results])
        self.all_scores_ = all_results
        self.predictions_actuals_ = all_predictions, all_actuals

class CustomGridCV():

    #GridSearchCV implements a “fit” and a “score” method. It also implements “predict”, “predict_proba”, “decision_function”, “transform” and “inverse_transform” if they are implemented in the estimator used.

    """
    Class for implementing grid search cross-validation
    whose folds are split along a particular feature.
    """

    def __init__(self, clf, params, fold_on, scorer=accuracy_score, standard_scale=True):
        self.clf = clf
        self.params = params
        self.scorer = scorer
        self.fold_on = fold_on
        self.standard_scale = standard_scale

    def fit_score(self, train, y_feat, cv=5, omit_mask=None, display=False, random_seed=None):

        best_score = float("-inf")
        best_predictions_actuals =[]

        for g in ParameterGrid(self.params):

            # Set parameters
            self.clf.set_params(**g)

            # Cross-validate
            custom_cv = CustomCV(clf=self.clf, fold_on=self.fold_on, scorer=self.scorer, standard_scale=self.standard_scale)
            custom_cv.fit_score(train, y_feat=y_feat, cv=cv,
                                omit_mask=omit_mask, display=False, random_seed=random_seed)
            current_score = custom_cv.score_

            # Display results
            if display:
                print('validation score: {}'.format(current_score))
                print('grid: {}'.format(g))
                print()

            # Update the best_score if needed
            if current_score > best_score:
                best_score = current_score
                best_grid = g
                best_predictions_actuals = custom_cv.predictions_actuals_
                best_model = copy.deepcopy(self.clf)

        self.best_score_ = best_score
        self.best_parameters_ = best_grid
        self.best_predictions_actuals_ = best_predictions_actuals
        self.best_model_ = best_model

###################
##### HELPERS #####
###################

def X_y_split(df, y_feat):
    """
    Split df into X and y dfs.
    """

    X = df.drop(y_feat,axis=1)
    y = df[[y_feat]]
    return X,y

def k_folds(df_list):
    """
    Folds a list of dataframes,
    i.e. Returns list of (train,val) tuples
    """

    folds = []
    for index in range(len(df_list)):
        train = concat(df_list[:index]+(df_list[index+1:]))
        val = df_list[index]
        folds.append((train,val))
    return folds

def concat(df_list):
    """
    Concatenates a list of dataframes
    """

    new_df = df_list[0]
    for df in df_list[1:]:
        new_df = new_df.append(df)
    new_df.reset_index(inplace=True)
    new_df.drop('index',axis=1,inplace=True)
    return new_df

def masked(df, omit_mask=None):
    """
    Returns df without omit_mask features, if present
    """

    if omit_mask:
        for feat in omit_mask:
            if feat in df.columns: df = df.drop(feat,axis=1)
    return df

###################
##### DISPLAY #####
###################

def draw_ROC(model, X_test, y_test):
    precision_curve, recall_curve, threshold_curve = precision_recall_curve(y_test, model.predict_proba(X_test)[:,1] )

    plt.figure(dpi=150)
    plt.plot(threshold_curve, precision_curve[1:],label='precision')
    plt.plot(threshold_curve, recall_curve[1:], label='recall')
    plt.legend(loc='lower left')
    plt.xlabel('threshold (above this probability, label as fraud)');
    plt.title('Precision-recall curve')

def draw_confusion(predictions, actual, classes, title='Confusion matrix'):
    confusion = confusion_matrix(actual, predictions)
    plt.figure(dpi=75)
    sns.heatmap(confusion, cmap=plt.cm.Blues, fmt='d', annot=True, square=True,
               xticklabels=classes,
               yticklabels=classes)

    plt.xlabel('Predicted composer')
    plt.ylabel('Actual composer')
    plt.title(title)

def draw_learning_curve(train, y_feat, split_on, clf, scorer=accuracy_score, cv=5, omit_mask=None, random_seed=None, title=''):

    # Split train and val sets
    train_examples, val_examples = custom_train_test_split(train, split_on=split_on, test_size=0.25, random_seed=random_seed)

    # Train and validate
    train_sizes = []
    train_scores = []
    val_scores = []

    for fold_i in range(1, cv+1):

        train_size = len(train_examples)*fold_i//cv
        val = val_examples

        # X_y split
        X_train, y_train = X_y_split(train_examples[:train_size], y_feat)
        X_val, y_val = X_y_split(val, y_feat)

        # Apply omit_masks
        X_train = masked(X_train, omit_mask)
        y_train = masked(y_train, omit_mask)
        X_val = masked(X_val, omit_mask)
        y_val = masked(y_val, omit_mask)

        # Fit
        clf.fit(X_train, y_train)

        # Predict
        predictions_train = clf.predict(X_train)
        predictions_val = clf.predict(X_val)

        # Score
        train_score = scorer(y_train, predictions_train)
        val_score = scorer(y_val, predictions_val)

        # Record
        train_sizes.append(train_size)
        train_scores.append(train_score)
        val_scores.append(val_score)

    learn_df = pd.DataFrame({
    'train_size': train_sizes,
    'train_score': train_scores,
    'val_score': val_scores
    })

    plt.plot(learn_df['train_size'], learn_df['train_score'], 'r--o', label='train scores')
    plt.plot(learn_df['train_size'], learn_df['val_score'], 'b--x', label='val scores')
    plt.xlabel('Train Size')
    plt.ylabel('Score')
    plt.legend(loc='lower right')
    plt.title(title)
    plt.ylim(0,1);
