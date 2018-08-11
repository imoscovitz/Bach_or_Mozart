import pandas as pd
import math
import numpy as np
import copy
import time

import warnings
warnings.filterwarnings("ignore")

"""
Pruning and non-pruning implementations of Cohen's IREP.
See http://www.cs.utsa.edu/~bylander/cs6243/cohen95ripper.pdf
"""

##########################
##### MAIN FUNCTIONS #####
##########################


CLASS_NAME = 'Composer'
POS = 'Bach'
NEG = 'Mozart'

def ire(df, display=False, sleep=False):
    """
    Learns a ruleset in disjunctive normal form
    """

    ruleset = []
    remaining = df.copy()
    while npos(remaining) > 0:
        if display: print('Remaining pos:',npos(remaining))
        new_rule = grow_rule(remaining, display)
        if display: print('Grown:',new_rule,'\n')
        new_rule_covers = rule_covers(new_rule, remaining)
        rule_covers_pos = pos(new_rule_covers)
        remaining.drop(rule_covers_pos.index, axis=0, inplace=True)
        ruleset.insert(0, new_rule)
    return ruleset

def irep(df, display=False, sleep=False):
    """
    With pruning!
    New and impruned.
    """

    ruleset = []
    remaining = df.copy()
    while npos(remaining) > 0:
        if len(ruleset) > 30: return ruleset
        growset, pruneset = train_test_split(remaining, .33)
        grown_rule = grow_rule(growset, display=display, sleep=sleep)
        if display: print("Grown:",grown_rule)
        if sleep: time.sleep(2)
        pruned_rule = prune_rule(grown_rule, pruneset)
        if display: print("Pruned to:",pruned_rule)
        if sleep: time.sleep(3)
        prune_precision = rule_precision(pruned_rule, pruneset)
        if not prune_precision or prune_precision < .50:
            return ruleset
        else:
            ruleset.append(pruned_rule)
            remaining.drop(rule_covers(pruned_rule, remaining).index, axis=0, inplace=True)
            if display: print("Updated ruleset:",ruleset,'\n')
            if sleep: time.sleep(3)
    return ruleset

########################
##### ACTION-STEPS #####
########################

def grow_rule(df, display=False, sleep=False):
    rule = []
    while rule_covered(rule, neg(df)) > 0:
        if display: print("Update:",rule)
        if sleep: time.sleep(1)
        rule.append(best_successor(rule, df))
        if len(rule) > 30: return ruleset
    return rule

def prune_rule(rule, df):
    # Best pruned rule
    best_pruned_rule = copy.deepcopy(rule)
    # Best pruned rule value
    best_v = 0
    # Cond subset to try next
    cond_subset = rule

    while cond_subset:
        v = prune_value(cond_subset, df)
        if v > best_v:
            best_v = v
            best_pruned_rule = copy.deepcopy(cond_subset)
        cond_subset.pop()

    return best_pruned_rule

############################
##### ACTION-STEP EVAL #####
############################

# Returns the information gain from rule0 to rule1
def gain(rule0, rule1, df):
    p0count = rule_covered(rule0, pos(df))
    p1count = rule_covered(rule1, pos(df))
    n0count = rule_covered(rule0, neg(df))
    n1count = rule_covered(rule1, neg(df))
    return p1count * (math.log2((p1count + 1) / (p1count + n1count + 1)) - math.log2((p0count + 1) / (p0count + n0count + 1)))

# Returns for a rule its best successor rule based on information gain
def best_successor(rule, df):
    best_gain = 0
    best_step = []

    for step in successor_steps(rule, df):
        successor = rule+[step]
        g = gain(rule, successor, df)
        if g > best_gain:
            best_gain = g
            best_step = step

    return best_step

# Returns the prune value calculation of a rule
def prune_value(rule, pruneset):
    pos_pruneset = pos(pruneset)
    neg_pruneset = neg(pruneset)
    P = len(pos_pruneset)
    N = len(neg_pruneset)
    p = rule_covered(rule, pos_pruneset)
    n = rule_covered(rule, neg_pruneset)
    return (p+(N - n)) / (P + N)

#######################
##### PERFORMANCE #####
#######################

def rule_precision(rule, df):
    covered = rule_covers(rule, df)
    if len(covered) == 0:
        return None
    else:
        return npos(covered) / len(covered)

def rule_performance(rule, df):
    return precision(rule, df)

def precision(ruleset, df):
    covered = ruleset_covers(ruleset, df)
    return npos(covered) / len(covered)

def recall(ruleset, df):
    if len(df) == 0:
        return None
    else:
        n_pos = npos(df)
        tp = npos(ruleset_covers(ruleset, df))
        return tp/n_pos

def predict(ruleset, df):
    covered = ruleset_covers(ruleset, df)
    uncovered = df.drop(covered.index, axis=0)
    return covered, uncovered

def performance(ruleset, df):
    return {'Precision':precision(ruleset, df),
            'Recall':recall(ruleset, df)}

def experiment(df, n=10, ttsplit=.3, display=False, sleep=False):
    results = pd.DataFrame()
    rulesets = []
    precisions = []
    recalls = []

    for i in range(n):
        print('test',str(i+1))
        train, test = train_test_split(df,test_percent=ttsplit)
        ruleset = irep(train, display=display, sleep=sleep)
        perform = performance(ruleset, test)
        precision, recall = perform['Precision'], perform['Recall']
        print(perform)
        print(precision, recall)

        ruleset = sort_R(ruleset, reverse=True)
        ruleset = str(ruleset).replace('"','').replace("'","")
        rulesets.append(str(ruleset))
        precisions.append(precision)
        recalls.append(recall)

    results['Ruleset'] = rulesets
    results['Precision'] = precisions
    results['Recall'] = recalls

    groups = results.groupby('Ruleset')
    cons_results = pd.DataFrame()
    cons_results['Precision'] = groups['Precision'].mean()
    cons_results['Recall'] = groups['Recall'].mean()
    cons_results['Freq_Selected'] = groups['Ruleset'].count()/len(results)
    cons_results = cons_results.sort_values('Freq_Selected',ascending=False)
    cons_results.sort_values('Freq_Selected',ascending=False,inplace=True)

    return cons_results

# Randomly assigns train and test sets according to test_percent split
def train_test_split(df, test_percent, seed=None):
    new_df = df.copy()
    train_df = pd.DataFrame(columns=df.columns)
    test_df = pd.DataFrame(columns=df.columns)

    np.random.seed(seed=seed)
    _rand_ = np.random.rand(len(new_df))
    new_df['_rand_'] = _rand_
    _rand_ = np.sort(_rand_)
    cutoff = _rand_[int(len(_rand_)*test_percent)]

    train_df = new_df[new_df['_rand_'] > cutoff]
    test_df = new_df[new_df['_rand_'] <= cutoff]
    train_df = train_df.drop('_rand_',axis=1)
    test_df = test_df.drop('_rand_',axis=1)

    return train_df, test_df

###################
##### HELPERS #####
###################

# Returns positive class members
def pos(df):
    return df[df[CLASS_NAME] == POS]

# Returns negative class members
def neg(df):
    return df[df[CLASS_NAME] == NEG]

# Returns count of positive class members
def npos(df):
    return len(pos(df))

# Returns count of negative class members
def nneg(df):
    return len(neg(df))

# Returns df subset that is covered by a cond
def cond_covers(cond, df):
    if not cond:
        return df
    else:
        return df[df[cond[0]]==cond[1]]

# Returns number of df examples covered by a cond
def cond_covered(cond, df):
    return len(cond_covers(cond,df))

# Returns df subset that is covered by a rule
def rule_covers(rule, df):
    if not rule:
        return df
    elif len(rule) == 1:
        return cond_covers(rule[0], df)
    else:
        return rule_covers(rule[1:], cond_covers(rule[0], df))

# Returns number of df examples covered by a rule
def rule_covered(rule, df):
    return len(rule_covers(rule, df))

# Returns df examples covered by a rule
def ruleset_covers(ruleset, df):
    if not ruleset:
        return df
    else:
        covered_df = rule_covers(ruleset[0], df).copy()
        for rule in ruleset[1:]:
            covered_df = covered_df.append(rule_covers(rule, df))
        return covered_df

# Returns number of df examples covered by a ruleset
def ruleset_covered(ruleset, df):
    return len(ruleset_covers(ruleset, df))

# Returns a list of action steps
def successor_steps(rule, df):
    steps = []
    for col in df.columns.values:
        for value in df[col].unique():
            if col != CLASS_NAME:
                cond = (col, value)
                steps.append(cond)
    return [step for step in steps if step not in rule]

# Recursively sort nested lists alphabetically for consolidating experiment results
def sort_R(item, reverse=False):
    def contains_list(item):
        has_list = False
        for thing in item:
            if type(thing) == list:
                return True
        return False

    if type(item) != list:
        return str(item)
    elif not contains_list(item):
        return sorted([str(thing) for thing in item], reverse=reverse)
    else:
        return sorted([sort_R(element, reverse=reverse) for element in item], reverse=reverse)

########################################
##### BONUS: FUNCTIONS FOR BINNING #####
########################################

def fit_bins(df, n_bins=5, output=False, ignore_feats=[]):
    """
    Returns a dict definings fits for numerical features
    A fit is an ordered list of tuples defining each bin's range

    Returned dict allows for fitting to training data and applying the same fit to test data
    to avoid information leak.
    """

    def bin_fit_feat(df, feat, n_bins=5):
        """
        Returns list of tuples defining bin ranges for a numerical feature
        """
        bin_size = len(df)//n_bins
        sorted_df = df.sort_values(by=[feat])
        sorted_values = sorted_df[feat].tolist()

        bin_ranges = []
        for bin_i in range(n_bins):
            start_i = bin_size*bin_i
            finish_i = bin_size*(bin_i+1)
            start_val = sorted_values[start_i]
            finish_val = sorted_values[finish_i]
            bin_range = (start_val, finish_val)
            bin_ranges.append(bin_range)
        return bin_ranges

    # Create dict to store fit definitions for each feature
    fit_dict = {}
    feats_to_fit = df.dtypes[(df.dtypes=='float64') | (df.dtypes=='int64')].index.tolist()
    feats_to_fit = [feat for feat in feats_to_fit if feat not in ignore_feats]

    # Collect fits in dict
    count=1
    for feat in feats_to_fit:
        fit = bin_fit_feat(df, feat, n_bins=n_bins)
        fit_dict[feat] = fit
        if output and not count%100: print('Feature',feat,',',count,'of',len(feats_to_dict),'done.')
    return fit_dict

def bin_transform(df, fit_dict, names_precision=2):
    """
    Uses a pre-collected dictionary of fits to transform df features into bins
    """

    def bin_transform_feat(df, feat, bin_fits, names_precision=names_precision):
        """
        Returns new dataframe with n_bin bins replacing each numerical feature
        """

        def renamed(bin_fit_list, value, names_precision=names_precision):
            """
            Returns bin string name for a given numberical value
            Assumes bin_fit_list is ordered
            """
            min = bin_fit_list[0][0]
            max = bin_fit_list[-1][1]
            for bin_fit in bin_fits:
                if value <= bin_fit[1]:
                    start_name = str(round(bin_fit[0], names_precision))
                    finish_name = str(round(bin_fit[1], names_precision))
                    bin_name = '-'.join([start_name, finish_name])
                    return bin_name
            if value < min:
                return min
            elif value > max:
                return max
            else:
                raise ValueError('No bin found for value', value)

        renamed_values = []
        for value in df[feat]:
            bin_name = renamed(bin_fits, value, names_precision)
            renamed_values.append(bin_name)

        return renamed_values

    # Replace each feature with bin transformations
    for feat, bin_fits in fit_dict.items():
        feat_transformation = bin_transform_feat(df, feat, bin_fits, names_precision=names_precision)
        df[feat] = feat_transformation
    return df
