import numpy as np
import random

# utility function to find class with minimum examples
def min_examples_class(annot, dataset):
    min_cnt = annot.shape[0]+1
    min_lbls = []

    label_ex_count = np.zeros(annot.shape[1])
    for example in dataset:
        for lbl, tag in enumerate(annot[example, :]):
            if tag == 0:
                continue
            label_ex_count[lbl] += 1
    
    for lbl, cnt in enumerate(label_ex_count):
        if cnt > 0:
            if cnt < min_cnt:
                min_cnt = cnt
                min_lbls = [lbl]
            elif cnt == min_cnt:
                min_lbls.append(lbl)
    
    if len(min_lbls) > 0:
        return random.choice(min_lbls)
    else:
        return -1

# utility function to return examples containing a specific label
def examples_with_label(lbl, annot, dataset):
    examples = []

    for example in dataset:
        if annot[example, lbl] == 1:
            examples.append(example)

    return examples

# utility function to get target subset in which a given example
# should be inserted
def get_target_subset(desired_ex_cnt, desired_ex_cnt_lbl, target_lbl):
    max_lbl_desiring_subsets = []
    max_lbl_desire = -100000

    # finding the subset(s) in which the desired examples
    # of target_lbl are maximum
    for subset in range(len(desired_ex_cnt_lbl)):
        desire = desired_ex_cnt_lbl[subset][target_lbl]
        if desire > max_lbl_desire:
            max_lbl_desire = desire
            max_lbl_desiring_subsets = [subset]
        elif desire == max_lbl_desire:
            max_lbl_desiring_subsets.append(subset)

    max_desiring_subsets = []
    max_desire = -100000

    # breaking ties by finding those subsets from above
    # found subsets which have maximum number of desired examples (total)
    for subset in max_lbl_desiring_subsets:
        if desired_ex_cnt[subset] > max_desire:
            max_desire = desired_ex_cnt[subset]
            max_desiring_subsets = [subset]
        elif desired_ex_cnt[subset] == max_desire:
            max_desiring_subsets.append(subset)
    
    # further ties are broken randomly
    return random.choice(max_desiring_subsets)

# This function distributes the examples in the dataset among k subsets
# according to the proportions [prop[0], prop[1], ..., prop[k-1]]
# such that the number of examples of each label in a particular subset
# is propotional to the number of examples of that label in the original dataset
# For example, if k = 10 and prop = [0.1, 0.1, ..., 0.1], then each subset 
# will contain approx. 1/10th of the examples and for each label i, the number 
# of examples with label i in any subset will be approx. 1/10th of the number of  
# examples with the label i in the original dataset.
# k is number of subsets and prop is an array of proportions
# of examples in each subset 
def stratify(annot, dataset, k, prop):
    assert(k == len(prop))

    num_examples = annot.shape[0]
    num_classes = annot.shape[1]

    # initially dataset contains all examples
#     dataset = list(range(num_examples))

    desired_ex_cnt = []
    desired_ex_cnt_lbl = []

    # list of subsets, visually, subsets[i] is a list of examples
    # that are distributed to the i-th subset
    subsets = [[] for i in range(k)]

    for proportion in prop:
        # calculate desired number of examples in subset
        desired_ex_cnt.append(int(proportion * num_examples))

        # calculate desired number of examples of each label in subset
        desired_ex_cnt_subset = []
        for lbl in range(num_classes):
            num_examples_with_lbl = len(examples_with_label(lbl, annot, dataset))
            desired_ex_cnt_subset.append(int(num_examples_with_lbl * proportion))

        desired_ex_cnt_lbl.append(desired_ex_cnt_subset)
    
    while len(dataset) > 0:
        target_lbl = min_examples_class(annot, dataset)
        # print(target_lbl)
        if target_lbl == -1:
            break

        ex_with_target_lbl = examples_with_label(target_lbl, annot, dataset)
        for example in ex_with_target_lbl:
            subset = get_target_subset(desired_ex_cnt, desired_ex_cnt_lbl, target_lbl)

            # add example to the target subset
            subsets[subset].append(example)

            # remove example from the dataset
            dataset.remove(example)

            # for each label of example, decrement desired number of examples
            # for that label in the target subset by 1
            for lbl, tag in enumerate(annot[example, :]):
                if tag == 0:
                    continue
                desired_ex_cnt_lbl[subset][lbl] -= 1
            
            # decrement total desired number of examples of target subset by 1
            desired_ex_cnt[subset] -= 1

    return subsets

# returns list of negative labels corresponding to positive label lbl
# based on semantic paths
def get_negative_labels(lbl, sp, nl, annot, singleton_nodes):
    negative_labels = []
    images_count = annot.shape[0]

    label_considered = np.zeros((annot.shape[1],))
    for i, path in enumerate(sp):

        leaf_node = 0
        for j, node in enumerate(path):
            if node == lbl:
                leaf_node = -1
                lbl_layer = nl[i][j]
                for k in range(j, len(path)):
                    if nl[i][k] != lbl_layer:
                        leaf_node = k
                        break
                break
        if leaf_node == -1:
            continue
        if label_considered[path[leaf_node]] == 1:
            continue
        negative_labels.append(path[leaf_node])
        leaf_node_layer = nl[i][leaf_node]
        for j in range(leaf_node, len(path)):
            if nl[i][j] == leaf_node_layer:
                label_considered[path[j]] = 1
            else:
                break

    for i in singleton_nodes:
        if i == lbl or label_considered[i] == 1:
            continue
        negative_labels.append(i)
    
    return negative_labels

def positive_negative_split(lbl, count, sp, nl, annot, singleton_nodes):
    # get all samples which have label lbl
    positive_samples_all = examples_with_label(lbl, annot, list(range(annot.shape[0])))

    if len(positive_samples_all) <= count:
        # select all positive samples
        count = len(positive_samples_all)
        positive_samples = positive_samples_all
    else:
        # randomly sample 'count' number of unique examples
        positive_samples = random.sample(positive_samples_all, count)

    negative_labels = get_negative_labels(lbl, sp, nl, annot, singleton_nodes)

    # select only negative labels
    negative_annot = annot[:, negative_labels]

    label_count = negative_annot.sum(axis=1)
    
    # select only those images which have atleast one negative label and don't contain lbl
    negative_samples_dataset = []
    
    for img in range(annot.shape[0]):
        if label_count[img] > 0 and annot[img, lbl] == 0:
            negative_samples_dataset.append(img)

    total_negative_samples = len(negative_samples_dataset)
    prop = [count/total_negative_samples, 1-count/total_negative_samples]

    # divide the negative samples into two subsets the first of which
    # has approx. count number of samples using stratification
    subsets = stratify(negative_annot, negative_samples_dataset, 2, prop)
    negative_samples = subsets[0]

    return positive_samples, negative_samples

