import os
import json
import argparse
import random
import collections


def do_splits(input_file: str, output_file: str):
    with open(input_file, 'r') as file:
        input_splits = json.load(file)

    all_images = set()
    for fold_dict in input_splits:
        for image_name in fold_dict['val']:
            all_images.add(image_name)
    
    output_splits = create_new_splits(all_images)
    with open(output_file, 'w') as file:
        json.dump(output_splits, file)


def create_random_prefix_partition(strings: set[str], num_sets: int = 5) -> list[set[str]]:
    prefix_groups = collections.defaultdict(list)
    for s in strings:
        prefix = s.rsplit('_', 1)[0]
        prefix_groups[prefix].append(s)
    list_of_groups = list(prefix_groups.values())
    random.shuffle(list_of_groups)
    partition = [set() for _ in range(num_sets)]
    set_sizes = [(0, i) for i in range(num_sets)]
    for group in list_of_groups:
        set_sizes.sort()
        smallest_set_index = set_sizes[0][1]
        partition[smallest_set_index].update(group)
        set_sizes[0] = (set_sizes[0][0] + len(group), smallest_set_index)
    return partition

def create_new_splits(all_images: set):
    folds = create_random_prefix_partition(all_images)
    new_splits = []
    for i in range(len(folds)):
        train = []
        val = []
        for fold_index, fold in enumerate(folds):
            if fold_index == i:
                val.extend(list(fold))
            else:
                train.extend(list(fold))
        train.sort()
        val.sort()
        new_splits.append({'train': train, 'val': val})
    return new_splits


def main():
    parser = argparse.ArgumentParser(description='Create a custom data split for the mama mia data which puts all images from the same subject into the same group')
    parser.add_argument('input_split_file', type=str, help='Input split file')
    parser.add_argument('output_split_file', type=str, help='Output split file')
    args = parser.parse_args()

    do_splits(args.input_split_file, args.output_split_file)


if __name__ == '__main__':
    main()