import csv
import random
import argparse

DATA_DIR = "../data/txt/office_home"

parser = argparse.ArgumentParser(description='SSDA Classification')
parser.add_argument('--source', type=str, default='Art',
                    help='source domain')
parser.add_argument('--budget', type=int, default=3,
                    help='number of labeled examples in the target')
parser.add_argument('--seed', type=int, default=0,
                    help='number of labeled examples in the target')
args = parser.parse_args()

all_info = []
classes = set()
with open(f"{DATA_DIR}/labeled_source_images_{args.source}.txt", "r") as csvfile:
    reader = csv.reader(csvfile, delimiter=' ')
    for row in reader:
        all_info.append(row)
        classes.add(int(row[1]))

labeled_num = len(classes) * args.budget
valid_num = len(classes) * 3  # fix to 3 by default like in paper

with open(f"{DATA_DIR}/labeled_target_images_{args.source}mix{args.seed}_{args.budget}.txt", "w") as lfile, open(f"{DATA_DIR}/unlabeled_target_images_{args.source}mix{args.seed}_{args.budget}.txt", "w") as ufile, open(f"{DATA_DIR}/validation_target_images_{args.source}mix{args.seed}_3.txt", "w") as vfile:
    lwriter = csv.writer(lfile, delimiter=' ')
    uwriter = csv.writer(ufile, delimiter=' ')
    vwriter = csv.writer(vfile, delimiter=' ')
    random.seed(args.seed)
    random.shuffle(all_info)
    for i, row in enumerate(all_info):
        if i < labeled_num:
            lwriter.writerow(row)
        else:
            uwriter.writerow(row)
            if i < labeled_num + valid_num:
                vwriter.writerow(row)
