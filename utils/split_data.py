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

all_info = {}
with open(f"{DATA_DIR}/labeled_source_images_{args.source}.txt", "r") as csvfile:
    reader = csv.reader(csvfile, delimiter=' ')
    for row in reader:
        cls = int(row[1])
        if cls not in all_info:
            all_info[cls] = []
        all_info[cls].append(row[0])

with open(f"{DATA_DIR}/labeled_target_images_{args.source}{args.seed}_{args.budget}.txt", "w") as lfile, open(f"{DATA_DIR}/unlabeled_target_images_{args.source}{args.seed}_{args.budget}.txt", "w") as ufile, open(f"{DATA_DIR}/validation_target_images_{args.source}{args.seed}_3.txt", "w") as vfile:
    lwriter = csv.writer(lfile, delimiter=' ')
    uwriter = csv.writer(ufile, delimiter=' ')
    vwriter = csv.writer(vfile, delimiter=' ')
    random.seed(args.seed)
    for cls, lst in all_info.items():
        random.shuffle(lst)
        for i, v in enumerate(lst):
            if i < args.budget:
                lwriter.writerow([v, cls])
            else:
                uwriter.writerow([v, cls])
                if i < args.budget + 3:  # fix to 3 by default like in paper
                    vwriter.writerow([v, cls])
