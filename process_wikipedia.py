import csv
from tqdm import tqdm
import pickle
import argparse

def convert_passages_to_dict(args):
    id2t = {}
    with open(args.input_path, "r") as fd:
        rd = csv.reader(fd, delimiter="\t")
        for idx, row in enumerate(tqdm(rd)):
            if idx == 0:
                continue
            id2t[int(row[0])] = str(row[1])

    with open("wiki_id2text.pickle", "wb") as f:
        pickle.dump(id2t, f, protocol=pickle.HIGHEST_PROTOCOL)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type = str, default = "./downloads/data/wikipedia-split/psgs_w100.tsv", required = True,
                        help = "Path of the wikipedia passages in tsv format.")
    args, unknown = parser.parse_known_args()
    return args

def main():
    args = get_args()
    convert_passages_to_dict(args)
    
if __name__ == "__main__":
    main()