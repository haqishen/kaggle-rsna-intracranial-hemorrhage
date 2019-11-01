import sys
import os
import argparse
import pickle
import time

import pandas as pd
import numpy as np
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='/Users/JP25565/Downloads/model001_fold0_ep1_test_tta5_policy2.csv', required=True)
    parser.add_argument('--output', default='./test_pseudo.csv', required=True)

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    df_submission = pd.read_csv(args.input)

    idx = []
    for i in tqdm(range(78545)):
        i_from = i*6
        if ((df_submission.loc[i_from : i_from+5].Label < 0.02) + (df_submission.loc[i_from : i_from+5].Label > 0.98)).sum() == 6:
            idx = idx + (np.arange(6) + i_from).tolist()
        
    df_pseudo = df_submission.loc[idx].reset_index(drop=True)
    df_pseudo.Label = (df_pseudo.Label > 0.98).astype(int)
    df_pseudo.to_csv(args.output, index=False)


if __name__ == '__main__':
    print(sys.argv)
    main()
