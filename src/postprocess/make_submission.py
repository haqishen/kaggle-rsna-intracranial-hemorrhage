import sys
import os
import argparse
import pickle
import time
import glob
import pandas as pd
import numpy as np

from ..utils import mappings


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument('--inputs', default='exp/effnet_b1_512_p2', help='workdir name, split by `,`')
    parser.add_argument('--output', default='exp/effnet_b1_512_p2/exp/effnet_b1_512_p2_5fold_5tta.csv', required=True)
    parser.add_argument('--sample_submission', default='/data/data/RSNA/stage_2_sample_submission.csv')
    parser.add_argument('--clip', type=float, default=1e-6)

    args = parser.parse_args()
    assert args.input or args.inputs
    return args


def avg_predictions(results):
    outputs_all = np.array([result['outputs'] for result in results])
    outputs = outputs_all.mean(axis=0)
    return {
        'ids': results[0]['ids'],
        'outputs': outputs,
    }


def read_prediction(path):
    print('loading %s...' % path)
    with open(path, 'rb') as f:
        results = pickle.load(f)
    return avg_predictions(results)
    

def parse_inputs(inputs):
    results = []
    for elem in inputs:
        if type(elem) is list:
            result = parse_inputs(elem)
        else:
            result = read_prediction(elem)
        results.append(result)
    return avg_predictions(results)


def main():
    args = get_args()

    if args.input:
        result = read_prediction(args.input)
    else:
        folders = args.inputs.split(',')
        inputs = []
        for folder in folders:
            inputs += glob.glob(f'{folder}/*ep2_test_tta5.pkl')
        result = parse_inputs(inputs)

    sub = pd.read_csv(args.sample_submission)
    IDs = {}
    for id, outputs in zip(result['ids'], result['outputs']):
        for i, output in enumerate(outputs):
            label = mappings.num_to_label[i]
            ID = '%s_%s' % (id, label)
            IDs[ID] = output

    sub['Label'] = sub.ID.map(IDs)
    sub.loc[sub.Label.isnull(),'Label'] = sub.Label.min()
    if args.clip:
        print('clip values by %e' % args.clip)
        sub['Label'] = np.clip(sub.Label, args.clip, 1-args.clip)

    sub.to_csv(args.output, index=False)
    print(sub.tail())
    print('saved to %s' % args.output)


if __name__ == '__main__':
    print(sys.argv)
    main()
