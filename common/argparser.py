import argparse

def get_args():
    parser = argparse.ArgumentParser(
            formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-w', type=str, required=False, dest='file_weights')
    parser.add_argument('-n', type=int, required=False, dest='dev_size')
    parser.add_argument('-c', type=int, required=False, dest='training_window')
    parser.add_argument('-e', type=int, required=False, dest='training_epochs')
    parser.add_argument('-b', type=int, required=False, dest='batch_size')
    parser.add_argument('-l', type=float, required=False, dest='lr')
    parser.add_argument('-o', type=str, required=False, dest='dir_out')
    parser.add_argument('-d', type=int, required=False, dest='arch_depth')
    parser.add_argument('-f', type=int, required=False, dest='filters_per_layer')
    parser.add_argument('-p', type=str, required=False, dest='dir_dataset')
    parser.add_argument('-v', type=int, required=False, dest='flag_eval_only')
    args = parser.parse_args()

    return args
