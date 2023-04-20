
import argparse
import os
from subprocess import check_call
import sys

from utils import Params


PYTHON = sys.executable
parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', default='./experiments/base_model/',
                    help="Directory containing params.json")
parser.add_argument('--data_dir', default='../datasets/datasets/',
                    help="Directory containing the dataset")


def launch_training_job(parent_dir, data_dir, job_name, params):

    model_dir = os.path.join(parent_dir, job_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    params.dataset = "wormml_v2_" + str(params.random_seed) + "_vgg_16_scratch/"
    json_path = os.path.join(model_dir, 'params.json')
    params.save(json_path)
    data_dir = 'wormml_v2_' + str(params.random_seed) + '_vgg_16_scratch/'

    cmd = "{python}CreateDatasetV2.py --random_seed {random_seed} --data_dir {data_dir}".format(python='D:/pycode/WormML-vgg19/',
                                                                                   random_seed=params.random_seed,
                                                                                   data_dir=data_dir)
    print(cmd)
    check_call(cmd, shell=True)

    cmd = "{python}wormml_dsnt.py --model_dir {model_dir} ".format(python='D:/pycode/WormML-vgg19/dsnt/',
            model_dir=model_dir)
    print(cmd)
    check_call(cmd, shell=True)


if __name__ == "__main__":

    args = parser.parse_args()
    json_path = os.path.join(args.parent_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    random_seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    for random_seed in random_seeds:


        params.random_seed = random_seed

        job_name = "data_random_seed_{}".format(random_seed)

        launch_training_job(args.parent_dir, args.data_dir, job_name, params)