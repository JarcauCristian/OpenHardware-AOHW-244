import glob
import os
import shutil

from tqdm import tqdm

from utils.log_utils import setup_logger
from utils.timing import timeit


def generate_train_test(src_train_genuine: str,
                        src_test_genuine: str,
                        src_train_attacked: str,
                        src_test_attacked: str,
                        dest_dir: str,
                        extension: str = "png") -> None:

    for directory in (src_train_genuine, src_test_genuine, src_train_attacked, src_test_attacked):
        assert os.path.exists(directory) and os.path.isdir(directory), f'Directory {directory} does not exist'

    shutil.rmtree(dest_dir, ignore_errors=True)
    os.makedirs(dest_dir)

    train_dir_genuine = os.path.join(dest_dir, 'train', '0=genuine')
    train_dir_attacked = os.path.join(dest_dir, 'train', '1=attacked')
    test_dir_genuine = os.path.join(dest_dir, 'test', '0=genuine')
    test_dir_attacked = os.path.join(dest_dir, 'test', '1=attacked')

    os.makedirs(train_dir_genuine)
    os.makedirs(train_dir_attacked)
    os.makedirs(test_dir_genuine)
    os.makedirs(test_dir_attacked)

    # copy genuine train files
    for file in glob.glob(os.path.join(src_train_genuine, '**', f'*.{extension}'), recursive=True):
        fname = os.path.split(file)[1]
        dest_name = os.path.join(train_dir_genuine, fname)
        shutil.copy(file, dest_name)

    # copy genuine test files
    for file in glob.glob(os.path.join(src_test_genuine, '**', f'*.{extension}'), recursive=True):
        fname = os.path.split(file)[1]
        dest_name = os.path.join(test_dir_genuine, fname)
        shutil.copy(file, dest_name)

    # copy attacked train files
    for file in glob.glob(os.path.join(src_train_attacked, '**', f'*.{extension}'), recursive=True):
        fname = os.path.split(file)[1]
        dest_name = os.path.join(train_dir_attacked, fname)
        shutil.copy(file, dest_name)

    # copy attacked test files
    for file in glob.glob(os.path.join(src_test_attacked, '**', f'*.{extension}'), recursive=True):
        fname = os.path.split(file)[1]
        dest_name = os.path.join(test_dir_attacked, fname)
        shutil.copy(file, dest_name)


@timeit
def main():

    # write genuine datasets

    dataset_root = '../data/german/'
    root_traffic_attacked: str = os.path.join(dataset_root, '_attacked')
    root_traffic_genuine: str = os.path.join(dataset_root, 'split')

    src_train_genuine = os.path.join(root_traffic_genuine, 'train')
    src_test_genuine = os.path.join(root_traffic_genuine, 'val')
    src_train_attacked = os.path.join(root_traffic_attacked, 'gtsrb_train')
    src_test_attacked = os.path.join(root_traffic_attacked, 'gtsrb_test')

    for attack_eps in tqdm(
            [
                'ResNet18_attacker=L1ProjectedGradientDescent_epsilon=0.01',
                            # 'ResNet18_attacker=L1ProjectedGradientDescent_epsilon=0.03',

                            'ResNet18_attacker=L2DeepFool_epsilon=0.01',
                            # 'ResNet18_attacker=L2DeepFool_epsilon=0.03',

                            'ResNet18_attacker=L2ProjectedGradientDescent_epsilon=0.01',
                            # 'ResNet18_attacker=L2ProjectedGradientDescent_epsilon=0.03',

                            # 'ResNet18_attacker=LinfFastGradient_epsilon=0.01',
                            # 'ResNet18_attacker=LinfFastGradient_epsilon=0.03',

                            'ResNet18_attacker=LinfProjectedGradientDescent_epsilon=0.01',
                            # 'ResNet18_attacker=LinfProjectedGradientDescent_epsilon=0.03'
                            ],
                           desc='Attacks'):

        generate_train_test(src_train_genuine=src_train_genuine,
                            src_test_genuine=src_test_genuine,
                            src_train_attacked=src_train_attacked,
                            src_test_attacked=src_test_attacked,
                            dest_dir=os.path.join(dataset_root, 'detect_attack', f'{attack_eps}'))


if __name__ == '__main__':
    caller_module: str = __file__.split(os.sep)[-1].split('.')[0]
    setup_logger(caller=caller_module)
    main()
