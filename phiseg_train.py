# -e git+https://github.com/lmkoch/medpy/@b06b6decf41c63489e746f6a83e8fa5ff509adfa#egg=MedPy
import logging

from importlib.machinery import SourceFileLoader
import argparse

from data.data_switch import data_switch
import os
import config.system as sys_config
import shutil
import utils

from phiseg import phiseg_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

def main(exp_config):

    logging.info('**************************************************************')
    logging.info(' *** Running Experiment: %s', exp_config.experiment_name)
    logging.info('**************************************************************')

    # Get Data
    data_loader = data_switch(exp_config.data_identifier)
    data = data_loader(exp_config)

    # Create Model
    phiseg = phiseg_model.phiseg(exp_config)

    # Fit model to data
    phiseg.train(data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for training")
    parser.add_argument("EXP_PATH", type=str, help="Path to experiment config file")
    args = parser.parse_args()
    # parser什么的在pycharm里的用法：https://blog.csdn.net/oMoDao1/article/details/83316546

    config_file = args.EXP_PATH

    #直接配置
    #config_file = "phiseg/experiments/phiseg_7_5.py"
    #config_file = "phiseg/experiments/phiseg_test_1annot_512.py"
    config_file = "phiseg/experiments/phiseg_test_1annot.py"

    config_module = config_file.split('/')[-1].rstrip('.py')

    exp_config = SourceFileLoader(config_module, config_file).load_module()  # 这是个什么鬼玩意，有问题啊。。

    log_dir = os.path.join(sys_config.log_root, exp_config.log_dir_name, exp_config.experiment_name)
    utils.makefolder(log_dir)


    shutil.copy(exp_config.__file__, log_dir)
    logging.info('!!!! Copied exp_config file to experiment folder !!!!')

    main(exp_config=exp_config)

