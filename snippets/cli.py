#!/usr/bin/python3
from argparse import ArgumentParser
from datetime import datetime
import logging
import os
import shutil
import sys
from decouple import config

""" Uncomment any one of the following imports and comment others to use a specific test generator
    Also, uncomment the function call of test generator
"""
# from mcts import MCTS
# from qlv0 import QLearningTestGenerator
# from qlv1 import QLearningGenerator
# from qlv2 import UCBGenerator
# from qlv3 import UCBGenerator
# from qlv4 import UCBGenerator
from qlv5 import UCBGenerator
# from qlv6 import UCBGenerator

TESTS_FOLDER = config("TESTS_FOLDER", default="./generated_tests/")
logger = logging.getLogger(__name__)


def arg_parse():
    main_parser = ArgumentParser(
        description="UAV Test Generator",
    )
    subparsers = main_parser.add_subparsers()
    parser = subparsers.add_parser(name="generate", description="generate tests")
    parser.add_argument("test", help="initial test description file address")

    parser.add_argument(
        "budget",
        type=int,
        help="test generation budget (total number of simulations allowed)",
    )

    args = main_parser.parse_args()
    return args


def config_loggers():
    os.makedirs("logs/", exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG,
        filename="logs/debug.txt",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    root = logging.getLogger()
    # terminal logs
    c_handler = logging.StreamHandler()
    c_handler.setLevel(logging.INFO)
    c_format = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    c_handler.setFormatter(c_format)
    root.addHandler(c_handler)

    # file logs
    f_handler = logging.FileHandler("logs/info.txt")
    f_handler.setLevel(logging.INFO)
    f_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    f_handler.setFormatter(f_format)
    root.addHandler(f_handler)


if __name__ == "__main__":
    config_loggers()
    try:
        args = arg_parse()

        """ 
            Uncomment the function call of chosen test generator
        """
        # generator = MCTS(case_study_file=args.test)
        # generator = QLearningTestGenerator(case_study_file=args.test)
        # generator = QLearningGenerator(case_study_file=args.test)
        generator = UCBGenerator(case_study_file=args.test)
        
        test_cases = generator.generate(args.budget)

        ### copying the test cases to the output folder
        tests_fld = f'{TESTS_FOLDER}{datetime.now().strftime("%d-%m-%H-%M-%S")}/'
        os.mkdir(tests_fld)        
        for i in range(len(test_cases)):
            test_cases[i].save_yaml(f"{tests_fld}/test_{i}.yaml")
            shutil.copy2(test_cases[i].log_file, f"{tests_fld}/test_{i}.ulg")
            shutil.copy2(test_cases[i].plot_file, f"{tests_fld}/test_{i}.png")
        print(f"{len(test_cases)} test cases generated")
        print(f"output folder: {tests_fld}")

    except Exception as e:
        logger.exception("program terminated:" + str(e), exc_info=True)
        sys.exit(1)
