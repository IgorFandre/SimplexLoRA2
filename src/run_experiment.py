import torch
import os
import sys
sys.path.append("src/libsvm")
sys.path.append("src/cv")
sys.path.append("src/fine_tuning/glue")

from config import parse_args
from utils import get_run_name
from libsvm import main_libsvm
from cv import main_cv
from fine_tuning.glue import main_glue

if __name__ == "__main__":
    if torch.cuda.is_available():
        print("~~~~~~~~~~~~~~~ GPU ~~~~~~~~~~~~~~~")
        for i in range(torch.cuda.device_count()):
            print(torch.cuda.get_device_name(i))
    else:
        print("~~~~~~~~~~~~~~~ USING CPU ~~~~~~~~~~~~~~~")
    args, parser = parse_args()
    args.run_name = get_run_name(args, parser)
    if args.problem.lower() == "libsvm":
        main_libsvm.main(args, parser)
    elif args.problem.lower() == "cv":
        main_cv.main(args, parser)
    elif args.problem.lower() == "fine-tuning" and args.dataset.lower() == "glue":
        main_glue.main(args)
    else:
        raise ValueError("Unsupported problem or dataset specified.")