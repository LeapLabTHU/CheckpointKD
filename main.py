from args import args
import teacher
import student
from dataloader import get_test_loader, get_train_loader
from utils import prepare_dirs, save_config
import os
import numpy as np

import time

def main():
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    if args.train_teacher:
        test_data_loader = get_test_loader(
            args.data, args.data_root, args.batch_size, args.workers)
        if args.is_train:
            train_data_loader = get_train_loader(
                    args.data, args.data_root, args.batch_size,
                    args.seed, args.workers, args.shuffle, args.pin_memory
                )
            
            data_loader = (train_data_loader, test_data_loader)
        else:
            data_loader = test_data_loader

        trainer = teacher.Trainer(args, data_loader)

        if args.is_train:
            save_config(args)
            trainer.train()
        else:
            trainer.test()

    elif args.train_student:
        test_data_loader = get_test_loader(
            args.data, args.data_root, args.batch_size, args.workers)
        if args.is_train:
            train_data_loader = get_train_loader(
                    args.data, args.data_root, args.batch_size,
                    args.seed,  args.shuffle, args.workers, args.pin_memory
                )
            
            data_loader = (train_data_loader, test_data_loader)
        else:
            data_loader = test_data_loader

        trainer = student.Trainer(args, data_loader)
        save_config(args)
        if args.is_train:
            trainer.train()
        else:
            trainer.test()
        print('*******save: ', args.save)

    else:  		
        raise Exception('Unknown training mode (choices: train_teacher, train_student)')

    return


if __name__ == '__main__':
    main()