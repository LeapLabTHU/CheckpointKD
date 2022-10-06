import os
import glob
import time
import argparse

def str2bool(str):
	return True if str.lower() == 'true' else False
    
arg_parser = argparse.ArgumentParser(description='knowledge distillation')

###### kd mode related, whether to train teacher or train student
kd_group = arg_parser.add_argument_group('kd', 'knowledge distillation')
kd_group.add_argument('--train_teacher', action='store_true', default=False,
                       help='train teacher model only')
kd_group.add_argument('--train_student', action='store_true', default=False,
                       help='train student model directly from pretrained teacher model')



###### settings for teacher and student
# experiment setting related
exp_group = arg_parser.add_argument_group('exp', 'experiment setting')
exp_group.add_argument('--save', default='save/default-{}'.format(time.time()),
                       type=str, metavar='SAVE',
                       help='path to the experiment logging directory'
                       '(default: save/debug)')
exp_group.add_argument('--resume', action='store_true', default=None,
                       help='path to latest checkpoint (default: none)')
exp_group.add_argument('--print_freq', '-p', default=10, type=int,
                       metavar='N', help='print frequency (default: 10)')
exp_group.add_argument('--seed', default=0, type=int,
                       help='random seed')
exp_group.add_argument('--gpu', default='0', type=str, help='GPU available.')
# dataset related
data_group = arg_parser.add_argument_group('data', 'dataset setting')
data_group.add_argument('--data', metavar='D', default='cifar100',
                        choices=['cifar10', 'cifar100', 'tiny_imagenet', 'imagenet'],
                        help='data to work on')
data_group.add_argument('--data_root', metavar='DIR', default='/data/cx/data',
                        help='path to dataset (default: data)')
data_group.add_argument('--use_valid', action='store_true', default=False,
                        help='use validation set or not')
data_group.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
# model arch related
arch_group = arg_parser.add_argument_group('arch', 'model architecture setting')
arch_group.add_argument('--arch', type=str, default='resnet20')
# training related
optim_group = arg_parser.add_argument_group('t-optimization', 'optimization setting')
optim_group.add_argument('--epochs', default=200, type=int, metavar='N',
                         help='number of total epochs to run (default: 200)')
optim_group.add_argument('--start_epoch', default=0, type=int, metavar='N',
                         help='manual epoch number (useful on restarts)')
optim_group.add_argument('-b', '--batch_size', default=128, type=int,
                         metavar='N', help='mini-batch size (default: 128)')
optim_group.add_argument('--optimizer', default='sgd',
                         choices=['sgd', 'rmsprop', 'adam'], metavar='N',
                         help='optimizer (default=sgd)')
optim_group.add_argument('--lr', '--learning_rate', default=0.1, type=float,
                         metavar='LR',
                         help='initial learning rate (default: 0.1)')
optim_group.add_argument('--lr_type', default='multistep', type=str, metavar='T',
                        help='learning rate strategy (default: multistep)',
                        choices=['cosine', 'multistep'])
optim_group.add_argument('--decay_rate', default=0.1, type=float, metavar='N',
                         help='decay rate of learning rate (default: 0.1)')
optim_group.add_argument('--momentum', default=0.9, type=float, metavar='M',
                         help='momentum (default=0.9)')
optim_group.add_argument('--weight_decay', '--wd', default=5e-4, type=float,
                         metavar='W', help='weight decay (default: 5e-4)')


exp_group.add_argument('--pin_memory', action='store_true', default=False,
                      help='whether to copy tensors into CUDA pinned memory')                      

exp_group.add_argument('--is_train', type=str2bool, default='True',
                       help='Whether to train or test the model')
exp_group.add_argument('--test_path', default=None, type=str, 
                        help='path to the test model')


exp_group.add_argument('--shuffle',action='store_false', default=True,
                      help='Whether to shuffle the train indices')
exp_group.add_argument('--model_num', type=int, default=1,
                      help='Number of models to train ')
exp_group.add_argument('--save_epoch_teacher', action='store_true', default=False)
exp_group.add_argument('--best', action='store_false', default=True,
                      help='Load best model or most recent for testing')
exp_group.add_argument('--baseline', action='store_true', default=False,
                      help='normal training loss ')


###### settings for teacher only
teacher_group = arg_parser.add_argument_group('teacher', 'for teacher training only')
teacher_group.add_argument('--epo', type=int, default=20,
                      help='the gap between epochs to save checkpoint in mode "desemble epoch" ')


###### settings for student only
student_group = arg_parser.add_argument_group('student', 'for student training only')
student_group.add_argument('--teacher_arch', default='resnet18', type=str, help='teacher model') 
student_group.add_argument('--teacher_num', type=int, default=1,
                      help='the number of teachers ')
student_group.add_argument('--teacher_path', default='results/', type=str, 
                        help='path to the teacher')
student_group.add_argument('--T', type=float, default=5.0, help='temperature for Soft Target') 
student_group.add_argument('--alpha', type=float, default=0.5, help='alpha / trade-off parameter for kd loss')
exp_group.add_argument('--fully_trained', action='store_true', default=False, help='fully trained teacher for comparison ')
exp_group.add_argument('--desemble_epoch', action='store_true', default=False)
exp_group.add_argument('--avg_teacher', action='store_true', default=True)
exp_group.add_argument('--cos_lr', action='store_true', default=False)
exp_group.add_argument('--every_epoch_test', action='store_true', default=False)

exp_group.add_argument('--multistep_lr', action='store_true', default=False)



args = arg_parser.parse_args()

if args.use_valid:
    args.splits = ['train', 'val', 'test']
else:
    args.splits = ['train', 'val']

if args.data == 'cifar10':
    args.num_classes = 10
elif args.data == 'cifar100':
    args.num_classes = 100
else:
    args.num_classes = 1000



