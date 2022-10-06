# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

import os
import time
import shutil

from utils import accuracy, AverageMeter, set_logger, save_dict_to_json
import models
import logging
from tqdm import tqdm

import wandb

class Trainer(object):

    def __init__(self, args, data_loader):

        self.args = args

        # data params
        if args.is_train:
            self.train_loader = data_loader[0]
            self.valid_loader = data_loader[1]
            self.num_train = len(self.train_loader.dataset)
            self.num_valid = len(self.valid_loader.dataset)
        else:
            self.test_loader = data_loader
            self.num_test = len(self.test_loader.dataset)
        
        if self.args.data == 'cifar100':
            self.num_classes = 100
        elif self.args.data == 'cifar10':
            self.num_classes = 10
        else:         
            raise Exception('Unknown dataset (choices: cifar100, cifar10, esle dataset not available)')
   
        self.model_num = self.args.model_num
        self.gpu = self.args.gpu

        self.epochs = self.args.epochs
        self.start_epoch = self.args.start_epoch
        self.resume = self.args.resume
        self.save = self.args.save
        self.best = self.args.best
        self.arch = self.args.arch
        self.lr = self.args.lr
        self.momentum = self.args.momentum
        self.weight_decay = self.args.weight_decay
        self.decay_rate = self.args.decay_rate
        self.save_epoch_teacher = self.args.save_epoch_teacher
        self.epo = self.args.epo
        self.baseline = self.args.baseline

        self.models = []
        self.optimizers = []
        self.schedulers = []

        self.loss_kl = nn.KLDivLoss(reduction='batchmean')
        self.loss_ce = nn.CrossEntropyLoss()
        self.best_valid_accs = [0.] * self.model_num
        
        set_logger(os.path.join(self.args.save, 'train.log'))

        for i in range(self.model_num):
            model = getattr(models, self.arch)()
            if self.gpu is not None:
                # model.cuda()   
                model = torch.nn.DataParallel(model.cuda())
             
            self.models.append(model)

            optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay, nesterov=True)               
            self.optimizers.append(optimizer)
                
            scheduler = optim.lr_scheduler.StepLR(self.optimizers[i], step_size=60, gamma=self.decay_rate, last_epoch=-1)
            self.schedulers.append(scheduler)              

        print('[*] Number of parameters of one model: {:,}'.format(
            sum([p.data.nelement() for p in self.models[0].parameters()])))
    
    def train(self):
        # if --resume, load the most recent checkpoint
        if self.resume:
            self.load_checkpoint(best=False)

        print("\n[*] Train on {} samples, validate on {} samples".format(
            self.num_train, self.num_valid)
        )

        for epoch in range(self.start_epoch, self.epochs):

            for scheduler in self.schedulers:
                scheduler.step(epoch)
            
            print(
                '\nEpoch: {}/{} - LR: {:.6f}'.format(
                    epoch+1, self.epochs, self.optimizers[0].param_groups[0]['lr'],)
            )
            
            logging.info("Epoch {}/{}, lr:{}".format(epoch + 1, self.epochs, self.optimizers[0].param_groups[0]['lr']))

            # train for 1 epoch
            train_losses, train_accs = self.train_one_epoch(epoch)

            # evaluate on validation set
            valid_losses, valid_accs = self.validate(epoch)

            for i in range(self.model_num):
                is_best = valid_accs[i].avg> self.best_valid_accs[i]
                self.best_valid_accs[i] = max(valid_accs[i].avg, self.best_valid_accs[i])
                msg1 = "model_{:d}: train loss: {:.3f} - train acc: {:.3f} "
                msg2 = "- val loss: {:.3f} - val acc: {:.3f}"
                if is_best:
                    msg2 += " [*]"
                    logging.info("- Founds new best accuracy")
                    best_json_path = os.path.join(self.save, "eval_best_results_mdoel%s.json"%(i+1))
                    best_val_log = {'accuracy': self.best_valid_accs[i],'epoch': (epoch+1)}
                    save_dict_to_json(best_val_log, best_json_path)
                msg = msg1 + msg2
                print(msg.format(i+1, train_losses[i].avg, train_accs[i].avg, valid_losses[i].avg, valid_accs[i].avg))

                self.save_checkpoint(i,
                    {'epoch': epoch + 1,
                    'model_state': self.models[i].state_dict(),
                    'optim_state': self.optimizers[i].state_dict(),
                    'best_valid_acc': self.best_valid_accs[i],
                    }, is_best)
                
                if self.save_epoch_teacher:
                    if (epoch+1)%self.epo==0:
                        self.save_checkpoint(i,
                            {'epoch': epoch + 1,
                            'model_state': self.models[i].state_dict(),
                            'optim_state': self.optimizers[i].state_dict(),
                            'best_valid_acc': self.best_valid_accs[i],
                            }, is_best=False, desemble=True)  
                
                wandb.log({
                    "train_loss": train_losses[i].avg,
                    "train_acc": train_accs[i].avg,
                    "valid_loss": valid_losses[i].avg,
                    "valid_acc": valid_accs[i].avg,
                    })

        for i in range(self.model_num):
            print('best val top1 of model {:d}: {:.3f} '.format(i+1, self.best_valid_accs[i]))

    def train_one_epoch(self, epoch):
        train_data = self.train_loader
        batch_time = AverageMeter()
        losses = []
        ce_losses = []
        kd_losses = []
        accs = []

        for i in range(self.model_num):
            self.models[i].train()
            losses.append(AverageMeter())
            ce_losses.append(AverageMeter())
            kd_losses.append(AverageMeter())
            accs.append(AverageMeter())
        
        tic = time.time()
        with tqdm(total=self.num_train) as pbar:

            for i, (images, labels) in enumerate(train_data):
                if self.gpu is not None:
                    images, labels = images.cuda(), labels.cuda()
                images, labels = Variable(images), Variable(labels)
                    
                #forward pass
                outputs=[]
                for model in self.models:
                    outputs.append(model(images))
                    
                for i in range(self.model_num):
                    if self.baseline:
                        ce_loss = self.loss_ce(outputs[i], labels)
                        kd_loss = 0
                        loss = ce_loss
                    else:  
                        raise Exception('Unknown training strategy for teacher model (choices: desemble_epoch)')

                    # measure accuracy and record loss
                            
                    prec = accuracy(outputs[i].data, labels.data, topk=(1,))[0]
                    losses[i].update(loss.item(), images.size()[0])
                    ce_losses[i].update(ce_loss.item(), images.size()[0])
                    kd_losses[i].update(kd_loss, images.size()[0])
                    accs[i].update(prec.item(), images.size()[0])
                    self.optimizers[i].zero_grad()
                    loss.backward()
                    self.optimizers[i].step() 
                                                        
                # measure elapsed time
                toc = time.time()
                batch_time.update(toc-tic)

                self.batch_size = images.shape[0]
                pbar.update(self.batch_size)
   
        for i in range(self.model_num):
            logging.info("-model%s - Train accuracy: %s, training loss: %s" %(i+1, accs[i].avg, losses[i].avg))
            logging.info("-model%s - ce_loss: %s, kd_loss: %s" %(i+1, ce_losses[i].avg, kd_losses[i].avg))
            print("-model%s - ce_loss: %s, kd_loss: %s" %(i+1, ce_losses[i].avg, kd_losses[i].avg))
        return losses, accs

    def validate(self, epoch):
        """
        Evaluate the model on the validation set.
        """
        losses = []
        accs = []

        for i in range(self.model_num):
            self.models[i].eval()
            losses.append(AverageMeter())
            accs.append(AverageMeter())

        for i, (images, labels) in enumerate(self.valid_loader):
            if self.gpu is not None:
                images, labels = images.cuda(), labels.cuda()
            images, labels = Variable(images), Variable(labels)

            #forward pass
            outputs=[]
            for model in self.models:
                outputs.append(model(images))
            
            for i in range(self.model_num):
                if self.args.baseline:
                    loss = self.loss_ce(outputs[i], labels)        
                else:  
                    raise Exception('Unknown training strategy for teacher model (choices: desemble_epoch)')

                # measure accuracy and record loss
                prec = accuracy(outputs[i].data, labels.data, topk=(1,))[0]
                losses[i].update(loss.item(), images.size()[0])
                accs[i].update(prec.item(), images.size()[0])


        for i in range(self.model_num):
            logging.info("-model%s- Eval accuracy: %s, eval loss: %s" %(i+1, accs[i].avg, losses[i].avg))

        return losses, accs

    def test(self):
        """
        Test the model on the held-out test data.
        This function should only be called at the very
        end once the model has finished training.
        """
        self.load_checkpoint(best=self.best)

        losses = []
        top1 = []
        top5 = []
        for i in range(self.model_num):
            self.models[i].eval()
            losses.append(AverageMeter())
            top1.append(AverageMeter())
            top5.append(AverageMeter())

        for i, (images, labels) in enumerate(self.valid_loader):
            if self.gpu is not None:
                images, labels = images.cuda(), labels.cuda()
            images, labels = Variable(images), Variable(labels)

            #forward pass
            outputs=[]
            for model in self.models:
                outputs.append(model(images))
            for i in range(self.model_num):
                if self.args.desemble_epoch:
                    loss = self.loss_ce(outputs[i], labels)        
                else:  
                    raise Exception('Unknown training strategy for teacher model (choices: desemble_epoch)')

                # measure accuracy and record loss
                prec1, prec5 = accuracy(outputs[i].data, labels.data, topk=(1,5))
                losses[i].update(loss.item(), images.size()[0])
                top1[i].update(prec1.item(), images.size()[0])
                top5[i].update(prec5.item(), images.size()[0])

        for i in range(self.model_num):
            logging.info("-model%s- Eval prec1: %s, - Eval prec5: %s,eval loss: %s" %(i+1, top1[i].avg, top5[i].avg, losses[i].avg))
            print("-model%s- Test prec1: %s, - Test prec5: %s,Test loss: %s" %(i+1, top1[i].avg, top5[i].avg, losses[i].avg))

    def save_checkpoint(self, i, state, is_best, desemble=False):
        """
        Save a copy of the model so that it can be loaded at a future
        date. This function is used when the model is being evaluated
        on the test data.

        If this model has reached the best validation accuracy thus
        far, a seperate file with the suffix `best` is created.
        """
        # print("[*] Saving model to {}".format(self.ckpt_dir))

        if desemble:
            note = int(state['epoch'])//self.epo
            filename = self.arch + str(i+1) + '_teacher' + str(note) + '_ckpt.pth.tar'
            save_dir = self.save + 'desemble_epoch/teacher_ckpt/'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)            
            ckpt_path = os.path.join(save_dir, filename)
            torch.save(state, ckpt_path)
        else:
            filename = self.arch + str(i+1) + '_ckpt.pth.tar'
            ckpt_path = os.path.join(self.save, filename)
            torch.save(state, ckpt_path)

        if is_best:
            filename = self.arch + str(i+1) + '_model_best.pth.tar'
            shutil.copyfile(
                ckpt_path, os.path.join(self.save, filename)
            )

    def load_checkpoint(self, best=False):
        """
        Load the best copy of a model. This is useful for 2 cases:

        - Resuming training with the most recent model checkpoint.
        - Loading the best validation model to evaluate on the test data.

        Params
        ------
        - best: if set to True, loads the best model. Use this if you want
          to evaluate your model on the test data. Else, set to False in
          which case the most recent version of the checkpoint is used.
        """
        print("[*] Loading model from {}".format(self.save))

        for i in range(self.model_num):
            filename = self.arch + str(i+1) +'_ckpt.pth.tar'
            if best:
                filename = self.arch + str(i+1) + '_model_best.pth.tar'
            ckpt_path = os.path.join(self.save, filename)
            ckpt = torch.load(ckpt_path)

            # load variables from checkpoint
            self.start_epoch = ckpt['epoch']
            self.best_valid_accs[i] = ckpt['best_valid_acc']
            self.models[i].load_state_dict(ckpt['model_state'])
            self.optimizers[i].load_state_dict(ckpt['optim_state'])

            if best:
                print(
                    "[*] Loaded {} checkpoint @ epoch {} "
                    "with best valid acc of {:.3f}".format(
                        filename, ckpt['epoch'], ckpt['best_valid_acc'])
                )
            else:
                print(
                    "[*] Loaded {} checkpoint @ epoch {}".format(
                        filename, ckpt['epoch'])
                )
