from environment import log, get_float_config_value, get_bool_config_value,get_int_config_value, device, workDir

from model import GPT, print_config
from data import DataLoader
import inspect
import math
import torch
from time import time
from torch.nn import functional as F
from timers import create_timer,delete_timer,has_timer, start, stop, get_time_sum, get_time_sum_fmt, get_time_avg_fmt, get_count, get_time_avg
from os.path import isfile
from pickle import dump, load

class Trainer:
    def __init__(self, minutes_to_train):
        #Params
        self.minutes_to_train = minutes_to_train
        self.weight_decay = get_float_config_value('weight_decay')
        self.learning_rate = get_float_config_value('learning_rate')
        self.min_learning_rate = get_float_config_value('min_learning_rate')
        self.betas = (get_float_config_value('beta1'), get_float_config_value('beta2'))
        self.decay_lr = get_bool_config_value('decay_lr')
        self.warmup_iters = get_int_config_value('warmup_iters')
        self.lr_decay_iters = get_int_config_value('lr_decay_iters')
        self.eval_interval = get_int_config_value('eval_interval')
        self.grad_clip = get_float_config_value('grad_clip')
        self.eval_iters = get_int_config_value('eval_iters')
        self.log_interval = get_int_config_value('log_interval')
        self.max_epochs_without_improvement = get_int_config_value('max_epochs_without_improvement')

        #State
        self.state_file = workDir+"state_dict.bin"
        self.state = {'lr_counter':0, 'min_val_loss': float("inf")}
        self.resuming = False
        
        #Model
        self.model_file = workDir+"model_dict.bin"
        self.model = GPT()
        self.model.to(device)

        #Resuming from last checkpoint, or partly from pretrained model
        self.resume()

        #Data Loader
        self.loader = DataLoader()

        #Optimizer 
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.model.named_parameters()}
        # filter out those that do not require grad -> eigentlich unnötig, da sie alle gradient erfordern
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': self.weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device == 'cuda'
        if (use_fused):
            log.info('Using fused version of the AdamW optimizer')
        extra_args = dict(fused=True) if use_fused else dict()
        self.optimizer = torch.optim.AdamW(optim_groups, lr=self.learning_rate, betas=self.betas, **extra_args)
    
    # learning rate decay scheduler (cosine with warmup)
    def get_lr(self, it):
        # 1) linear warmup for warmup_iters steps
        if it < self.warmup_iters:
            return self.learning_rate * it / self.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > self.lr_decay_iters:
            return self.min_learning_rate
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - self.warmup_iters) / (self.lr_decay_iters - self.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return self.min_learning_rate + coeff * (self.learning_rate - self.min_learning_rate)

    def calculate_loss(self, logits, targets):
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
    
    def write_checkpoint(self):
        log.info("Saving model to "+self.model_file)
        torch.save(self.model.state_dict(), self.model_file)
        log.info("Saving state to "+self.state_file)
        f = open (self.state_file,"wb")
        dump(self.state, f)
        f.close()
    
    def resume(self):
         if (isfile(self.model_file)):
            log.info("Loading model from "+self.model_file)
            self.model.load_state_dict(torch.load(self.model_file))
         if (isfile(self.state_file)):
             log.info("Loading state from "+self.state_file)
             f = open (self.state_file,"rb")
             self.state = load(f)
             f.close()
             self.resuming = True

    def run(self):
        #Training loop
        epochCounter = 0
        start_time = time()
        calculationTime = 0.0
        iter_counter = 0
        min_val_loss_counter = 0
        

        #Timers
        create_timer('loop')
        create_timer('train')
        create_timer('validate')
        

        #Start message
        log.info("################################################################################")
        log.info("Starting training for "+str(self.minutes_to_train)+" minutes")
        log.info("The model has "+str(self.model.get_num_params())+" parameters")
        print_config()
        if (self.resuming):
            log.info("Resuming from saved state with min_val_loss = "+str(self.state['min_val_loss']))
        log.info("################################################################################")

        while True:
            start('loop')
            start('train')
            if (not has_timer('last_train')):
                create_timer('last_train')
            start('last_train')
            #Get next batch
            train_batch = self.loader.batch()
            #Set learning rate
            # determine and set the learning rate for this iteration
            lr = self.get_lr(self.state['lr_counter']+1) if self.decay_lr else self.learning_rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            # zero the parameter gradients
            self.optimizer.zero_grad(set_to_none=True)
            # forward + backward + optimize
            calc_start_time = time()
            logits = self.model(train_batch[0])
            loss = self.calculate_loss(logits, train_batch[1])
            loss.backward()
            self.optimizer.step()
            calc_end_time = time()
            calculationTime +=(calc_end_time-calc_start_time)
            iter_counter+=1
            self.state['lr_counter'] = self.state['lr_counter']+1
            stop('train')
            stop('last_train')
            if (iter_counter == 1 or iter_counter%self.log_interval == 0):
                log.info("Iteration "+str(iter_counter)+" last learning rate = "+str(lr)+", last loss = "+str(loss.item()))

            if iter_counter%self.eval_interval == 0:
                epochCounter+=1
                #Validation
                validation_loss = 0.0
                train_loss = 0.0
                val_loss = 0.0
                for it in range(self.eval_iters):
                    start('validate')
                    self.model.eval()
                    train_batch = self.loader.batch()
                    logits = self.model(train_batch[0])
                    loss = self.calculate_loss(logits, train_batch[1])
                    train_loss+=loss.item()
                    val_batch = self.loader.batch(train=False)
                    logits = self.model(val_batch[0])
                    loss = self.calculate_loss(logits, val_batch[1])
                    val_loss+=loss.item()
                    self.model.train()
                    stop('validate')
                stop('loop')
                log.info("######################################Epoch Report#######################################################")
                current_val_loss = val_loss/self.eval_iters
                log.info('Epoch '+str(epochCounter)+" done, mean train loss = "+str(train_loss/self.eval_iters)+", mean validation loss = "+str(current_val_loss))
                log.info('Has been running since '+get_time_sum_fmt('loop'))
                log.info('Training time '+get_time_sum_fmt('train')+", "+str(get_time_avg('train'))+" sec per iteration ("+str(get_time_avg('last_train'))+" sec in the last epoch)")
                log.info('Validation time '+get_time_sum_fmt('validate')+", "+str(get_time_avg('validate'))+" sec per iteration")
                if (current_val_loss < self.state['min_val_loss']):
                    self.state['min_val_loss'] = current_val_loss
                    min_val_loss_counter = 0
                    self.write_checkpoint()
                else:
                    min_val_loss_counter+=1
                    log.info("No improvement to best validation loss "+str(self.state['min_val_loss'])+" since "+str(min_val_loss_counter)+" epochs")
                log.info("###############################################################################################################")
                running_loss = 0.0
                delete_timer('last_train')
                if (get_time_sum('loop')>self.minutes_to_train*60.0):
                    log.info("Time is up, stopping training")
                    break
                if (min_val_loss_counter >= self.max_epochs_without_improvement):
                    log.info("Stopping because no improvement since "+str(min_val_loss_counter)+" epochs")
                    break
            else:
                stop('loop')
            
        log.info("######################################Validation Report#######################################################")
        log.info("Training done in "+get_time_sum_fmt('loop')+", got best validation loss of "+str(self.state['min_val_loss']))
        log.info("###############################################################################################################")





        


        
    
   

