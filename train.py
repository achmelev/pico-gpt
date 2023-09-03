from environment import log, get_float_config_value, get_bool_config_value,get_int_config_value, device

from model import GPT
from data import DataLoader
import inspect
import math
import torch
from time import time
from torch.nn import functional as F

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
        
        #Model
        self.model = GPT()

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

    def run(self):
        #Training loop
        epochCounter = 0
        start_time = time()
        calculationTime = 0.0
        iter_counter = 0

        log.info("Starting training")

        while True:
            #Get next batch
            train_batch = self.loader.batch()
            #Set learning rate
            # determine and set the learning rate for this iteration
            lr = self.get_lr(iter_counter+1) if self.decay_lr else self.learning_rate
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
            if (iter_counter == 1 or iter_counter%100 == 0):
                log.debug("Iteration "+str(iter_counter)+" last learning rate = "+str(lr)+", last loss = "+str(loss.item()))

            if iter_counter%self.eval_interval == 0:
                epochCounter+=1
                #Validation
                validation_loss = 0.0
                train_loss = 0.0
                val_loss = 0.0
                for it in range(self.eval_iters):
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
                log.info('Epoch '+str(epochCounter)+" done, mean train loss = "+str(train_loss/self.eval_iters)+", mean validation loss = "+str(val_loss/self.eval_iters))
                running_loss = 0.0





        


        
    
   

