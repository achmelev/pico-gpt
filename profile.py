from environment import log, get_float_config_value, get_bool_config_value,get_int_config_value, device, workDir

from model import GPT
from data import DataLoader
import inspect
import torch
import torch.nn as nn
from torch.nn import functional as F

from torch.profiler import profile, record_function, ProfilerActivity
from model import GPTConfig 

groupByInputShape = get_bool_config_value('profile_group_by_input_shape')
if (device == 'cpu'):
    totalSortKey = "cpu_time_total" 
    selfSortKey = "self_cpu_time_total"
else:
    totalSortKey = "cuda_time_total" 
    selfSortKey = "self_cuda_time_total"


class ProfileCase:

    def __init__(self):

        log.info("Device = "+device)

        #Params
        self.weight_decay = get_float_config_value('weight_decay')
        self.learning_rate = get_float_config_value('learning_rate')
        self.betas = (get_float_config_value('beta1'), get_float_config_value('beta2'))

        self.batch_size = get_int_config_value('batch_size')
        self.block_size = get_int_config_value('block_size')
        self.embedding_size = get_int_config_value('embedding_size')
        
        #Model
        self.model_file = workDir+"model_dict.bin"
        self.model = GPT()
        self.model.to(device)

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

        if (device == 'cpu'):
            log.info('Profiling CPU')
            self.profiler_activities = [ProfilerActivity.CPU]
            self.totalSortKey = "cpu_time_total" 
            self.selfSortKey = "cpu_time_self"
        else:
            log.info('Profiling CPU und CUDA')
            self.profiler_activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
            self.totalSortKey = "cuda_time_total" 
            self.selfSortKey = "cuda_time_self"
        
   
    
    def calculate_loss(self, logits, targets):
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

class ForwardProfileCase(ProfileCase):

    def run(self, iterations = 1):
        train_batch = self.loader.batch()

        with profile(activities=self.profiler_activities, record_shapes=True) as prof:
            for _ in range(iterations):
                self.model(train_batch[0])
        return prof

class CalcLossProfileCase(ProfileCase):

    def run(self, iterations = 1):
        train_batch = self.loader.batch()
        logits = self.model(train_batch[0])
        

        with profile(activities=self.profiler_activities, record_shapes=True) as prof:
            
            for _ in range(iterations):
                loss = self.calculate_loss(logits, train_batch[1])
        return prof

class BackwardProfileCase(ProfileCase):

    def run(self, iterations = 1):
        train_batch = self.loader.batch()
        logits = self.model(train_batch[0])
        loss = self.calculate_loss(logits, train_batch[1])

        with profile(activities=self.profiler_activities, record_shapes=True) as prof:
            for _ in range(iterations):
                loss.backward(retain_graph=True)
        return prof

class BackwardProfileCase(ProfileCase):

    def run(self, iterations = 1):
        train_batch = self.loader.batch()
        logits = self.model(train_batch[0])
        loss = self.calculate_loss(logits, train_batch[1])

        with profile(activities=self.profiler_activities, record_shapes=True) as prof:
            for _ in range(iterations):
                loss.backward(retain_graph=True)
        return prof

class ForwardDPAttentionProfileCase(ProfileCase):

    def run(self, iterations = 1):
        q = torch.randn((self.batch_size, 1, self.block_size, self.embedding_size), requires_grad = True)
        q.to(device)
        k = torch.randn((self.batch_size, 1, self.block_size, self.embedding_size), requires_grad = True)
        k.to(device)
        v = torch.randn((self.batch_size, 1, self.block_size, self.embedding_size), requires_grad = True)
        v.to(device)


        with profile(activities=self.profiler_activities, record_shapes=True) as prof:
            for _ in range(iterations):
                F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0, is_causal=True)
        return prof

class BackwardDPAttentionProfileCase(ProfileCase):

    def run(self, iterations = 1):
        q = torch.full((self.batch_size, 1, self.block_size, self.embedding_size),1.0, requires_grad = True)
        q.to(device)
        k = torch.full((self.batch_size, 1, self.block_size, self.embedding_size),2.0, requires_grad = True)
        k.to(device)
        v = torch.full((self.batch_size, 1, self.block_size, self.embedding_size),3.0, requires_grad = True)
        v.to(device)

        grad = torch.randn((self.batch_size, 1, self.block_size, self.embedding_size), requires_grad = True)
        grad.to(device)

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0, is_causal=True)
        y.retain_grad()

        with profile(activities=self.profiler_activities, record_shapes=True) as prof:
            for _ in range(iterations):
                y.backward(gradient = grad, retain_graph = True)
        return prof
    
    



def profile_report(prof, ):
    result = prof.key_averages()
    print("############################Sorted by Self time#####################################")
    print(prof.key_averages(group_by_input_shape = groupByInputShape).table(sort_by=selfSortKey))
    print("#####################################################################################")
    print("############################Sorted by Total time#####################################")
    print(prof.key_averages(group_by_input_shape = groupByInputShape).table(sort_by=totalSortKey))
    print("#####################################################################################")

def profile_run(name, iterations):
    profileCase = None
    if (name == 'forward'):
        profileCase = ForwardProfileCase()
    elif (name == 'calcloss'):
        profileCase = CalcLossProfileCase()
    elif (name == 'backward'):
        profileCase = BackwardProfileCase()
    elif (name == 'forwarddpa'):
        profileCase = ForwardDPAttentionProfileCase()
    elif (name == 'backwarddpa'):
        profileCase = BackwardDPAttentionProfileCase()
    else:
        raise Exception('Unknown profile case '+profileCase)
    prof = profileCase.run(iterations = iterations)
    profile_report(prof)












