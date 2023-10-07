from environment import log, get_float_config_value, get_bool_config_value,get_int_config_value, device, workDir

from model import GPT
from data import DataLoader
import inspect
import torch
from torch.nn import functional as F

from torch.profiler import profile, record_function, ProfilerActivity

class ProfileCase:

    def __init__(self):

        #Params
        self.weight_decay = get_float_config_value('weight_decay')
        self.learning_rate = get_float_config_value('learning_rate')
        self.betas = (get_float_config_value('beta1'), get_float_config_value('beta2'))
        
        
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
            self.profiler_activities = [ProfilerActivity.CPU]
        else:
            self.profiler_activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    
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



def profile_report(prof):
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

def profile_run(name, iterations):
    profileCase = None
    if (name == 'forward'):
        profileCase = ForwardProfileCase()
    elif (name == 'calcloss'):
        profileCase = CalcLossProfileCase()
    elif (name == 'backward'):
        profileCase = BackwardProfileCase()
    else:
        raise Exception('Unknown profile case '+profileCase)
    prof = profileCase.run(iterations = iterations)
    profile_report(prof)












