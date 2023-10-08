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

_config = None
def get_config():
    global _config
    if _config == None:
        result = GPTConfig()
        result.block_size = get_int_config_value('block_size')
        result.vocab_size = get_int_config_value('vocab_size')
        result.n_layer = get_int_config_value('layers_number')
        result.n_head = get_int_config_value('heads_number')
        result.n_embd = get_int_config_value('embedding_size')
        result.dropout = get_float_config_value('dropout')
        result.bias = get_bool_config_value('bias')
        _config = result
    return _config

class OneHeadCausalSelfAttention(nn.Module):

    def __init__(self):
        super().__init__()
        config = get_config()
        self.config = config
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, 1, C).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, 1, C).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, 1, C).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class EmbeddingModel(nn.Module):
    def __init__(self):
        super().__init__()
        config = get_config()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, idx):
        device = idx.device
        b,t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.drop(tok_emb + pos_emb)

        return x

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

        #Self Attention
        self.embedding = EmbeddingModel()
        self.embedding.to(device)

        #Self Attention
        self.sattn = OneHeadCausalSelfAttention()
        self.sattn.to(device)

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
            self.totalSortKey = "cpu_time_total" 
            self.selfSortKey = "cpu_time_self"
        else:
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

class ForwardSAProfileCase(ProfileCase):

    def run(self, iterations = 1):
        train_batch = self.loader.batch()
        x = self.embedding(train_batch[0])

        with profile(activities=self.profiler_activities, record_shapes=True) as prof:
            for _ in range(iterations):
                self.sattn(x)
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
    elif (name == 'forwardsa'):
        profileCase = ForwardSAProfileCase()
    else:
        raise Exception('Unknown profile case '+profileCase)
    prof = profileCase.run(iterations = iterations)
    profile_report(prof)












