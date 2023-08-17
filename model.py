import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F

class GraphConvolutionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        super(GraphConvolutionLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
        if self.use_bias:
            nn.init.zeros_(self.bias)
    
    def norm_adjacency(self, A):
        d = A.sum(1)
        D = torch.diag(torch.pow(d, -0.5))
        return D.mm(A).mm(D)
    
    def forward(self, X, A):
        support = torch.einsum('ntvb,bh->ntvh', X, self.weight)
        norm_A = self.norm_adjacency(A)
        output = torch.einsum('uv,ntvh->ntuh', norm_A, support)
        if self.use_bias:
            output += self.bias
        return output

class ChannelAttention(nn.Module):
    def __init__(self, num_nodes, num_dim): 
        super(ChannelAttention, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(num_nodes, num_nodes))
        self.bias = nn.Parameter(torch.Tensor(num_nodes, num_dim))
        self.softmax = nn.Softmax(1)
        nn.init.normal_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        alpha = torch.einsum('uv,ntvh->ntuh', self.weight, x) + self.bias
        alpha = self.softmax(alpha)
        new_fea = alpha * x
        hidden_feature = new_fea.sum(2).sum(1)
        attention_weight = alpha.mean(3).mean(1).mean(0)
        return new_fea, hidden_feature, attention_weight

class TemporalAttention(nn.Module):
    def __init__(self,channel,reduction=4):   
        super().__init__()
        self.maxpool=nn.AdaptiveMaxPool2d(1)
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.se=nn.Sequential( 
            nn.Linear(channel,channel//reduction,bias=False), 
            nn.LeakyReLU(inplace = True),
            nn.Linear(channel//reduction,channel,bias=False))
        self.softmax = nn.Softmax(1)
    
    def forward(self, x) :
        n, t, _, _ = x.size()
        max_result, avg_result = self.maxpool(x).view(n,t), self.avgpool(x).view(n,t)
        max_out , avg_out  = self.se(max_result), self.se(avg_result)
        attention_weight   = self.softmax(max_out+avg_out)
        new_fea = x * attention_weight.view(n, t, 1 ,1).expand_as(x)
        return new_fea, attention_weight 

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class MAMHGCN(nn.Module):
    def __init__(self, num_bands=5, output_dim=2, hidden_dim=16, device = None, num_nodes=21, time = 13,  use_attention=True, use_domain_adaptation=True, learn_adjacency=True):
        super(MAMHGCN, self).__init__()
        self.device, self.num_nodes = device, num_nodes
        self.use_attention, self.use_domain_adaptation = use_attention, use_domain_adaptation

        self.bn = nn.ModuleList([nn.BatchNorm1d(time *  num_bands * num_nodes), 
                                 nn.BatchNorm1d(time * hidden_dim * num_nodes),
                                 nn.BatchNorm1d(time * hidden_dim * num_nodes), 
                                 nn.BatchNorm1d(time * hidden_dim * num_nodes)])
        self.gcn_layer = nn.ModuleList([GraphConvolutionLayer( num_bands, hidden_dim), 
                                        GraphConvolutionLayer(hidden_dim, hidden_dim),
                                        GraphConvolutionLayer(hidden_dim, hidden_dim)])
        
        self.fc_sleep = nn.Linear(hidden_dim, output_dim)
        self.fc_bect  = nn.Linear(hidden_dim, output_dim)

        if self.use_attention:
            self.channel_attention  = ChannelAttention(num_nodes, hidden_dim)
            self.temporal_attention = TemporalAttention(time, reduction=4)        
            self.freqency_attention = TemporalAttention(hidden_dim, reduction=2)  
        if self.use_domain_adaptation:
            self.domain_classifier = nn.Linear(hidden_dim, 2)

        self.xs, self.ys = torch.tril_indices(self.num_nodes, self.num_nodes, offset=-1)
        adjacency = torch.Tensor(self.num_nodes, self.num_nodes)
        nn.init.uniform_(adjacency)                           
        self.A = nn.Parameter(adjacency[self.xs, self.ys], requires_grad=learn_adjacency)
    
    def _batch_norm(self, X, index):
        N, T, V, B = X.shape
        X = X.contiguous().view(N, -1)
        X = self.bn[index](X)
        X = X.view(N, T, V, B)
        return X

    def forward(self, X, alpha=None):
        adacency = torch.zeros((self.num_nodes, self.num_nodes)).to(self.device)
        adacency[self.xs, self.ys] = self.A
        adacency = adacency + adacency.T + torch.eye(self.num_nodes).to(self.device)

        #share_net
        X = F.leaky_relu(self.gcn_layer[0](X, adacency.to(self.device)))
        X, hidden_feature, self.attention_weight = self.channel_attention(X)

        #sleep_head
        X_sleep = F.leaky_relu(self.gcn_layer[1](X, adacency.to(self.device)))
        X_sleep, attention_weight_sleep = self.temporal_attention(X_sleep)
        hidden_sleep_featute = X_sleep.sum(2).sum(1)
        sleep_output = self.fc_sleep(hidden_sleep_featute)
        
        #bect_head
        X_bect = F.leaky_relu(self.gcn_layer[2](X, adacency.to(self.device)))
        X_bect = X_bect.transpose(1,3)
        X_bect, attention_weight_bect = self.freqency_attention(X_bect)
        X_bect = X_bect.transpose(1,3)
        hidden_bect_featute = X_bect.sum(2).sum(1)
        bect_output = self.fc_bect(hidden_bect_featute)

        domain_output = None
        if self.use_domain_adaptation:
            reverse_X = ReverseLayerF.apply(hidden_sleep_featute+ hidden_bect_featute, alpha)
            domain_output = self.domain_classifier(reverse_X)       
        return sleep_output, bect_output, domain_output