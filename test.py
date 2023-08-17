import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import random
import torch
import numpy as np
from model import MAMHGCN

CUDA = 0
SEED = 0
HIDDEN_DIM = 16
RANDOMSTATE = -7

def seed_torch(seed=SEED):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
 

def eval_crosssubject(model, test_data):
    device = torch.device("cuda:{}".format(CUDA) if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        test_data        = test_data.to(device).to(torch.float32).clone().detach()
        sleep_output, bect_output, _ = model(test_data)
    return sleep_output, bect_output


if __name__ == '__main__':
    seed_torch()
    device = torch.device("cuda:{}".format(CUDA) if torch.cuda.is_available() else "cpu")
    model = MAMHGCN(time = 13, num_nodes=21, num_bands=5, hidden_dim=HIDDEN_DIM, output_dim=2, 
                        use_attention=True, use_domain_adaptation=True, learn_adjacency=True, device=device)
    model.load_state_dict(torch.load('pretrained_model.pt'))
    model.to(device)
    data = torch.rand(10,13,21,5)*RANDOMSTATE
    sleep_output, bect_output = eval_crosssubject(model, data)
    sleep_pred, bect_pred = sleep_output.argmax(1), bect_output.argmax(1)
    for i in range(len(data)):
        sleep, bect = 'NREM and REM'if sleep_pred[i] else 'Wakefulness', 'BECT'if bect_pred[i] else 'Non-BECT'
        print ("SAMPLE{}:{};{}".format(i+1,sleep,bect))
