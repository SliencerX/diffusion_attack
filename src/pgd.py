import argparse
from pathlib import Path
from typing import Tuple

# from huggingface_hub import hf_hub_download
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader

from agent import Agent
from agent_ddpm import Agent_DDPM
from coroutines.collector import make_collector, NumToCollect
from data import BatchSampler, collate_segments_to_batch, Dataset
from envs import make_atari_env, WorldModelEnv, make_atari_env_test, WorldModelEnv_DDPM
#from game import ActionNames, DatasetEnv, Game, get_keymap_and_action_names, Keymap, NamedEnv, PlayEnv
from utils import get_path_agent_ckpt, prompt_atari_game
from model import *
import cv2
from utils import *
import sys
sys.path.append("..")
from autoencoder_models import *
import argparse
import csv
from attacks import *
from denoising_diffusion_pytorch import Unet, GaussianDiffusion

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

with initialize(version_base="1.3", config_path="../config"):
    cfg = compose(config_name="trainer")
    #OmegaConf.resolve(cfg)

cfg.env.test.id = "FreewayNoFrameskip-v4"

test_env = make_atari_env(num_envs=cfg.collection.test.num_envs, device=device, **cfg.env.test)

file_path = "/mnt/c/Users/tulan/Desktop/Xiaolin/diffusion_attack/diamond/toshow/motivations/ori1401.png"

policy = model_setup("FreewayNoFrameskip-v4", test_env, False, None, True, True, 1).to(device)
policy.features.load_state_dict(torch.load('src/pre_trained/Freeway-natural.model'))
pgd_in = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
pgd_in = np.expand_dims(pgd_in, axis= 0)
print(pgd_in.shape)
print(pgd_in)

pgd_in = torch.from_numpy(pgd_in/255).to(device).float()
#pgd_in = torch.permute(pgd_in, (2,1,0))
pgd_in.requires_grad = True
# pgd_in = Variable(pgd_in.data, requires_grad = True)
att_state_tensor = pgd(policy, (pgd_in), policy.act((pgd_in), 0), env_id = "Pong")
att_state = att_state_tensor.cpu().numpy()
cv2.imwrite('/mnt/c/Users/tulan/Desktop/Xiaolin/diffusion_attack/diamond/toshow/motivations/pgd1401.png', (att_state*255).transpose(1,2,0))