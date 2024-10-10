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
import time


OmegaConf.register_new_resolver("eval", eval)

def replace_layers(model, old, new):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            ## compound module, go inside it
            replace_layers(module, old, new)
            
        if isinstance(module, old):
            ## simple module
            setattr(model, n, new)


@torch.no_grad()
def main(args):
    print(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    U_net = Unet(
        dim = 64,
        dim_mults = (1, 2),
        channels = 1
    )

    denoiser = GaussianDiffusion(
        U_net,
        image_size = 84,
        # channels = 1,
        timesteps = 1000,           # number of steps
        sampling_timesteps = 250,   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
        loss_type = 'l1'            # L1 or L2
    )

    with initialize(version_base="1.3", config_path="../config"):
        cfg = compose(config_name="trainer")
        OmegaConf.resolve(cfg)

    #test_env = make_atari_env(num_envs=, device=device, **cfg.env.train)
    cfg.env.test.id = args.env
    cfg.env.train.id = args.env
    test_env = make_atari_env(num_envs=cfg.collection.test.num_envs, device=device, **cfg.env.test)

    #Freeway Model
    if "Bank" in args.env:
        agent = Agent(instantiate(cfg.agent, num_actions=test_env.num_actions)).to(device).eval()
        path_ckpt = get_path_agent_ckpt("/mnt/c/Users/tulan/Desktop/Xiaolin/diffusion_attack/diamond/outputs/2024-09-09/11-03-37/checkpoints", epoch = -1)
        agent.load(path_ckpt)

    #Freeway Model
    if "Freeway" in args.env:
        agent = Agent(instantiate(cfg.agent, num_actions=test_env.num_actions)).to(device).eval()
        path_ckpt = get_path_agent_ckpt("/mnt/c/Users/tulan/Desktop/Xiaolin/diffusion_attack/diamond/outputs/2024-08-14/15-40-55/checkpoints", epoch = -1)
        agent.load(path_ckpt)

    if "Pong" in args.env:
    # Pong Model
        agent = Agent(instantiate(cfg.agent, num_actions=4)).to(device).eval()
        path_ckpt = get_path_agent_ckpt("/mnt/c/Users/tulan/Desktop/Xiaolin/diffusion_attack/diamond/outputs/2024-08-12/17-44-08/checkpoints", epoch = -1)
        agent.load(path_ckpt)

    # agent = Agent_DDPM(instantiate(cfg.agent, num_actions=test_env.num_actions)).to(device).eval()
    # path_ckpt = get_path_agent_ckpt("/mnt/c/Users/tulan/Desktop/Xiaolin/diffusion_attack/diamond/outputs/2024-08-08/18-14-20/checkpoints", epoch = -1)
    # agent.load(path_ckpt)

    n = 4
    dataset = Dataset(Path(f"dataset/{path_ckpt.stem}_{n}"))
    dataset.load_from_default_path()
    # if len(dataset) == 0:
    #     print(f"Collecting {n} steps in real environment for world model initialization.")
    #     collector = make_collector(test_env, agent.actor_critic, dataset, epsilon=0)
    #     collector.send(NumToCollect(steps=n))
    #     dataset.save_to_default_path()

    # World model environment
    bs = BatchSampler(dataset, 1, cfg.agent.denoiser.inner_model.num_steps_conditioning, None, False)
    dl = DataLoader(dataset, batch_sampler=bs, collate_fn=collate_segments_to_batch)

    wm_env_cfg = instantiate(cfg.world_model_env, num_batches_to_preload=1)
    wm_env = WorldModelEnv(agent.denoiser, agent.rew_end_model, dl, wm_env_cfg, return_denoising_trajectory=True)
    # wm_env = WorldModelEnv_DDPM(agent.denoiser, agent.rew_end_model, dl, wm_env_cfg, return_denoising_trajectory=True)

    if "Pong" in args.env and (args.model == "natural" or args.model == "diffusion_history"):
        policy = model_setup(cfg.env.train.id, test_env, False, None, True, True, 1).to(device)
        policy.features.load_state_dict(torch.load('src/pre_trained/Pong-natural.model'))
    if "Pong" in args.env and args.model == "sa-dqn-convex":
        policy = model_setup(cfg.env.train.id, test_env, True, None, True, True, 1).to(device)
        policy.features.load_state_dict(torch.load('src/pre_trained/Pong-convex.model'))
    if "Pong" in args.env and args.model == "sa-dqn-pgd":
        policy = model_setup(cfg.env.train.id, test_env, False, None, True, False, 1).to(device)
        policy.features.load_state_dict(torch.load('src/pre_trained/Pong-pgd.model'))
    if "Pong" in args.env and args.model == "wocar":
        policy = model_setup(cfg.env.train.id, test_env, False, None, True, False, 1).to(device)
        policy.features.load_state_dict(torch.load('src/pre_trained/Pong-wocar-pgd.pth'))
    if "Pong" in args.env and args.model == "car-dqn-pgd":
        policy = model_setup(cfg.env.train.id, test_env, False, None, True, True, 1).to(device)
        policy.features.load_state_dict(torch.load('src/pre_trained/Pong_car_pgd.pth'))

    if "Pong" in args.env and "dp-dqn" in args.model:
        policy = model_setup(cfg.env.train.id, test_env, False, None, True, True, 1).to(device)
        policy.features.load_state_dict(torch.load('src/pre_trained/Pong-DP-DQN-O.pth'))
        denoiser.load_state_dict(torch.load("/mnt/c/Users/tulan/Desktop/Xiaolin/stackberg/SA_DQN/results_Pong/model-150.pt")['model'])
        denoiser.to(device)

    if "Freeway" in args.env and (args.model == "natural" or args.model == "diffusion_history"):  
        policy = model_setup(cfg.env.train.id, test_env, False, None, True, True, 1).to(device)
        policy.features.load_state_dict(torch.load('src/pre_trained/Freeway-natural.model'))
    if "Freeway" in args.env and args.model == "sa-dqn-convex":
        policy = model_setup(cfg.env.train.id, test_env, True, None, True, True, 1).to(device)
        policy.features.load_state_dict(torch.load('src/pre_trained/Freeway-convex.model'))
    if "Freeway" in args.env and args.model == "sa-dqn-pgd":
        policy = model_setup(cfg.env.train.id, test_env, False, None, True, False, 1).to(device)
        policy.features.load_state_dict(torch.load('src/pre_trained/Freeway-pgd.model'))
    if "Freeway" in args.env and args.model == "wocar":
        policy = model_setup(cfg.env.train.id, test_env, False, None, True, False, 1).to(device)
        policy.features.load_state_dict(torch.load('src/pre_trained/Freeway-wocar-pgd.pth'))
    if "Freeway" in args.env and args.model == "car-dqn-pgd":
        policy = model_setup(cfg.env.train.id, test_env, False, None, True, True, 1).to(device)
        policy.features.load_state_dict(torch.load('src/pre_trained/Freeway_car_pgd.pth'))

    if "Freeway" in args.env and "dp-dqn" in args.model:
        policy = model_setup(cfg.env.train.id, test_env, False, None, True, True, 1).to(device)
        policy.features.load_state_dict(torch.load('src/pre_trained/Freeway-DP-DQN-O.pth'))
        denoiser.load_state_dict(torch.load("/mnt/c/Users/tulan/Desktop/Xiaolin/stackberg/SA_DQN/results_Freeway/model-150.pt")['model'])
        denoiser.to(device)

    if "Bank" in args.env and (args.model == "natural" or args.model == "diffusion_history"):  
        policy = model_setup(cfg.env.train.id, test_env, False, None, True, True, 1).to(device)
        policy.features.load_state_dict(torch.load('src/pre_trained/Bank-natural.pth'))
    if "Bank" in args.env and args.model == "sa-dqn-convex":
        policy = model_setup(cfg.env.train.id, test_env, True, None, True, True, 1).to(device)
        policy.features.load_state_dict(torch.load('src/pre_trained/Bank-convex.pth'))
    if "Bank" in args.env and args.model == "sa-dqn-pgd":
        policy = model_setup(cfg.env.train.id, test_env, False, None, True, False, 1).to(device)
        policy.features.load_state_dict(torch.load('src/pre_trained/Freeway-pgd.model'))
    if "Bank" in args.env and args.model == "wocar":
        policy = model_setup(cfg.env.train.id, test_env, False, None, True, False, 1).to(device)
        policy.features.load_state_dict(torch.load('src/pre_trained/Bank-wocar-pgd.pth'))
    if "Bank" in args.env and args.model == "car-dqn-pgd":
        policy = model_setup(cfg.env.train.id, test_env, False, None, True, True, 1).to(device)
        policy.features.load_state_dict(torch.load('src/pre_trained/Bank_car_pgd.pth'))

    if "Bank" in args.env and "dp-dqn" in args.model:
        policy = model_setup(cfg.env.train.id, test_env, False, None, True, True, 1).to(device)
        policy.features.load_state_dict(torch.load('src/pre_trained/Bank-DP-DQN-O.pth'))
        denoiser.load_state_dict(torch.load("/mnt/c/Users/tulan/Desktop/Xiaolin/stackberg/SA_DQN/results/model-150.pt")['model'])
        denoiser.to(device)


    if "Pong" in args.env:
        net = torch.load("/mnt/c/Users/tulan/Desktop/Xiaolin/diffusion_attack/diamond/ae/pong_autoencoder49").to(device)
    if "Freeway" in args.env:
        net = torch.load("/mnt/c/Users/tulan/Desktop/Xiaolin/diffusion_attack/diamond/ae/freeway_autoencoder49").to(device)
    if "Bank" in args.env:
        net = torch.load("/mnt/c/Users/tulan/Desktop/Xiaolin/diffusion_attack/diamond/ae/bank_autoencoder45").to(device)
    diff = torch.nn.MSELoss(reduce= 'sum')

    # for interval in [16,8,2]:
    total_rew_write = []
    total_mani_write = []
    total_dev_write = []
    total_invalid_write = []
    pgd_time = []
    attack_time = []
    pgd_1_error = []
    pgd_3_error = []
    pgd_15_error = []
    minbest_1_error = []
    minbest_3_error = []
    minbest_15_error = [] 
    minbest_time = []

    ours_error = []
    seed = 1000

    for i in range(10):
        end = False
        obs, _ = test_env.reset(seed = 1000+seed)

        count = 0
        obs_his = torch.zeros((4,1,84,84)).to(device)
        perturb_obs_his = torch.zeros((4,1,84,84)).to(device)
        print(obs.shape)
        for i in range(4):
            obs_his[i] = obs
            perturb_obs_his[i] = obs
        act_his = torch.zeros(4, dtype= torch.int).to(device)
        obs_his[-1] = obs
        perturb_obs_his[-1] = obs
        generated = None 
        total_rew = 0
        target_act = 0
        optimal_act = 0
        count_sec = 0
        count_div = 0
        tmp = obs.squeeze(0)
        tmp = torch.permute(tmp,(0,2,1))
        l_infinite_diff = 0
        invalid_set = set()
        while not end:
            act_his = act_his.roll(-1)
            #cv2.imwrite('/mnt/c/Users/tulan/Desktop/Xiaolin/diffusion_attack/diamond/testpic/'+'ori'+str(count-1)+'.png', norm_zero_pos(tmp).cpu().numpy().transpose(2,1,0)*255)
            # pgd_in = norm_zero_pos(tmp)
            # pgd_in.requires_grad = True
            # # pgd_in = Variable(pgd_in.data, requires_grad = True)
            # pgd_start = time.time()
            # att_state_tensor_1 = pgd(policy, norm_zero_pos(tmp), policy.act(norm_zero_pos(tmp), 0), env_id = "Pong", epsilon= 1/255).unsqueeze(0)
            # pgd_end = time.time()
            # pgd_time.append(pgd_end - pgd_start)
            # pgd_1_error.append(torch.norm(att_state_tensor_1 - torch.permute(net(att_state_tensor_1).squeeze(0),(0,2,1)), p = 2).cpu().numpy())
            # att_state_tensor_3 = pgd(policy, norm_zero_pos(tmp), policy.act(norm_zero_pos(tmp), 0), env_id = "Pong", epsilon= 3/255).unsqueeze(0)
            # pgd_3_error.append(torch.norm(att_state_tensor_3 - torch.permute(net(att_state_tensor_3).squeeze(0),(0,2,1)), p = 2).cpu().numpy())
            # att_state_tensor_15 = pgd(policy, norm_zero_pos(tmp), policy.act(norm_zero_pos(tmp), 0), env_id = "Pong", epsilon= 15/255).unsqueeze(0)
            # pgd_15_error.append(torch.norm(att_state_tensor_15 - torch.permute(net(att_state_tensor_15).squeeze(0),(0,2,1)), p = 2).cpu().numpy())
            # minbest_start = time.time()
            # att_state_tensor_1 = min_best(policy, norm_zero_pos(tmp), 1/255, pgd_steps=10, lr=1e-1, fgsm=False,norm=np.inf, rand_init=False, momentum=False, env_id = "Freeway").unsqueeze(0)
            # minbest_end = time.time()
            # minbest_time.append(minbest_end - minbest_start)
            # minbest_1_error.append(torch.norm(att_state_tensor_1 - torch.permute(net(att_state_tensor_1).squeeze(0),(0,2,1)), p = 2).cpu().numpy())
            # att_state_tensor_3 = min_best(policy, norm_zero_pos(tmp), 3/255, pgd_steps=10, lr=1e-1, fgsm=False,norm=np.inf, rand_init=False, momentum=False, env_id = "Freeway").unsqueeze(0)
            # minbest_3_error.append(torch.norm(att_state_tensor_3 - torch.permute(net(att_state_tensor_3).squeeze(0),(0,2,1)), p = 2).cpu().numpy())
            # att_state_tensor_15 = min_best(policy, norm_zero_pos(tmp), 15/255, pgd_steps=10, lr=1e-1, fgsm=False,norm=np.inf, rand_init=False, momentum=False, env_id = "Freeway").unsqueeze(0)
            # minbest_15_error.append(torch.norm(att_state_tensor_15 - torch.permute(net(att_state_tensor_15).squeeze(0),(0,2,1)), p = 2).cpu().numpy())
            #att_state = net(att_state_tensor_1).squeeze(0).cpu().numpy()
            #cv2.imwrite('/mnt/c/Users/tulan/Desktop/Xiaolin/diffusion_attack/diamond/testpic/'+'pgd'+str(count-1)+'.png', (att_state*255).transpose(1,2,0))
            if generated == None:
            #if True:
                act = policy.act(norm_zero_pos(tmp), 0)
            else:
                if args.model == "diffusion_history":
                    tmp2 = norm_zero_pos(torch.permute(victim_gen, (0,2,1)))
                    act = policy.act(tmp2, 0)
                elif args.model == "dp-dqn":
                    att_state_diff = denoiser.p_sample_loop_with_prior(norm_zero_pos(torch.permute(generated, (0,2,1))).cpu().numpy(), samples = 1, steps = 2).cpu().numpy()
                    att_state_diff = att_state_diff.squeeze(0)
                    cv2.imwrite('/mnt/c/Users/tulan/Desktop/Xiaolin/diffusion_attack/diamond/testpic/'+'test'+str(count-1)+'.png', (att_state_diff*255).transpose(2,1,0))
                    act = policy.act(torch.from_numpy(att_state_diff).to(device), 0)
                elif args.model == "dp-dqn-history":
                    att_state_diff = denoiser.p_sample_loop_with_prior(norm_zero_pos(torch.permute(victim_gen, (0,2,1))).cpu().numpy(), samples = 1, steps = 2).cpu().numpy()
                    att_state_diff = att_state_diff.squeeze(0)
                    cv2.imwrite('/mnt/c/Users/tulan/Desktop/Xiaolin/diffusion_attack/diamond/testpic/'+'test'+str(count-1)+'.png', (att_state_diff*255).transpose(2,1,0))
                    act = policy.act(torch.from_numpy(att_state_diff).to(device), 0)
                else:
                    tmp1 = norm_zero_pos(torch.permute(generated, (0,2,1)))
                    act = policy.act(tmp1, 0)
                # act_1 = policy.act(norm_zero_pos(tmp),0)
                # print(act, act_1)
            print("target ",target_act, "actual_act", act, "optimal_act", optimal_act, "current_reward", total_rew, "count", count-1, "diff", l_infinite_diff)
            if count > 15:    
                if (target_act.detach().cpu() == act.detach().cpu() and (target_act.detach().cpu()!=optimal_act.detach().cpu())):
                    count_sec  += 1 
                if (act.detach().cpu()!=optimal_act.detach().cpu()):
                    count_div += 1
            act_his[-1] = act
            #print(act)

            obs_input = obs_his.unsqueeze(0)
            act_input = act_his.unsqueeze(0)
            perturb_obs_his_input = perturb_obs_his.unsqueeze(0)
            #generated,_ = wm_env.sampler.sample_next_obs(obs_input, act_input)


            # print(generated)

            # if count == 30 :
            #     for i in range(4):
            #         obs_his[i] = obs
            obs, rew, end, _, _ = test_env.step(act)
            tmp = obs.squeeze(0)
            tmp = torch.permute(tmp,(0,2,1))
            optimal_act = policy.act(norm_zero_pos(tmp), 0)
            if args.attack == 'random':
                target_act = torch.randint(low=0, high=test_env.num_actions, size= (obs.size(0),), device=obs.device)
                while target_act == optimal_act:
                    target_act = torch.randint(low=0, high=test_env.num_actions, size= (obs.size(0),), device=obs.device)
            if args.attack == "min_q":
                q_value = policy.forward(norm_zero_pos(tmp).unsqueeze(0))
                target_act = q_value.min(1)[1]
            # generated,_ = wm_env.sampler.sample_next_obs_classifier_guide(obs_input, act_input, target_act, policy)
            valid_gen = False
            if "Pong" in args.env:
                if args.model!="natural" and args.model!="diffusion_history" and args.model != "dp-dqn-history":
                    try_strength = 2
                else:
                    try_strength = 3.5
            if "Freeway" in args.env:
                if args.model!="natural" and args.model!="diffusion_history" and args.model != "dp-dqn-history":
                    try_strength = 4.5
                else:
                    try_strength = 6
            if "Bank" in args.env:
                try_strength = 4
            # if "Freeway" in args.env:
            #     try_strength = 4.5
            # Pong strength 3.5
            # try_strength = 3.5
            # Freeway Strength 4
            # try_strength = 4.5
            while not valid_gen:    
                #generated,_ = wm_env.sampler.sample_next_obs(obs_input, act_input)
                ours_start = time.time()
                generated,_ = wm_env.sampler.sample_next_obs_classifier_guide_fade(obs_input, act_input, target_act, policy, try_strength, net)
                ours_end = time.time()
                attack_time.append(ours_end - ours_start)
                to_check = norm_zero_pos(generated)
                # to_check = norm_zero_pos(torch.permute(to_check, (1,2,0)))
                # cv2.imwrite('/mnt/c/Users/tulan/Desktop/Xiaolin/diffusion_attack/diamond/testpic/'+'check'+str(count)+'.png', (to_check[0]).cpu().numpy().transpose(1,2,0))
                loss = diff(to_check,net(to_check)).data.cpu()
                ours_error.append(torch.norm(to_check - net(to_check), p =2).cpu().numpy())
                # print("ours", loss)
                # print("normal", diff(norm_zero_pos(tmp.unsqueeze(0)), net(norm_zero_pos(tmp.unsqueeze(0)))).data.cpu())
                #if False:
                if loss >= 0.0003:
                #if loss >= 0.05:
                    print("invalid detected, retry with lower strength")
                    invalid_set.add(count)
                    try_strength = max(try_strength - 0.1, 0)
                    if try_strength <= 1:
                        break
                else:
                    valid_gen = True
            if args.model == 'diffusion_history' or args.model == "dp-dqn-history":
                victim_gen,_ = wm_env.sampler.sample_next_obs(perturb_obs_his_input, act_input)
                victim_gen = victim_gen.squeeze(0)
            # generated,_ = wm_env.sampler.sample_next_obs(obs_input, act_input)
            generated = generated.squeeze(0)

            #Pong 0.4, Freeway 0.25
            if args.record_change:
                l_infinite_diff = torch.max(torch.abs(norm_zero_pos(torch.permute(generated, (0,2,1))) - norm_zero_pos(tmp)))
                check_diff = None
                if "Pong" in args.env:
                    check_diff = 0.4
                if "Freeway" in args.env:
                    check_diff = 0.25
                if l_infinite_diff>check_diff:
                    cv2.imwrite('/mnt/c/Users/tulan/Desktop/Xiaolin/diffusion_attack/diamond/changepic/'+'att'+str(count)+'.png', norm_zero_pos(generated).cpu().numpy().transpose(1,2,0)*255)
                    cv2.imwrite('/mnt/c/Users/tulan/Desktop/Xiaolin/diffusion_attack/diamond/changepic/'+'ori'+str(count)+'.png', norm_zero_pos(tmp).cpu().numpy().transpose(2,1,0)*255)
            # victim_gen = victim_gen.squeeze(0)
            cv2.imwrite('/mnt/c/Users/tulan/Desktop/Xiaolin/diffusion_attack/diamond/testpic/'+'att'+str(count)+'.png', norm_zero_pos(generated).cpu().numpy().transpose(1,2,0)*255)
            # cv2.imwrite('/mnt/c/Users/tulan/Desktop/Xiaolin/diffusion_attack/diamond/testpic/'+'vic'+str(count)+'.png', norm_zero_pos(victim_gen).cpu().numpy().transpose(1,2,0)*255)
            # print(obs)
            total_rew+=rew
            obs_his = obs_his.roll(-1,dims=0)
            obs_his[-1] = obs
            perturb_obs_his = perturb_obs_his.roll(-1, dims= 0)
            # if count % interval !=0:
            if True:
                perturb_obs_his[-1] = generated
            else:
                print("add real")
                perturb_obs_his[-1] = obs
            count += 1
        print(total_rew)
        total_rew_write.append(total_rew.detach().cpu().numpy())
        print(count_sec/(count))
        total_mani_write.append(count_sec/(count))
        print(count_div/(count))
        total_dev_write.append(count_div/count)
        print(len(invalid_set)/(count))
        total_invalid_write.append((len(invalid_set)/count))
        filename = args.env+"_"+args.model+"_"+args.attack+".csv"
        filename = "output_results/"+filename
        file = open(filename, 'w')
        writer = csv.writer(file)
        writer.writerow(total_rew_write)
        writer.writerow(total_mani_write)
        writer.writerow(total_dev_write)
        writer.writerow(total_invalid_write)
        file.close()
            
        # filename = "timestudy.csv"
        # filename = "output_results/"+filename
        # file = open(filename, 'w')
        # writer = csv.writer(file)
        # writer.writerow(pgd_time)
        # writer.writerow(minbest_time)
        # writer.writerow(attack_time)
        # writer.writerow(pgd_1_error)
        # writer.writerow(pgd_3_error)
        # writer.writerow(pgd_15_error)
        # writer.writerow(minbest_1_error)
        # writer.writerow(minbest_3_error)
        # writer.writerow(minbest_15_error)
        # writer.writerow(ours_error)
        # file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--attack', type=str, default="random", help="attack types, random or minq")
    parser.add_argument('--env', type=str, default='PongNoFrameskip-v4', help='environment types')
    parser.add_argument('--model', type=str, default='natural', help="defense types, natural, sa-dqn-pgd, sa-dqn-convex, wocar, diffsuion_history")
    parser.add_argument('--record_change', type= bool, default= False, help="record semantic change images")
    args = parser.parse_args()
    main(args)
