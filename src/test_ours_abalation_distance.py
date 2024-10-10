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

    if "Pong" in args.env and args.model == "dp-dqn":
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

    if "Freeway" in args.env and args.model == "dp-dqn":
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

    if "Bank" in args.env and args.model == "dp-dqn":
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


    seed = 1000

    for try_strength in [1,2,3,4,5,6]:
        total_rew_write = []
        total_mani_write = []
        total_dev_write = []
        total_invalid_write = []
        total_mse_loss = []
        for i in range(5):
            perturbed_dis = []
            pgd_dis = []
            mse_dis = []
            end = False
            obs, _ = test_env.reset(seed = 1000+seed)

            count = 0
            obs_his = torch.zeros((4,1,84,84)).to(device)
            perturb_obs_his = torch.zeros((4,1,84,84)).to(device)

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
            to_check = None
            while not end:
                act_his = act_his.roll(-1)
                # cv2.imwrite('/mnt/c/Users/tulan/Desktop/Xiaolin/diffusion_attack/diamond/testpic/'+'ori'+str(count-1)+'.png', norm_zero_pos(tmp).cpu().numpy().transpose(2,1,0)*255)
                # pgd_in = norm_zero_pos(tmp)
                # pgd_in.requires_grad = True
                # pgd_in = Variable(pgd_in.data, requires_grad = True)
                # att_state_tensor = pgd(policy, norm_zero_pos(tmp), policy.act(norm_zero_pos(tmp), 0), env_id = "Pong")
                # pgd_diff = torch.norm(norm_zero_pos(tmp)-att_state_tensor, p = 2).cpu().numpy()
                # pgd_dis.append(pgd_diff)
                # print("pgd", pgd_dis[-1])
                # att_state = att_state_tensor.cpu().numpy()
                # cv2.imwrite('/mnt/c/Users/tulan/Desktop/Xiaolin/diffusion_attack/diamond/testpic/'+'pgd'+str(count-1)+'.png', (att_state*255).transpose(2,1,0))
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

                obs, rew, end, _, _ = test_env.step(act)
                tmp = obs.squeeze(0)
                tmp = torch.permute(tmp,(0,2,1))
                optimal_act = policy.act(norm_zero_pos(tmp), 0)
                if args.attack == 'random':
                    target_act = torch.randint(low=0, high=test_env.num_actions, size= (obs.size(0),), device=obs.device)
                    while target_act == optimal_act:
                        target_act = torch.randint(low=0, high=test_env.num_actions, size= (obs.size(0),), device=obs.device)
                if args.attack == "min_q":
                    q_value = policy.forward(norm_zero_pos(tmp))
                    target_act = q_value.min(1)[1]
                # generated,_ = wm_env.sampler.sample_next_obs_classifier_guide(obs_input, act_input, target_act, policy)
                valid_gen = False
                # if "Pong" in args.env:
                #     if args.model!="natural" and args.model!="diffusion_history":
                #         try_strength = 2
                #     else:
                #         try_strength = 4
                # if "Freeway" in args.env:
                #     if args.model!="natural" and args.model!="diffusion_history":
                #         try_strength = 4.5
                #     else:
                #         try_strength = 6
                # if "Bank" in args.env:
                #     try_strength = 4
                # if "Freeway" in args.env:
                #     try_strength = 4.5
                # Pong strength 3.5
                # try_strength = 3.5
                # Freeway Strength 4
                # try_strength = 4.5
                while not valid_gen:    
                    #generated,_ = wm_env.sampler.sample_next_obs(obs_input, act_input)
                    generated,_ = wm_env.sampler.sample_next_obs_classifier_guide_fade(obs_input, act_input, target_act, policy, try_strength, net)
                    to_check = norm_zero_pos(generated)
                    # to_check = norm_zero_pos(torch.permute(to_check, (1,2,0)))
                    # cv2.imwrite('/mnt/c/Users/tulan/Desktop/Xiaolin/diffusion_attack/diamond/testpic/'+'check'+str(count)+'.png', (to_check[0]).cpu().numpy().transpose(1,2,0))
                    to_check_1 = net(to_check)
                    loss = diff(to_check,to_check_1).data.cpu()
                    #mse_dis.append(loss.data.cpu().numpy())
                    mse_dis.append(torch.norm(to_check-to_check_1, p=2).cpu().numpy())
                    # print(mse_dis[-1])
                    # print("ours", loss)
                    # print("normal", diff(norm_zero_pos(tmp.unsqueeze(0)), net(norm_zero_pos(tmp.unsqueeze(0)))).data.cpu())
                    #if False:
                    if loss >= 0.0003:
                    #if loss >= 0.05:
                        print("invalid detected, retry with lower strength")
                        invalid_set.add(count)
                        #try_strength = max(try_strength - 0.1, 0)
                        if try_strength <= 1:
                            break
                        valid_gen = True
                    else:
                        valid_gen = True
                if args.model == 'diffusion_history':
                    victim_gen,_ = wm_env.sampler.sample_next_obs(perturb_obs_his_input, act_input)
                    victim_gen = victim_gen.squeeze(0)
                # generated,_ = wm_env.sampler.sample_next_obs(obs_input, act_input)
                generated = generated.squeeze(0)
                # if to_check != None:
                #     gen_loss = torch.norm(norm_zero_pos(torch.permute(generated, (0,2,1))) - norm_zero_pos(tmp), p=2)
                #     perturbed_dis.append(gen_loss.cpu().numpy())
                #     print("perturbed", perturbed_dis[-1])
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
                # cv2.imwrite('/mnt/c/Users/tulan/Desktop/Xiaolin/diffusion_attack/diamond/testpic/'+'att'+str(count)+'.png', norm_zero_pos(generated).cpu().numpy().transpose(1,2,0)*255)
                # cv2.imwrite('/mnt/c/Users/tulan/Desktop/Xiaolin/diffusion_attack/diamond/testpic/'+'vic'+str(count)+'.png', norm_zero_pos(victim_gen).cpu().numpy().transpose(1,2,0)*255)
                # print(obs)
                total_rew+=rew
                obs_his = obs_his.roll(-1,dims=0)
                obs_his[-1] = obs
                perturb_obs_his = perturb_obs_his.roll(-1, dims= 0)
                perturb_obs_his[-1] = generated
                count += 1
                if count >= 500:
                    break
            print(total_rew)
            total_rew_write.append(total_rew.detach().cpu().numpy())
            print(count_sec/(count))
            total_mani_write.append(count_sec/(count))
            print(count_div/(count))
            total_dev_write.append(count_div/count)
            print(len(invalid_set)/(count))
            total_invalid_write.append((len(invalid_set)/count))
            total_mse_loss.append(np.mean(mse_dis))
            filename = args.env+"_"+args.model+"_"+str(try_strength)+args.attack+"abaltion_only_l2.csv"
            filename = "output_results/"+filename
            file = open(filename, 'w')
            writer = csv.writer(file)
            writer.writerow(total_rew_write)
            writer.writerow(total_mani_write)
            writer.writerow(total_dev_write)
            writer.writerow(total_invalid_write)
            # tmp = [np.mean(mse_dis), np.std(mse_dis)]
            # writer.writerow(tmp)
            writer.writerow(total_mse_loss)
            file.close()
            # filename = "output_results_dis_abaltion.csv"
            # file = open(filename,"w")
            # writer = csv.writer(file)
            # writer.writerow(perturbed_dis)
            # writer.writerow(pgd_dis)
            # file.close()
            # filename = "output_results_with_autoencoder.csv"
            # file = open(filename,"w")
            # writer = csv.writer(file)
            # writer.writerow(mse_dis)
            # file.close()
            # print(np.mean(mse_dis))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--attack', type=str, default="random", help="attack types, random or minq")
    parser.add_argument('--env', type=str, default='PongNoFrameskip-v4', help='environment types')
    parser.add_argument('--model', type=str, default='natural', help="defense types, natural, sa-dqn-pgd, sa-dqn-convex, wocar, diffsuion_history")
    parser.add_argument('--record_change', type= bool, default= False, help="record semantic change images")
    args = parser.parse_args()
    main(args)
