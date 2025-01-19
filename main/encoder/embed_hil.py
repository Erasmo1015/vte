import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" # before import torch, keras
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import psutil
# print(f'Number of CPUs: {psutil.cpu_count()}')
p = psutil.Process()
# print(f'CPU pool before assignment: {p.cpu_affinity()}')
p.cpu_affinity(range(0,32))
# print(f'CPU pool after assignment: {p.cpu_affinity()}')
os.environ["PYTHONPATH"] = "/home/zichang/proj/adapt-image-models/love"
import argparse

import numpy as np
import torch
import tqdm

import dqn.config as cfg
import dqn.rl as rl
import utils
from tqdm import tqdm
import logging
import pickle
import gym
from pyvirtualdisplay import Display
from gym import wrappers
import matplotlib.pyplot as plt
from scipy.stats import zscore

LOGGER = logging.getLogger(__name__)



def run_exp(config, expert_file, datafile, embed_mode, cond_dim):
    import sys
    sys.path.append('/home/zichang/proj/IQ-Learn/iq_learn')
    hssm = torch.load(config.get("checkpoint")).cpu()
    hssm._use_min_length_boundary_mask = True
    hssm.eval()
    print("Model Loaded")
    if config.get("env") == "cheetah":
        # cus: no env
        full_loader = utils.cheetah_full_loader(1, expert_file)
        hssm.post_obs_state._output_normal = True
        hssm._output_normal = True
    elif config.get("env") == "hopper":
        full_loader = utils.hopper_full_loader(1, expert_file)
        hssm.post_obs_state._output_normal = True
        hssm._output_normal = True
    elif config.get("env") == "walker":
        full_loader = utils.walker_full_loader(1, expert_file)
        hssm.post_obs_state._output_normal = True
        hssm._output_normal = True
    elif config.get("env") == "ant":
        full_loader = utils.ant_full_loader(1, expert_file)
        hssm.post_obs_state._output_normal = True
        hssm._output_normal = True
    elif config.get("env") == "humanoid":
        full_loader = utils.humanoid_full_loader(1, expert_file)
        hssm.post_obs_state._output_normal = True
        hssm._output_normal = True
    elif config.get("env") == "cartpole":
        full_loader = utils.cartpole_full_loader(1, expert_file)
        hssm.post_obs_state._output_normal = True
        hssm._output_normal = True
    elif config.get("env") == "lunar":
        full_loader = utils.lunar_full_loader(1, expert_file)
        hssm.post_obs_state._output_normal = True
        hssm._output_normal = True
    elif config.get("env") == "pusher":
        full_loader = utils.pusher_full_loader(1, expert_file)
        hssm.post_obs_state._output_normal = True
        hssm._output_normal = True
    elif config.get("env") == "swimmer":
        full_loader = utils.swimmer_full_loader(1, expert_file)
        hssm.post_obs_state._output_normal = True
        hssm._output_normal = True
    elif config.get("env") == "humanoid_standup":
        full_loader = utils.humanoid_standup_full_loader(1, expert_file)
        hssm.post_obs_state._output_normal = True
        hssm._output_normal = True
    elif config.get("env") == "invertedp":
        full_loader = utils.invertedp_full_loader(1, expert_file)
        hssm.post_obs_state._output_normal = True
        hssm._output_normal = True
    elif config.get("env") == "lunarlander":
        full_loader = utils.lunarlander_full_loader(1, expert_file)
        hssm.post_obs_state._output_normal = True
        hssm._output_normal = True
    elif config.get("env") == "antmaze":
        full_loader = utils.antmaze_full_loader(1, expert_file)
        hssm.post_obs_state._output_normal = True
        hssm._output_normal = True
    elif config.get("env") == "kitchen":
        full_loader = utils.kitchen_full_loader(1, expert_file)
        hssm.post_obs_state._output_normal = True
        hssm._output_normal = True
    else:
        raise ValueError()
    # device = "cpu"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # device = torch.device("cuda", 0)
    hssm.to(device)
    seq_size = full_loader.dataset.seq_size
    init_size = 1
    b_idx = 0
    emb_list = {"num_m":[],"emb": [], "level":[], "num_z":[], "z":[], "logit_m":[]}
    count = 0
    for obs_list, action_list, level_list in tqdm(full_loader):
        # obs_list 100 1000 17
        # action_list 100 1000 6
        # trai_level_list 100
        b_idx += 1
        if embed_mode !="dummy":
            obs_list = obs_list.to(device)
            action_list = action_list.to(device)
            results = hssm(obs_list, action_list, seq_size, init_size)
            if embed_mode == "det": # deterministic encoding
                emb_list["emb"].extend(results[-3]) 
            elif embed_mode == "z_logit": # logit_list_t_tensor: 1000, 10
                embedding = np.array([[tensor.cpu().detach().numpy() for tensor in sublist] for sublist in results[-5]])
                embedding = np.mean(embedding, axis=1)
                emb_list["emb"].extend(embedding)
            else:
                mean, std = hssm.get_dist_params()
                if embed_mode == "mean": # mean as embedding 
                    mean = mean.detach().cpu().numpy()
                    emb_list["emb"].extend(mean)
                elif embed_mode == "prob": # probabilistic encoding
                    def reparameterize(mu, logvar):
                        """
                        Reparameterization trick to sample from N(mu, var) from
                        N(0,1).
                        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
                        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
                        :return: (Tensor) [B x D]
                        """
                        std = torch.exp(0.5 * logvar)
                        # std = std/100
                        eps = torch.randn_like(std)
                        return eps * std + mu
                    emb = reparameterize(mean, std)
                    emb = emb.detach().cpu().numpy()
                    emb_list["emb"].extend(emb)
                else:
                    raise ValueError("Invalid embedding mode")
            ## --> Use dummy value to replace the embedding
            # if count<=9:
            #     dummy_value = -1
            # else:
            #     dummy_value = 1
            # count += 1
            # # create dummy which is the same shape as result[-3] and the values are dummy_value
            # dummy = [[dummy_value for i in range(len(j))] for j in results[-3]]
            # emb_list["emb"].extend(dummy)

            emb_list["num_m"].extend([len(i) for i in results[-4]])
            emb_list["level"].extend(level_list)
            emb_list["num_z"].extend([len(i) for i in results[-2]])
            emb_list["z"].extend(results[-1])
            logit_arrays, m_arrays = hssm.get_logit_m()
            emb_list["logit_m"].append((logit_arrays, m_arrays))
            if b_idx >= 1000000:
                print("Stopped early at 1000000 batches")
                break
        elif embed_mode == "dummy":
            # make a [1, cond_dim] dummy condition copying the level_list as a float32
            dummy_cond = np.array(level_list).astype(np.float32)
            dummy_cond = [[dummy_cond[0] for i in range(cond_dim)]]
            emb_list["emb"].extend(dummy_cond)
        else:
            raise ValueError("Invalid embedding mode")
    ## --> Normalize the emb using z score normalization
    emb_list["emb"] = zscore(emb_list["emb"])
    os.makedirs(os.path.dirname(datafile), exist_ok=True)
    with open(datafile, 'wb') as f:
        pickle.dump(emb_list, f)
        
    return emb_list

def clustering_report(emb_list, exp_name, logname, n_features, env_name):
    # compare traj embeddings
    # option 1. k_means num_clusters = n_proficiency_levels
    # expected the same level in the same cluster
    from numpy import unique
    from numpy import where
    from sklearn.datasets import make_classification
    from sklearn.cluster import KMeans
    from matplotlib import pyplot
    from scipy.optimize import linear_sum_assignment
    # define dataset
    X, _ = make_classification(n_samples=1000, n_features=n_features, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=4)
    # define the model
    model = KMeans(n_clusters=n_features, n_init=10)
    # fit the model
    emb_list["emb"] = [np.squeeze(i) for i in emb_list["emb"]]
    # check if the emb is 1D
    if np.array(emb_list["emb"]).ndim == 1:
        emb_list["emb"] = np.array(emb_list["emb"]).reshape(-1,1)
    model.fit(emb_list["emb"])
    # assign a cluster to each example
    yhat = model.predict(emb_list["emb"])
    # retrieve unique clusters
    clusters = unique(yhat)
    # create scatter plot for samples from each cluster
    for cluster in clusters:
        # get row indexes for samples with this cluster
        row_ix = where(yhat == cluster)
        # create scatter of these samples
        pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
    # show the plot
    pyplot.show()
    # pyplot.savefig(f'plot/{exp_name}_kmeans.png')
    # calculate the ratio
    ratio = [[0 for i in range(n_features)] for i in range(n_features)]
    if "level" not in emb_list.keys(): # temporary fix for baseline embs missing level
        # 0, 1, 2 for proficiency levels corresponding to 0to9, 10to19, 20to29
        cluster_num = len(emb_list["emb"])/n_features
        emb_list["level"] = [int(i/cluster_num) for i in range(len(emb_list["emb"]))]
    for index, cluster in enumerate(yhat):
        proficiency_level = emb_list["level"][index]
        ratio[cluster][proficiency_level] += 1
    
    
    LOGGER.info("#" * 80)
    for index_i, i in enumerate(ratio):
        LOGGER.info(" >>> Cluster {} | Total Count: {}".format(index_i, sum(i)))
        for index_j, j in enumerate(i):
            if sum(i)>0:
                LOGGER.info("Proficiency Level {}: {:.2f}% | Count: {}".format(index_j, 100*j/sum(i), j))
            else:
                LOGGER.info("Proficiency Level {}: {:.2f}% | Count: 0".format(index_j, 0))
        LOGGER.info("")
    LOGGER.info("#" * 80)
    print("Log saved at {}".format(logname))

    real_acc = np.eye(n_features, n_features)
    pred_acc = ratio / (np.sum(ratio, axis=1, keepdims=True) + 1e-8) # avoid devision by 0
    # Calculate optimal assignment between real_acc and pred_acc using Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(-np.dot(real_acc, pred_acc.T))
    # Rearrange pred_acc rows according to optimal assignment
    pred_acc_matched = pred_acc[col_ind]

    # Calculate Mean Squared Error (MSE) based on optimal assignment
    mse = np.mean((real_acc - pred_acc_matched) ** 2)
    LOGGER.info(f"Clustering Mean Squared Error (MSE): {mse}.")
    print(f"Clustering Mean Squared Error (MSE): {mse}. Detailed log saved at {logname}.")

def pca(emb_list, exp_name, n_features, env_name, exp_id):
    from sklearn.decomposition import PCA
    import pandas as pd
    import matplotlib.pyplot as plt

    from sklearn.preprocessing import StandardScaler
    x = emb_list['emb']
    x = StandardScaler().fit_transform(x) # normalizing the features
    feat_cols = ['feature'+str(i) for i in range(x.shape[1])]
    normalised = pd.DataFrame(x,columns=feat_cols)

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)

    principal_Df = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

    plt.figure()
    plt.figure(figsize=(18,18))
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)
    plt.xlabel('Principal Component - 1',fontsize=50)
    plt.ylabel('Principal Component - 2',fontsize=50)
    # plt.title(f"Principal Component Analysis of {env_name}",fontsize=50)
    # plt.title("Hopper",fontsize=50)
    # plt.title(f"Walker2D",fontsize=50)
    plt.title("Half Cheetah",fontsize=50)

    targets = [0,1,2]
    colors = [(180/255, 180/255, 73/255), (60/255, 137/255, 138/255),(223/255, 126/255, 79/255)]
    finalDf = pd.concat([principal_Df, pd.Series(emb_list['level'], name='level')], axis = 1)
    for target, color in zip(targets,colors):
        indicesToKeep = finalDf['level'] == target
        plt.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                , finalDf.loc[indicesToKeep, 'principal component 2'], c = color, s = 200)
    # Shrink current axis's height by 10% on the bottom
    ax = plt.subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                    box.width, box.height * 0.9])
    legend = ['Low','Medium', 'Expert']
    # plt.legend(legend,prop={'size': 15})
    plt.legend(legend,loc='upper center', bbox_to_anchor=(0.5, -0.1),
          fancybox=True, shadow=True, ncol=5, prop={'size': 50})
    plt.show()
    datadir = f"plot/{env_name}/{exp_id}"
    if not os.path.exists(datadir):
        os.makedirs(datadir)
    savepic = os.path.join(datadir, exp_name + "_PCA.png")
    plt.savefig(savepic)
    print("Plot saved at {}".format(savepic))

def tSNE(emb_list, exp_name, n_features, env_name, exp_id):
    import numpy as np
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    x = emb_list['emb']
    x = StandardScaler().fit_transform(x) # normalizing the features
    XX_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(x)
    principal_Df = pd.DataFrame(data = XX_embedded
             , columns = ['component 1', 'component 2'])
    plt.figure()
    plt.figure(figsize=(18,18))
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)
    plt.xlabel('Component - 1',fontsize=50)
    plt.ylabel('Component - 2',fontsize=50)
    # plt.title(f"tSNE",fontsize=20)
    # plt.title(f"tSNE of {env_name}",fontsize=20)
    # plt.title(f"Hopper",fontsize=50)
    # plt.title(f"Walker2D",fontsize=50)
    plt.title(f"Half Cheetah",fontsize=50)
    

    targets = [0,1,2]
    colors = [(180/255, 180/255, 73/255), (60/255, 137/255, 138/255),(223/255, 126/255, 79/255)]
    finalDf = pd.concat([principal_Df, pd.Series(emb_list['level'], name='level')], axis = 1)
    for target, color in zip(targets,colors):
        indicesToKeep = finalDf['level'] == target
        plt.scatter(finalDf.loc[indicesToKeep, 'component 1']
                , finalDf.loc[indicesToKeep, 'component 2'], c = color, s = 200)
    # Shrink current axis's height by 10% on the bottom
    ax = plt.subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                    box.width, box.height * 0.9])
    legend = ['Low','Medium', 'Expert']
    # plt.legend(legend,prop={'size': 15})
    plt.legend(legend,loc='upper center', bbox_to_anchor=(0.5, -0.1),
          fancybox=True, shadow=True, ncol=5, prop={'size': 50})
    plt.show()
    datadir = f"plot/{env_name}/{exp_id}"
    if not os.path.exists(datadir):
        os.makedirs(datadir)
    savepic = os.path.join(datadir, exp_name + "_tSNE.png")
    plt.savefig(savepic)
    print("Plot saved at {}".format(savepic))

def render(action_list, vidsavedir = "./video" ):
    # render one traj and save the video    
    video_length = len(action_list)
    env = gym.make('HalfCheetah-v3')
    env = wrappers.Monitor(env, vidsavedir, force = True)
    env.reset()
    display = Display(visible=0, size=(1400, 900))
    display.start()
    try:
        for i in range(video_length):
            action = action_list[i].detach().cpu().numpy()
            obs, _, _, _ = env.step(action)
            env.render()
        env.close()
    except KeyboardInterrupt:
        pass
    finally:
        # Release everything if job is finished
        display.stop()

def flatten(matrix, length=-1):
    if length==-1:
        length = len(matrix[0])
    # 300 1000  -> 300 * length
    res = []
    for i in range(len(matrix)):
        res.extend(matrix[i][:length])
    return res

def render_skills(config, emb_list, exp_name):
    # render sub-traj for each skill
    if config.get("env") == "cheetah":
        # cus: no env
        full_loader = utils.cheetah_full_loader(1, "../experts/cheetah/HalfCheetah-v2_10_2402r.pkl,../experts/cheetah/HalfCheetah-v2_10_4208r.pkl,../experts/cheetah/HalfCheetah-v2_10_6301r.pkl")
    # if True:
    # # if config.get("env") == "hil":
    #     full_loader = utils.hil_full_loader(100)
    else:
        raise ValueError()
    data = [[] for i in range(3)]
    # data[0], data[1], data[2] = full_loader
    z = flatten(emb_list["z"])
    # action_list = [] 
    # for i in range(3):
    #     action_list.extend(data[i][1])
    action_list = full_loader.dataset.action
    # 300 1000
    action_list = flatten(action_list, length=998)
    # 300 * 998
    skill = z[0] # first skill
    start_index = 0
    videodir = "./video/skill/"+exp_name+"/"
    length = 300*998 # num of total timsteps to read from
    success = 0
    for i in tqdm(range(299, -1,-1)):
        start_index = i*998
        # skill = z[i*998]
        skill = z[0] # TODO: uncomment above
        for j in range(998):
            current_index = i*998 + j
            # if z[current_index]!=skill:
            if True:  # TODO: uncomment above
                # if  current_index-start_index>5 and skill==2:
                if True: # TODO: uncomment above
                    print("Rendering skill {}...".format(skill))
                    # render(action_list[start_index:current_index+1],videodir+str(skill))
                    render(action_list[start_index:start_index+998],videodir+str(skill)) # TODO: uncomment above
                    success += 1
                skill = z[current_index]
                start_index = current_index
    print("{} videos saved at ./videos".format(success))

def run_episode(env, policy, experience_observers=None, test=False,
                return_render=False):
    """Runs a single episode on the environment following the policy.

    Args:
        env (gym.Environment): environment to run on.
        policy (Policy): policy to follow.
        experience_observers (list[Callable] | None): each observer is called with
            with each experience at each timestep.

    Returns:
        episode (list[Experience]): experiences from the episode.
    """
    def maybe_render(env, instruction, action, reward, info, timestep):
        if return_render:
            render = env.render()
            render.write_text("Action: {}".format(str(action)))
            render.write_text("Instruction: {}".format(instruction))
            render.write_text("Reward: {}".format(reward))
            render.write_text("Timestep: {}".format(timestep))
            render.write_text("Info: {}".format(info))
            return render
        return None

    if experience_observers is None:
        experience_observers = []

    episode = []
    state = env.reset()
    timestep = 0
    renders = [maybe_render(env, state[1], None, 0, {}, timestep)]
    hidden_state = None
    while True:
        action, next_hidden_state = policy.act(state, hidden_state, test=test)
        next_state, reward, done, info = env.step(action)
        timestep += 1
        renders.append(maybe_render(env, next_state[1], action, reward, info, timestep))
        experience = rl.Experience(
                state, action, reward, next_state, done, info, hidden_state,
                next_hidden_state)
        episode.append(experience)
        for observer in experience_observers:
            observer(experience)

        if "experiences" in info:
            del info["experiences"]

        state = next_state
        hidden_state = next_hidden_state
        if done:
            return episode, renders

def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
            '-c', '--configs', action='append', default=["configs/default.json"])
    arg_parser.add_argument(
            '-b', '--config_bindings', action='append', default=[],
            help="bindings to overwrite in the configs.")
    arg_parser.add_argument(
            "-x", "--base_dir", default="experiments",
            help="directory to log experiments")
    arg_parser.add_argument(
            "-p", "--checkpoint", default=None,
            help="path to checkpoint directory to load from or None")
    arg_parser.add_argument(
            "-f", "--force_overwrite", action="store_true",
            help="Overwrites experiment under this experiment name, if it exists.")
    arg_parser.add_argument(
            "-s", "--seed", default=0, help="random seed to use.", type=int)
    arg_parser.add_argument("exp_name", help="name of the experiment to run")
    arg_parser.add_argument("expert_file", help="full path of expert demo")
    arg_parser.add_argument("--obs-std", type=float, default=1.0)
    arg_parser.add_argument("--batchsize", type=int, default=8)
    arg_parser.add_argument("--n_features", type=int, default=3)
    arg_parser.add_argument("--embed_mode", type=str, default="det", help="det, mean, dummy or prob", choices=["det", "mean", "dummy", "prob", "z_logit"])
    arg_parser.add_argument("--exp_id", type=str, default="no_id", help="experiment id for saving")
    arg_parser.add_argument("--cond_dim", type=int, default=10, help="condition dimension")
    args = arg_parser.parse_args()
    config = cfg.Config.from_files_and_bindings(
            args.configs, args.config_bindings)
    env_name = config.get("env")
    if not os.path.exists(f"result_clustering/{env_name}/{args.exp_id}"):
        os.makedirs(f"result_clustering/{env_name}/{args.exp_id}")    
    logname = os.path.join(f"result_clustering/{env_name}/{args.exp_id}", args.exp_name + ".log")
    logging.basicConfig(filename=logname,
                        filemode='w',
                        # format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        format='%(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Fetching data...")
    datadir = f"../cond/{env_name}/{args.exp_id}"
    os.makedirs(datadir, exist_ok=True)
    datafile = os.path.join(datadir,args.exp_name + ".pkl")
    if os.path.isfile(datafile):
        with open(datafile, 'rb') as f:
            emb_list = pickle.load(f)
        print("Loaded previous data")
    else:
        emb_list = run_exp(config, args.expert_file, datafile, args.embed_mode, args.cond_dim)
        print("Saved data")
    
    # LOGGER.info(">>> Num of skills in one traj: {}~{}, Average {}".format(min(emb_list["num_z"]), 
    #                                                                   max(emb_list["num_z"]), 
    #                                                                   sum(emb_list["num_z"])/len(emb_list["num_z"])))
    # LOGGER.info(">>> Num of boundaries m=1 in one traj: {}~{}, Average {}".format(min(emb_list["num_m"]), 
    #                                                                   max(emb_list["num_m"]), 
    #                                                                   sum(emb_list["num_m"])/len(emb_list["num_m"])))
    # collect traj embeddings
    # step 1: dict of {embedding, proficiency_level}
    clustering_report(emb_list, args.exp_name, logname, args.n_features, env_name)

    # step 2: PCA principle component analysis
    # draw a figure, different colors for different proficiency levels   
    pca(emb_list, args.exp_name, args.n_features, env_name, args.exp_id) 

    # step 3: tSNE
    tSNE(emb_list, args.exp_name, args.n_features, env_name, args.exp_id)

    # step 4: render one trajectory
    render_skills(config, emb_list, args.exp_name)

    # step 5: prediction



if __name__ == '__main__':
    main()
