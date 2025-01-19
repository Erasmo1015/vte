# import os
# os.chdir("/home/zichang/proj/IQ-Learn/iq_learn/encoder")
from torch.nn.modules.linear import Linear
from modules import *
from utils import *
import torch.nn as nn
import numpy as np
from transformer import ShallowTransformer, DeepTransformer
class HierarchicalStateSpaceModel(nn.Module):
    def __init__(
        self,
        action_encoder,
        encoder,
        decoder,
        belief_size,
        state_size,
        num_layers,
        max_seg_len,
        max_seg_num,
        latent_n=10,
        use_min_length_boundary_mask=False,
        ddo=False,
        output_normal=True
    ):
        super(HierarchicalStateSpaceModel, self).__init__()
        ######################
        # distribution size ##
        ######################
        self.dist_size = 10 # dist_size = cond_size
        self.mean = -1
        self.std = -1
        self.latent_n = latent_n

        ################
        # network size #
        ################
        # abstract level
        self.abs_belief_size = belief_size
        self.abs_state_size = belief_size
        self.abs_feat_size = belief_size

        # observation level
        self.obs_belief_size = belief_size
        self.obs_state_size = state_size
        self.obs_feat_size = belief_size

        # other size
        self.num_layers = num_layers
        self.feat_size = belief_size

        # sub-sequence information
        self.max_seg_len = max_seg_len
        self.max_seg_num = max_seg_num

        # for concrete distribution
        self.mask_beta = 1.0

        #################################
        # observation encoder / decoder #
        #################################
        self.action_encoder = action_encoder
        self.enc_obs = encoder
        self.dec_obs = decoder
        self.combine_action_obs = nn.Linear(
            belief_size + belief_size,
            belief_size,
        )

        #####################
        # boundary detector #
        #####################
        self.prior_boundary = PriorBoundaryDetector(input_size=self.obs_feat_size)
        self.post_boundary = PostBoundaryDetector(
            input_size=self.feat_size, num_layers=self.num_layers, causal=True
        )

        #####################
        # feature extractor #
        #####################
        self.abs_feat = LinearLayer(
            input_size=self.abs_belief_size + self.abs_state_size,
            output_size=self.abs_feat_size,
            nonlinear=nn.Identity(),
        )
        self.obs_feat = LinearLayer(
            input_size=self.obs_belief_size + self.obs_state_size,
            output_size=self.obs_feat_size,
            nonlinear=nn.Identity(),
        )

        #########################
        # belief initialization #
        #########################
        self.init_abs_belief = nn.Identity()
        self.init_obs_belief = nn.Identity()

        #############################
        # belief update (recurrent) #
        #############################
        self.update_abs_belief = RecurrentLayer(
            input_size=self.abs_state_size, hidden_size=self.abs_belief_size
        )
        self.update_obs_belief = RecurrentLayer(
            input_size=self.obs_state_size + self.abs_feat_size,
            hidden_size=self.obs_belief_size,
        )

        #####################
        # posterior encoder #
        #####################
        self.abs_post_fwd = RecurrentLayer(
            input_size=self.feat_size, hidden_size=self.abs_belief_size
        )
        self.abs_post_bwd = RecurrentLayer(
            input_size=self.feat_size, hidden_size=self.abs_belief_size
        )
        self.obs_post_fwd = RecurrentLayer(
            input_size=self.feat_size, hidden_size=self.obs_belief_size
        )

        ####################
        # prior over state #
        ####################
        self.prior_abs_state = DiscreteLatentDistributionVQ(
            input_size=self.abs_belief_size, latent_n=latent_n
        )
        self.prior_obs_state = LatentDistribution(
            input_size=self.obs_belief_size, latent_size=self.obs_state_size
        )

        ########################
        # posterior over state #
        ########################
        self.post_abs_state = DiscreteLatentDistributionVQ(
            input_size=self.abs_belief_size + self.abs_belief_size,
            latent_n=latent_n,
            feat_size=self.abs_belief_size,
        )
        self.post_obs_state = LatentDistribution(
            input_size=self.obs_belief_size + self.abs_feat_size,
            latent_size=self.obs_state_size,
            output_normal=output_normal,
        )

        self.z_embedding = LinearLayer(
            input_size=latent_n, output_size=self.abs_state_size
        )

        self._use_min_length_boundary_mask = use_min_length_boundary_mask
        self.ddo = ddo
        self._output_normal = output_normal
        self.state_linear_transform = LinearLayer(
            input_size=17, output_size=self.abs_state_size
        )
        self.action_linear_transform = LinearLayer(
            input_size=6, output_size=self.abs_state_size
        )

        ##########################
        # distribution generator #
        ##########################
        self.z_logit_feat = None
        self.m_feat = None
        self.transformer = None
        self.compact_last = None
    
        self.prob_encoder = False
    # sampler
    def boundary_sampler(self, log_alpha):
        # sample and return corresponding logit
        if self.training:
        # if False:
            log_sample_alpha = gumbel_sampling(log_alpha=log_alpha, temp=self.mask_beta)
        else:
            log_sample_alpha = log_alpha / self.mask_beta

        # probability
        log_sample_alpha = log_sample_alpha - torch.logsumexp(
            log_sample_alpha, dim=-1, keepdim=True
        )
        sample_prob = log_sample_alpha.exp()
        sample_data = torch.eye(2, dtype=log_alpha.dtype, device=log_alpha.device)[
            torch.max(sample_prob, dim=-1)[1]
        ]

        # sample with rounding and st-estimator
        sample_data = sample_data.detach() + (sample_prob - sample_prob.detach())

        # return sample data and logit
        return sample_data, log_sample_alpha

    # set prior boundary prob
    def regularize_prior_boundary(self, log_alpha_list, boundary_data_list):
        # only for training
        if not self.training:
            return log_alpha_list

        #################
        # sequence size #
        #################
        num_samples = boundary_data_list.size(0)
        seq_len = boundary_data_list.size(1)

        ###################
        # init seg static #
        ###################
        seg_num = log_alpha_list.new_zeros(num_samples, 1)
        seg_len = log_alpha_list.new_zeros(num_samples, 1)

        #######################
        # get min / max logit #
        #######################
        one_prob = 1 - 1e-3
        max_scale = np.log(one_prob / (1 - one_prob))

        near_read_data = log_alpha_list.new_ones(num_samples, 2) * max_scale
        near_read_data[:, 1] = -near_read_data[:, 1]
        near_copy_data = log_alpha_list.new_ones(num_samples, 2) * max_scale
        near_copy_data[:, 0] = -near_copy_data[:, 0]

        # for each step
        new_log_alpha_list = []
        for t in range(seq_len):
            ##########################
            # (0) get length / count #
            ##########################
            read_data = boundary_data_list[:, t, 0].unsqueeze(-1)
            copy_data = boundary_data_list[:, t, 1].unsqueeze(-1)
            seg_len = read_data * 1.0 + copy_data * (seg_len + 1.0)
            seg_num = read_data * (seg_num + 1.0) + copy_data * seg_num
            over_len = torch.ge(seg_len, self.max_seg_len).float().detach()
            over_num = torch.ge(seg_num, self.max_seg_num).float().detach()

            ############################
            # (1) regularize log_alpha #
            ############################
            # if read enough times (enough segments), stop
            new_log_alpha = (
                over_num * near_copy_data + (1.0 - over_num) * log_alpha_list[:, t]
            )

            # if length is too long (long segment), read
            new_log_alpha = over_len * near_read_data + (1.0 - over_len) * new_log_alpha

            ############
            # (2) save #
            ############
            new_log_alpha_list.append(new_log_alpha)

        # return
        return torch.stack(new_log_alpha_list, dim=1)

    # forward for reconstruction
    def forward(self, obs_data_list, action_list, seq_size, init_size):
        #############
        # data size #
        #############
        num_samples = action_list.size(0)
        full_seq_size = action_list.size(1)  # [B, S, C, H, W]

        #######################
        # observation encoder #
        #######################
        if action_list.size(-1) == full_seq_size: # if action list is (B, S), make it (B, S, 1)
            enc_obs_list = torch.squeeze(obs_data_list) # [B, S, D]
            enc_obs_list = self.enc_obs(enc_obs_list)

            action_list = action_list.unsqueeze(-1)  # From (B, S) to (B, S, 1)
            enc_action_list = self.action_encoder(action_list) # Now enc_action_list is [B, S, D] without squeezing
        elif action_list.size(-1)!= 1 and action_list.size(0)!=1: # if action dimension is 1, do not squeeze it
            enc_obs_list =  torch.squeeze(obs_data_list) # [B, S, D]
            enc_obs_list = self.enc_obs(enc_obs_list)

            enc_action_list = torch.squeeze(action_list) # [B, S, D]
            enc_action_list = self.action_encoder(enc_action_list)
        
        else:
            enc_obs_list =  obs_data_list # [B, S, D]
            enc_obs_list = self.enc_obs(enc_obs_list)

            enc_action_list = action_list # [B, S, D]
            enc_action_list = self.action_encoder(enc_action_list)

        # Shift sequence length dimension forward and 0 out first one
        shifted_enc_actions = torch.roll(enc_action_list, 1, 1)
        mask = torch.ones_like(shifted_enc_actions, device=shifted_enc_actions.device)
        mask[:, 0, :] = 0
        shifted_enc_actions = shifted_enc_actions * mask

        # Print the shapes to debug
        # print(f"enc_action_list shape: {enc_action_list.shape}")  # Expected: [B, 1000, 1]
        # print(f"enc_obs_list shape: {enc_obs_list.shape}")        # Expected: [B, 1000, D]
        # print(f"shifted_enc_actions shape: {shifted_enc_actions.shape}")  # Expected: [B, 1000, 1]
        # Ensure enc_obs_list has the same number of dimensions as enc_action_list
        if enc_obs_list.dim() > enc_action_list.dim() or enc_obs_list.dim() > shifted_enc_actions.dim():
            enc_action_list = enc_action_list.unsqueeze(0)
            shifted_enc_actions = shifted_enc_actions.unsqueeze(0)
        elif enc_action_list.dim() > enc_obs_list.dim() or shifted_enc_actions.dim() > enc_obs_list.dim():
            enc_obs_list = enc_obs_list.unsqueeze(0)
            
        
        # print(f"squeezed enc_action_list shape: {enc_action_list.shape}")  # Expected: [B, 1000, 1]
        # print(f"squeezed enc_obs_list shape: {enc_obs_list.shape}")        # Expected: [B, 1000, D]
        # print(f"squeezed shifted_enc_actions shape: {shifted_enc_actions.shape}")  # Expected: [B, 1000, 1]

        # check if the dimensions are (B, S, D)
        if enc_obs_list.dim() > 3:
            enc_obs_list = enc_obs_list.squeeze()
        if enc_action_list.dim() > 3:
            enc_action_list = enc_action_list.squeeze()
        if shifted_enc_actions.dim() > 3:
            shifted_enc_actions = shifted_enc_actions.squeeze()
            
        enc_combine_obs_action_list = self.combine_action_obs(
            torch.cat((enc_action_list, enc_obs_list), -1)
        )
        shifted_combined_action_list = self.combine_action_obs(
            torch.cat((shifted_enc_actions, enc_obs_list), -1)
        )

        ######################
        # boundary sampling ##
        ######################
        post_boundary_log_alpha_list = self.post_boundary(shifted_combined_action_list)
        boundary_data_list, post_boundary_sample_logit_list = self.boundary_sampler(
            post_boundary_log_alpha_list
        )
        boundary_data_list[:, : (init_size + 1), 0] = 1.0
        boundary_data_list[:, : (init_size + 1), 1] = 0.0

        if self._use_min_length_boundary_mask:
            mask = torch.ones_like(boundary_data_list)
            for batch_idx in range(boundary_data_list.shape[0]):
                reads = torch.where(boundary_data_list[batch_idx, :, 0] == 1)[0]
                prev_read = reads[0]
                for read in reads[1:]:
                    if read - prev_read <= 2:
                        mask[batch_idx][read] = 0
                    else:
                        prev_read = read

            boundary_data_list = boundary_data_list * mask
            boundary_data_list[:, :, 1] = 1 - boundary_data_list[:, :, 0]

        boundary_data_list[:, : (init_size + 1), 0] = 1.0
        boundary_data_list[:, : (init_size + 1), 1] = 0.0
        boundary_data_list[:, -init_size:, 0] = 1.0
        boundary_data_list[:, -init_size:, 1] = 0.0

        ######################
        # posterior encoding #
        ######################
        abs_post_fwd_list = []
        abs_post_bwd_list = []
        obs_post_fwd_list = []
        abs_post_fwd = action_list.new_zeros(num_samples, self.abs_belief_size).float()
        abs_post_bwd = action_list.new_zeros(num_samples, self.abs_belief_size).float()
        obs_post_fwd = action_list.new_zeros(num_samples, self.obs_belief_size).float()
        for fwd_t, bwd_t in zip(range(full_seq_size), reversed(range(full_seq_size))):
            # forward encoding
            fwd_copy_data = boundary_data_list[:, fwd_t, 1].unsqueeze(-1)  # (B, 1)
            abs_post_fwd = self.abs_post_fwd(
                enc_combine_obs_action_list[:, fwd_t], abs_post_fwd
            )  # abs_post_fwd is psi for z
            obs_post_fwd = self.obs_post_fwd(
                enc_obs_list[:, fwd_t], fwd_copy_data * obs_post_fwd
            )  # obs_post_fwd is phi for s
            abs_post_fwd_list.append(abs_post_fwd)
            obs_post_fwd_list.append(obs_post_fwd)
            # backward encoding
            bwd_copy_data = boundary_data_list[:, bwd_t, 1].unsqueeze(-1)
            abs_post_bwd = self.abs_post_bwd(
                enc_combine_obs_action_list[:, bwd_t], abs_post_bwd
            )
            abs_post_bwd_list.append(abs_post_bwd)
        abs_post_bwd_list = abs_post_bwd_list[::-1]

        #############
        # init list #
        #############
        obs_rec_list = []
        prior_abs_state_list = []
        post_abs_state_list = []
        prior_obs_state_list = []
        post_obs_state_list = []
        prior_boundary_log_alpha_list = []
        selected_option = []
        onehot_z_list = []
        abs_state_list = []
        vq_loss_list = []
        z_logit_list = []

        #######################
        # init state / latent #
        #######################
        abs_belief = action_list.new_zeros(num_samples, self.abs_belief_size).float()
        abs_state = action_list.new_zeros(num_samples, self.abs_state_size).float()
        obs_belief = action_list.new_zeros(num_samples, self.obs_belief_size).float()
        obs_state = action_list.new_zeros(num_samples, self.obs_state_size).float()
        # this zero is ignored because first time step is always read
        p = torch.zeros(num_samples, self.post_abs_state.latent_n).to(abs_state.device)

        ######################
        # forward transition #
        ######################
        option = p
        for t in range(init_size, init_size + seq_size):
            #####################
            # (0) get mask data #
            #####################
            read_data = boundary_data_list[:, t, 0].unsqueeze(-1)
            copy_data = boundary_data_list[:, t, 1].unsqueeze(-1)

            #############################
            # (1) sample abstract state #
            #############################

            abs_belief = abs_post_fwd_list[t - 1] * 0

            vq_loss, z, perplexity, onehot_z, z_logit, code_book = self.post_abs_state(
                concat(abs_post_fwd_list[t - 1], abs_post_bwd_list[t])
            )
            abs_state = read_data * z + copy_data * abs_state

            z_logit_list.append(z_logit)

            abs_feat = self.abs_feat(
                concat(abs_belief, abs_state)
            )
            selected_state = np.argmax(
                onehot_z.detach().cpu().numpy(), axis=-1
            )  # size of batch
            onehot_z_list.append(onehot_z)

            ################################
            # (2) sample observation state #
            ################################
            obs_belief = read_data * self.init_obs_belief(
                abs_feat
            ) + copy_data * self.update_obs_belief(
                concat(obs_state, abs_feat), obs_belief
            )  # this is h
            obs_belief *= 0
            prior_obs_state = self.prior_obs_state(obs_belief)
            if self._output_normal:
              post_obs_state = self.post_obs_state(concat(enc_obs_list[:, t], abs_feat))
            else:
              # Use recurrent embedder
              post_obs_state = self.post_obs_state(concat(obs_post_fwd_list[t], abs_feat))

            if self._output_normal:
              if self.ddo:
                  obs_state = post_obs_state.mean
              else:
                  obs_state = post_obs_state.rsample()
            else:
              obs_state = post_obs_state
            obs_feat = self.obs_feat(concat(obs_belief, obs_state))

            ##########################
            # (3) decode observation #
            ##########################
            obs_rec_list.append(obs_feat)

            ##################
            # (4) mask prior #
            ##################
            prior_boundary_log_alpha = self.prior_boundary(obs_feat)

            ############
            # (5) save #
            ############
            prior_boundary_log_alpha_list.append(prior_boundary_log_alpha)
            # prior_abs_state_list.append(prior_abs_state)
            # post_abs_state_list.append(post_abs_state)
            prior_obs_state_list.append(prior_obs_state)
            post_obs_state_list.append(post_obs_state)
            selected_option.append(selected_state)
            abs_state_list.append(abs_state)
            vq_loss_list.append(vq_loss)

        #### get logits ####
        # onehot_z_list traj_len 64 10
        logit_list = [i.cpu().detach().numpy() for i in z_logit_list]
        # traj_len 64
        length = len(logit_list[0]) # 64
        logit_list_t = [[] for i in range(length)]
        # transpose z_list
        for i in range(len(logit_list)): # traj_len
            for j in range(len(logit_list[i])): # 64
                logit_list_t[j].append(logit_list[i][j])
        # z_list_t 64 traj_len
        # one_traj_z = z_list_t[0]

        # z_logit_list -> logit_list_t_tensor
        logit_list_tensor = [i for i in z_logit_list]
        # traj_len 64
        length = len(logit_list[0]) # 64
        logit_list_t_tensor = [[] for i in range(length)]
        # transpose z_list
        for i in range(len(logit_list_tensor)): # traj_len
            for j in range(len(logit_list_tensor[i])): # 64
                logit_list_t_tensor[j].append(logit_list_tensor[i][j])

        # boundary m 64 1000 2
        # boundary_list = boundary_data_list[:,init_size:init_size + seq_size,:].cpu().detach().numpy()
        # boundary_list 64 998 2



        ##### get embeddings #####

        # onehot_z_list 16 64 10
        z_list = [np.argmax(i.cpu().numpy(), axis=1) for i in onehot_z_list]
        # 16 64
        length = len(z_list[0]) # 64
        z_list_t = [[] for i in range(length)]
        # transpose z_list
        for i in range(len(z_list)): # 16
            for j in range(len(z_list[i])): # 64
                z_list_t[j].append(z_list[i][j])
        # z_list_t 64 16
        # one_traj_z = z_list_t[0]
        def count_z(one_traj_z):
            unique = [one_traj_z[0]]
            unique_index = [0 for i in range(len(one_traj_z))]
            unique_index[0] = 1
            for i in range(1, len(one_traj_z)):
                if one_traj_z[i]!= unique[-1]:
                    unique.append(one_traj_z[i])
                    unique_index[i] = 1
            # print("Unique skills in one traj:", unique)
            return unique,unique_index
        unique_z_list = []
        unique_z_index_list = []
        for i in z_list_t:
            count, unique_index = count_z(i)
            unique_z_list.append(count)
            unique_z_index_list.append(unique_index)

        #### get the boundary m ####
        # boundary_data_list 64 998 2
        m_list = boundary_data_list[:,init_size:init_size + seq_size,0].cpu().detach().numpy()

        # - stack the embeddings (Option: logits on m=1)
        # embeddings 64 x
        # point multiplication of traj_logit and unique_z_index_list
        m_count_list = [[] for i in m_list]
        emb_list = []
        for index, traj_logit  in enumerate(logit_list_t):
            # make pooling of traj_option so that (traj_len, 10) becomes (10)
            new_logit = traj_logit[0]
            logit_per_m = []
            m = m_list[index]
            for i in range(len(m)):
                if m[i] == 1:
                    logit_per_m.append(traj_logit[i])
                    new_logit = traj_logit[i]
                    m_count_list[index].append(1)
                # # weighted version: 
                # else:
                #     logit_per_m.append(new_logit)
            # logit_per_logit_z size X 10
            logit = np.average(logit_per_m, axis=0)
            emb_list.append(logit)

        ############################################################################################
        ###################### GET THE DISTRIBUTION ################################################
        ############################################################################################
        
        # convert logit_list_t_tensor to tensor
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logit_tensors = torch.stack([torch.stack(row) for row in logit_list_t_tensor]).to(device) 
        m_tensors = boundary_data_list[:,init_size:init_size + seq_size,:]
        # self.z_logit_feat = LinearLayer(input_size=self.latent_n, output_size=self.dist_size)
        # self.m_feat = LinearLayer(input_size=2, output_size=self.dist_size)
        # self.transformer = nn.Transformer(d_model=256, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1)
        # self.compact_last = LinearLayer(input_size=self.dist_size*2, output_size=self.dist_size)
        
        self.logit_tensors = logit_tensors
        self.m_tensors = m_tensors
        if self.prob_encoder:
            z_logit_feat = self.z_logit_feat(logit_tensors)
            m_feat = self.m_feat(m_tensors)
            
            concated = torch.cat((z_logit_feat, m_feat), dim=2)
            # 998 128 + 998 128 = 998 256
            transformed = self.transformer(concated, None)
            # last_token = transformed[:, -1, :] # 1 1000 256 -> 1 256
            last_token = transformed 
            # 1 256
            # compacted = self.compact_last(last_token)
            # # 1 128
            # self.mean = compacted[:, :self.dist_size] # 64
            # self.std = compacted[:, self.dist_size:] # 64
            self.mean = self.mu_layer(last_token)
            self.std = self.logvar_layer(last_token)

            
        # print(f"logit shape: {z_logit_feat.shape}") # 64,998,10
        # print(f"m shape: {m_feat.shape}")
        # print(f"concated shape: {concated.shape}")
        # print(f"transformed shape: {transformed.shape}")
        # print(f"last_token shape: {last_token.shape}")
        # print(f"compacted shape: {compacted.shape}")
        # print(f"mean shape: {self.mean.shape}")
        # print(f"std shape: {self.std.shape}")

        # # - stack the embeddings (Option: logits on unique skill)
        # # embeddings 64 x
        # emb_list = []
        # for index, traj_logit  in enumerate(logit_list_t):
        #     # make pooling of traj_option so that (X, 10) becomes (10)
        #     logit_per_unique_z = []
        #     traj_unique_z_index = unique_z_index_list[index]
        #     for i in range(len(traj_unique_z_index)):
        #         if traj_unique_z_index[i] == 1:
        #             logit_per_unique_z.append(traj_logit[i])
        #     # logit_per_logit_z size X 10
        #     logit = np.average(logit_per_unique_z, axis=0)
        #     emb_list.append(logit)
        # # - stack the embeddings (Option: codebook embedding)
        # # embeddings 64 x
        # emb_list = []
        # for traj_option in unique_z_list:
        #     traj_emb = [code_book[i] for i in traj_option]
        #     feat = torch.stack(traj_emb,dim=1)
        #     # 128 x
        #     avg_pool = nn.AdaptiveAvgPool1d(1)
        #     feat = avg_pool(feat)
        #     emb_list.append(feat)
        
        # # compare two vectors (compare two traj)
        # emb_list = [i.cpu().detach().numpy() for i in emb_list]
        # # dist_list = [np.linalg.norm(emb_list[0]-i) for i in emb_list]
        # # dist_list[0] = sum(dist_list)/len(dist_list) # prevent 0 to be compared as min distance
        # # print("Traj 0 skill order:",embeddings[0])
        # # print("Longest distance:",max(dist_list))
        # # print("Longest distance traj index:",dist_list.index(max(dist_list)))
        # # print("Longest distance traj skill order:",embeddings[dist_list.index(max(dist_list))])
        # # print("Shortest distance:",min(dist_list))
        # # print("Shortest distance traj index:",dist_list.index(min(dist_list)))
        # # print("Shortest distance traj skill order:",embeddings[dist_list.index(min(dist_list))])
        # # print()

        # decode all together
        obs_rec_list = torch.stack(obs_rec_list, dim=1)

        obs_rec_list = self.dec_obs(obs_rec_list.view(num_samples * seq_size, -1))

        # (batch_size, sequence length, action size)
        obs_rec_list = obs_rec_list.view(num_samples, seq_size, -1)

        # stack results
        prior_boundary_log_alpha_list = torch.stack(
            prior_boundary_log_alpha_list, dim=1
        )

        # remove padding
        boundary_data_list = boundary_data_list[:, init_size : (init_size + seq_size)]
        post_boundary_log_alpha_list = post_boundary_log_alpha_list[
            :, (init_size + 1) : (init_size + 1 + seq_size)
        ]
        post_boundary_sample_logit_list = post_boundary_sample_logit_list[
            :, (init_size + 1) : (init_size + 1 + seq_size)
        ]

        # fix prior by constraints
        prior_boundary_log_alpha_list = self.regularize_prior_boundary(
            prior_boundary_log_alpha_list, boundary_data_list
        )

        # compute log-density
        prior_boundary_log_density = log_density_concrete(
            prior_boundary_log_alpha_list,
            post_boundary_sample_logit_list,
            self.mask_beta,
        )
        post_boundary_log_density = log_density_concrete(
            post_boundary_log_alpha_list,
            post_boundary_sample_logit_list,
            self.mask_beta,
        )

        # compute boundary probability
        prior_boundary_list = F.softmax(
            prior_boundary_log_alpha_list / self.mask_beta, -1
        )[..., 0]
        post_boundary_list = F.softmax(
            post_boundary_log_alpha_list / self.mask_beta, -1
        )[..., 0]
        prior_boundary_list = Bernoulli(probs=prior_boundary_list)
        post_boundary_list = Bernoulli(probs=post_boundary_list)
        boundary_data_list = boundary_data_list[..., 0].unsqueeze(-1)

        # process options
        selected_option = np.stack(selected_option).transpose((1, 0))  # size (B, S)
        onehot_z_list = torch.stack(onehot_z_list, axis=1)  # (B, S, Z)

        # process vq loss
        vq_loss_list = torch.stack(vq_loss_list)

        # return
        return [
            obs_rec_list,
            prior_boundary_log_density,
            post_boundary_log_density,
            prior_obs_state_list,
            post_obs_state_list,
            boundary_data_list,
            prior_boundary_list,
            post_boundary_list,
            abs_state_list,
            selected_option,
            onehot_z_list,
            vq_loss_list,
            logit_list_t_tensor, # z_logit list 1000, 10
            m_count_list,
            emb_list, # deterministic traj embedding 10
            unique_z_list,
            z_list_t # one hot z list
        ]
    
    # def get_dist(self):
    #     # form a diag Gaussian using self.mean and self.std
    #     mean = self.mean.cpu().detach().numpy()
    #     std = self.std.cpu().detach().numpy()
    #     distribution = np.random.normal(loc=64, scale=64, size=1000)
    #     return distribution
        
    def get_dist_params(self):
        return self.mean, self.std

    def get_logit_m(self):
        logit_arrays = [i.cpu().detach().numpy() for i in self.logit_tensors]
        m_arrays = [i.cpu().detach().numpy() for i in self.m_tensors]
        return logit_arrays, m_arrays
    
    def get_dist(self, logit_tensors, m_tensors):
        """
        Forward the logit and m tensors to get the distribution sample
        """
        self.logit_tensors = logit_tensors
        self.m_tensors = m_tensors
        z_logit_feat = self.z_logit_feat(logit_tensors)
        m_feat = self.m_feat(m_tensors)
        
        concated = torch.cat((z_logit_feat, m_feat), dim=2)
        # 998 128 + 998 128 = 998 256
        transformed = self.transformer(concated, None)
        # last_token = transformed[:, -1, :] # 1 1000 256 -> 1 256
        last_token = transformed 
        # 1 256
        # compacted = self.compact_last(last_token)
        # # 1 128
        # self.mean = compacted[:, :self.dist_size] # 64
        # self.std = compacted[:, self.dist_size:] # 64
        self.mean = self.mu_layer(last_token)
        self.std = self.logvar_layer(last_token)
        def reparameterize(mu, logvar):
            """
            Reparameterization trick to sample from N(mu, var) from
            N(0,1).
            :param mu: (Tensor) Mean of the latent Gaussian [B x D]
            :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
            :return: (Tensor) [B x D]
            """
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps * std + mu
        cond = reparameterize(self.mean, self.std)
        cond = cond.squeeze(0)
        return cond
    
    

    def instantiate_prob_encoder(self, dist_size=None, seq_len=1000, key_hidden_size=256,value_hidden_size=256):
        '''
        Instantiate the probability encoder
        :param dist_size: size of the distribution, equal to the condition size
        '''
        self.dist_size = dist_size if dist_size is not None else 10
        self.z_logit_feat = LinearLayer(input_size=self.latent_n, output_size=int(key_hidden_size/2))
        self.m_feat = LinearLayer(input_size=2, output_size=int(key_hidden_size/2))
        self.transformer = DeepTransformer(
            key_hidden_size,
            seq_len,
            key_hidden_size=key_hidden_size, 
            value_hidden_size=value_hidden_size,
            num_blocks=1
        )
        self.compact_last = LinearLayer(input_size=key_hidden_size, output_size=self.dist_size*2)
        self.mu_layer = LinearLayer(input_size=key_hidden_size, output_size=self.dist_size)
        self.logvar_layer = LinearLayer(input_size=key_hidden_size, output_size=self.dist_size)
        self.prob_encoder = True
        return self.prob_encoder
        
    def abs_marginal(self, obs_data_list, action_list, seq_size, init_size, n_sample=3):
        #############
        # data size #
        #############
        num_samples = action_list.size(0)
        full_seq_size = action_list.size(1)  # [B, S, C, H, W]

        #######################
        # observation encoder #
        #######################
        enc_obs_list = self.enc_obs(obs_data_list)

        if action_list.size(-1) == full_seq_size: # if action list is (B, S), make it (B, S, 1)
            action_list = action_list.unsqueeze(-1)  # From (B, S) to (B, S, 1)
        enc_action_list = self.action_encoder(action_list)

        # Shift sequence length dimension forward and 0 out first one
        shifted_enc_actions = torch.roll(enc_action_list, 1, 1)
        mask = torch.ones_like(shifted_enc_actions, device=shifted_enc_actions.device)
        mask[:, 0, :] = 0
        shifted_enc_actions = shifted_enc_actions * mask

        enc_combine_obs_action_list = self.combine_action_obs(
            torch.cat((enc_action_list, enc_obs_list), -1)
        )
        shifted_combined_action_list = self.combine_action_obs(
            torch.cat((shifted_enc_actions, enc_obs_list), -1)
        )

        ######################
        # boundary sampling ##
        ######################
        post_boundary_log_alpha_list = self.post_boundary(shifted_combined_action_list)
        marginal, n = 0, 0

        #############
        # init list #
        #############
        all_codes = []
        all_boundaries = []

        for _ in range(n_sample):
            boundary_data_list, _ = self.boundary_sampler(post_boundary_log_alpha_list)
            boundary_data_list[:, : (init_size + 1), 0] = 1.0
            boundary_data_list[:, : (init_size + 1), 1] = 0.0

            if self._use_min_length_boundary_mask:
                mask = torch.ones_like(boundary_data_list)
                for batch_idx in range(boundary_data_list.shape[0]):
                    reads = torch.where(boundary_data_list[batch_idx, :, 0] == 1)[0]
                    prev_read = reads[0]
                    for read in reads[1:]:
                        if read - prev_read <= 2:
                            mask[batch_idx][read] = 0
                        else:
                            prev_read = read

                boundary_data_list = boundary_data_list * mask
                boundary_data_list[:, :, 1] = 1 - boundary_data_list[:, :, 0]

            boundary_data_list[:, : (init_size + 1), 0] = 1.0
            boundary_data_list[:, : (init_size + 1), 1] = 0.0
            boundary_data_list[:, -init_size:, 0] = 1.0
            boundary_data_list[:, -init_size:, 1] = 0.0

            ######################
            # posterior encoding #
            ######################
            abs_post_fwd_list = []
            abs_post_bwd_list = []
            abs_post_fwd = action_list.new_zeros(
                num_samples, self.abs_belief_size
            ).float()
            abs_post_bwd = action_list.new_zeros(
                num_samples, self.abs_belief_size
            ).float()
            # generating the latent state
            for fwd_t, bwd_t in zip(
                range(full_seq_size), reversed(range(full_seq_size))
            ):
                # forward encoding
                abs_post_fwd = self.abs_post_fwd(
                    enc_combine_obs_action_list[:, fwd_t], abs_post_fwd
                )
                abs_post_fwd_list.append(abs_post_fwd)

                # backward encoding
                bwd_copy_data = boundary_data_list[:, bwd_t, 1].unsqueeze(-1)
                abs_post_bwd = self.abs_post_bwd(
                    enc_combine_obs_action_list[:, bwd_t], abs_post_bwd
                )
                abs_post_bwd_list.append(abs_post_bwd)
                # abs_post_bwd = bwd_copy_data * abs_post_bwd
            abs_post_bwd_list = abs_post_bwd_list[::-1]

            ######################
            # forward transition #
            ######################
            codes = []
            for t in range(init_size, init_size + seq_size):
                #####################
                # (0) get mask data #
                #####################
                read_data = boundary_data_list[:, t, 0].unsqueeze(-1)

                #############################
                # (1) sample abstract state #
                #############################
                _, _, _, onehot_z, z_logit, code_book = self.post_abs_state(
                    concat(abs_post_fwd_list[t - 1], abs_post_bwd_list[t])
                )
                log_p = z_logit
                log_p = log_p - torch.logsumexp(log_p, dim=-1, keepdim=True)
                prob = log_p.exp()
                marginal += (prob * read_data).sum(axis=0)
                n += read_data.sum()
                codes.append(onehot_z)

            all_codes.append(
                torch.stack(codes, axis=1)
            )  # permute such that the shape is (B, S, Z)
            all_boundaries.append(boundary_data_list[:, init_size:-init_size, 0])

        return marginal / n.detach(), all_codes, all_boundaries

    def encoding_cost(self, marginal, codes, boundaries):
        log_marginal = -torch.log(marginal)
        entropy = (log_marginal * marginal).sum()
        num_reads = boundaries.sum(dim=1).mean()
        return entropy * num_reads

    def initial_boundary_state(self, state):
        # Initial padding token
        # Padding action *embedding* is masked out
        enc_action = self.action_encoder(torch.zeros(1).long())
        enc_action = enc_action.squeeze(0) * 0
        padding_state = state * 0
        enc_obs = (
            self.enc_obs(padding_state.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
        )
        boundary_state = [self.combine_action_obs(torch.cat((enc_action, enc_obs), -1))]

        # First action is set to 0
        enc_action = self.action_encoder(torch.zeros(1).long()).squeeze(0)
        enc_obs = self.enc_obs(state.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
        boundary_state.append(
            self.combine_action_obs(torch.cat((enc_action, enc_obs), -1))
        )

        return boundary_state

    def z_terminates(self, state, prev_action, boundary_state=None):
        """Returns whether the current z terminates.

        Args:
            state: current state s_t
            prev_action: action a_{t - 1} taken in the previous timestep,
                returned by play_z
            boundary_state: previously returned value from z_terminates or None
                on the first timestep of a new z.

        Returns:
            terminate (bool): True if a new z should be sampled at s_t
            boundary_state: hidden state to be passed back to next call to
                z_terminates.
        """
        # List of combined action and obs embeddings of shape (embed_dim,)
        # The list is of length equal to number of timesteps T current z has
        # been active
        assert boundary_state is not None
        if boundary_state is None:
            boundary_state = []

        # Copy so you don't destructively modify
        boundary_state = list(boundary_state)

        # Dummy batch dimension
        enc_obs = self.enc_obs(state.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
        enc_action = self.action_encoder(
            torch.tensor(prev_action, device=enc_obs.device)
        )
        # (embed_dim,)
        enc_combine_obs_action = self.combine_action_obs(
            torch.cat((enc_action, enc_obs), 0)
        )

        boundary_state.append(enc_combine_obs_action)
        # (1, T, embed_dim)
        # Needs batch dimension inside of post boundary
        enc_combine_obs_action_list = torch.stack(boundary_state).unsqueeze(0)

        # (2,)
        read_logits = self.post_boundary(enc_combine_obs_action_list)[0, -1]
        terminate = read_logits[0] > read_logits[1]
        return terminate, boundary_state

    def play_z(self, z, state, hidden_state=None, recurrent=False):
        """Returns the action from playing the z at state: a ~ pi(a | s, z).

        Caller should call z_terminates after every call to play_z to determine
        if the same z should be used at the next timestep.

        Args:
            z (int): the option z to use, represented as a single integer (not
                1-hot).
            state: current state s_t

        Returns:
            action (int): a ~ pi(a | z, s_t)
        """
        if hidden_state is None:
            hidden_state = torch.zeros(1, self.abs_belief_size).float()

        # Convert integer
        # No batch dimension here
        # z = self.permitted_zs[z]
        z = self.post_abs_state.z_embedding(z)

        dummy_abs_belief = torch.zeros(self.abs_belief_size, device=z.device)
        abs_feat = self.abs_feat(torch.cat((dummy_abs_belief, z), 0))

        # Add dummy batch dimension before embedding, and then remove, since
        # some embedders require batching
        enc_obs = self.enc_obs(state.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
        if recurrent:
            hidden_state = self.obs_post_fwd(enc_obs.unsqueeze(0), hidden_state)
            enc_obs = hidden_state.squeeze(0)
        post_obs_state = self.post_obs_state(torch.cat((enc_obs, abs_feat), 0))
        obs_state = post_obs_state
        if self._output_normal:
            obs_state = post_obs_state.mean

        dummy_obs_belief = torch.zeros(abs_feat.shape[0], device=abs_feat.device)
        obs_feat = self.obs_feat(torch.cat((dummy_obs_belief, obs_state), 0))
        return torch.argmax(self.dec_obs(obs_feat)).item(), hidden_state


class EnvModel(nn.Module):
    def __init__(
        self,
        action_encoder,
        encoder,
        decoder,
        belief_size,
        state_size,
        num_layers,
        max_seg_len,
        max_seg_num,
        latent_n,
        rec_coeff=1.0,
        kl_coeff=1.0,
        use_abs_pos_kl=True,
        coding_len_coeff=10.0,
        use_min_length_boundary_mask=False,
        ddo=False,
        output_normal=True
    ):
        super(EnvModel, self).__init__()
        ################
        # network size #
        ################
        self.belief_size = belief_size
        self.state_size = state_size
        self.num_layers = num_layers
        self.max_seg_len = max_seg_len
        self.max_seg_num = max_seg_num
        self.latent_n = latent_n
        self.coding_len_coeff = coding_len_coeff
        self.use_abs_pos_kl = use_abs_pos_kl
        self.kl_coeff = kl_coeff
        self.rec_coeff = rec_coeff

        ##########################
        # baseline related flags #
        ##########################
        self.ddo = ddo

        ###############
        # init models #
        ###############
        # state space model
        self.state_model = HierarchicalStateSpaceModel(
            action_encoder=action_encoder,
            encoder=encoder,
            decoder=decoder,
            belief_size=self.belief_size,
            state_size=self.state_size,
            num_layers=self.num_layers,
            max_seg_len=self.max_seg_len,
            max_seg_num=self.max_seg_num,
            latent_n=self.latent_n,
            use_min_length_boundary_mask=use_min_length_boundary_mask,
            ddo=ddo,
            output_normal=output_normal
        )
        self._output_normal = output_normal

    def forward(self, obs_data_list, action_list, seq_size, init_size, obs_std=1.0):
        ############################
        # (1) run over state model #
        ############################
        [
            obs_rec_list,
            prior_boundary_log_density_list,
            post_boundary_log_density_list,
            prior_obs_state_list,
            post_obs_state_list,
            boundary_data_list,
            prior_boundary_list,
            post_boundary_list,
            abs_state_list,
            selected_option,
            onehot_z_list,
            vq_loss_list,
            z_logit_list,
            m_count_list,
            emb_list,
            unique_z_list,
            z_list
        ] = self.state_model(obs_data_list, action_list, seq_size, init_size)

        # mean, std = self.state_model.get_dist_params()
        # print(f"mean shape: {mean.shape}")
        # print(f"std shape: {std.shape}")
        # distribution = self.state_model.get_dist()
        # print(f"distribution shape: {distribution.shape}")
        # print(distribution)

        ########################################################
        # (2) compute obs_cost (sum over spatial and channels) #
        ########################################################
        # obs_rec_list: (batch_size, seq_len, action_dim)
        # action_list: (batch_size, seq_len)
        obs_cost = F.mse_loss(
            torch.flatten(obs_rec_list.reshape(-1, obs_rec_list.shape[-1])),
            action_list[:, init_size:-init_size].reshape(-1),
        )
        

        #######################
        # (3) compute kl_cost #
        #######################
        # compute kl related to states, since we are not using KL for RL
        # setting we avoid the computation
        if self._output_normal:
          kl_obs_state_list = []
          for t in range(seq_size):
             kl_obs_state = kl_divergence(post_obs_state_list[t], prior_obs_state_list[t])
             kl_obs_state_list.append(kl_obs_state.sum(-1))
          kl_obs_state_list = torch.stack(kl_obs_state_list, dim=1)

          # compute kl related to boundary
          kl_mask_list = post_boundary_log_density_list - prior_boundary_log_density_list
        else:
          kl_obs_state_list = torch.zeros(
              post_obs_state_list[0].shape[0], seq_size)

        ###############################
        # (4) compute encoding length #
        ###############################
        marginal, all_codes, all_boundaries = self.state_model.abs_marginal(
            obs_data_list, action_list, seq_size, init_size
        )
        encoding_length = self.state_model.encoding_cost(
            marginal, onehot_z_list, boundary_data_list.squeeze(-1)
        )

        if self.ddo:
            train_loss = self.rec_coeff * obs_cost.mean() + \
                    self.kl_coeff * kl_mask_list.mean() + \
                    self.coding_len_coeff * encoding_length + \
                    torch.mean(vq_loss_list)
        else:
            train_loss = (
                self.rec_coeff * obs_cost.mean()
                + self.kl_coeff * (kl_obs_state_list.mean() + kl_mask_list.mean())
                + self.coding_len_coeff * encoding_length
                + torch.mean(vq_loss_list)
            )

        pos_obs_state = [x for x in post_obs_state_list]
        if self._output_normal:
            pos_obs_state = [x.mean for x in post_obs_state_list]

        return {
            "rec_data": obs_rec_list,
            "mask_data": boundary_data_list,
            "obs_cost": obs_cost,
            "kl_abs_state": torch.zeros_like(kl_obs_state_list),
            "kl_obs_state": kl_obs_state_list,
            "kl_mask": kl_mask_list,
            "p_mask": prior_boundary_list.mean,
            "q_mask": post_boundary_list.mean,
            "p_ent": prior_boundary_list.entropy(),
            "q_ent": post_boundary_list.entropy(),
            "beta": self.state_model.mask_beta,
            "encoding_length": encoding_length,
            "marginal": marginal.detach().cpu().numpy(),
            "train_loss": train_loss,
            "option_list": selected_option,
            "pos_obs_state": torch.stack(pos_obs_state, axis=1),
            "abs_state": torch.stack(abs_state_list, axis=1),
            "all_boundaries": all_boundaries,
            "vq_loss_list": torch.mean(vq_loss_list).detach(),
            "z_logit_list": z_logit_list,
            "m_count_list": m_count_list,
            "embedding": emb_list,
            "unique_z_list": unique_z_list,
            "z_list": z_list,
        }

def concat(*data_list):
    return torch.cat(data_list, 1)

def gumbel_sampling(log_alpha, temp, margin=1e-4):
    noise = log_alpha.new_empty(log_alpha.size()).uniform_(margin, 1 - margin)
    gumbel_sample = -torch.log(-torch.log(noise))
    return torch.div(log_alpha + gumbel_sample, temp)

def log_density_concrete(log_alpha, log_sample, temp):
    exp_term = log_alpha - temp * log_sample
    log_prob = torch.sum(exp_term, -1) - 2.0 * torch.logsumexp(exp_term, -1)
    return log_prob