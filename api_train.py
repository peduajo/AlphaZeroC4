from fastapi import FastAPI, Request
from lib import model

import os 
import copy

from types import SimpleNamespace

from torch.utils.data import DataLoader
import torch_tensorrt
from pathlib import Path
app = FastAPI()

from torch.utils.tensorboard import SummaryWriter

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

import numpy as np

alphazero_params = SimpleNamespace(**{
    'run_name':             'alphazero_connect4_resnet',
    'checkpoint_directory': 'models/experiment_1',
    'num_par_games':         3000,
    'weight_decay':          1e-4,
    'max_lr':                0.2,
    'initial_lr':            1e-4,
    'exp_decay_gamma':       0.975,
    'batch_size':            1024,
    'max_buffer_size':       2e6,
    'max_steps_per_iteration': 500,
    'resnet_num_blocks':     15,
    'resnet_hidden':         128,
    'max_grad_norm':         0.5,
    'target_kl':             0.75,
    'coef_policy':           1.0,
    'coef_value':            1.0,
    'initial_iteration':     0
})



class AlphaZero:
    def __init__(self, net, writer, optimizer):
        self.model = net
        
        self.writer = writer
        self.iteration = alphazero_params.initial_iteration
        self.checkpoint_models = alphazero_params.checkpoint_directory
        self.epoch_history = 0
        self.first_drop_lr = False
        self.second_drop_lr = False
        self.buffer = model.ExperienceBuffer(int(alphazero_params.max_buffer_size), self.model.device)
        self.optimizer = optimizer
        self.scheduler = optim.lr_scheduler.ExponentialLR(optimizer, alphazero_params.exp_decay_gamma)

        self.quantization_dataloader = None
        self.batch_size_quantization = 64     

        checkpoint_models_dir = Path(self.checkpoint_models)

        checkpoint_models_dir.mkdir(parents=True, exist_ok=True)

    def adjust_learning_rate_for_warmup(self, current_step, warmup_steps):
        lr = alphazero_params.max_lr * (current_step / warmup_steps)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr 
        

    def add_data_to_buffer(self, data_path):
        filenames = os.listdir(f"{data_path}states")
        state_tensors_list, policy_tensors_list, value_tensors_list, q_value_tensors_list = [], [], [], []

        for i in range(len(filenames)):
            state_tensors = torch.load(f"{data_path}states/{i}.zip")
            policy_tensors = torch.load(f"{data_path}policy/{i}.zip")
            value_tensors = torch.load(f"{data_path}values/{i}.zip")
            q_value_tensors = torch.load(f"{data_path}q_values/{i}.zip")
            state_tensors_list.append(state_tensors)
            policy_tensors_list.append(policy_tensors)
            value_tensors_list.append(value_tensors)
            q_value_tensors_list.append(q_value_tensors)

        state_tensors_iteration = torch.cat(state_tensors_list, dim=0)
        policy_tensors_iteration = torch.cat(policy_tensors_list, dim=0)
        value_tensors_iteration = torch.cat(value_tensors_list, dim=0)
        q_value_tensors_iteration = torch.cat(q_value_tensors_list, dim=0)

        memory = (state_tensors_iteration, policy_tensors_iteration, q_value_tensors_iteration, value_tensors_iteration)
        memory_deduplicated = self.drop_duplicates(memory)
        self.set_winratio_stats(memory_deduplicated)
        state_tensors_iteration, policy_tensors_iteration, q_value_tensors_iteration, value_tensors_iteration = memory_deduplicated

        states_path = f"{data_path}clean/states.pt"
        policy_path = f"{data_path}clean/policy.pt"
        q_values_path = f"{data_path}clean/q_values.pt"
        values_path = f"{data_path}clean/values.pt"

        states_path_sample = f"{data_path}states/0.zip"
        policy_path_sample = f"{data_path}policy/0.zip"
        q_values_path_sample = f"{data_path}q_values/0.zip"

        torch.save(state_tensors_iteration, states_path)
        torch.save(policy_tensors_iteration, policy_path)
        torch.save(q_value_tensors_iteration, q_values_path)
        torch.save(value_tensors_iteration, values_path)

        #adding to buffer
        for state, policy, q_value, value in zip(state_tensors_iteration, policy_tensors_iteration, q_value_tensors_iteration, value_tensors_iteration):
            exp = model.Experience(state, policy.unsqueeze(0), q_value, value)                      
            self.buffer.append(exp)
        
        self.writer.add_scalar('buffer length', len(self.buffer), self.iteration)
        custom_dataset_quant = model.CustomDataset(states_path_sample, policy_path_sample, q_values_path_sample, quant=True)
        self.quantization_dataloader = DataLoader(custom_dataset_quant, batch_size=self.batch_size_quantization, shuffle=False, drop_last=True)     
        

    def set_model_mode(self, mode):
        if mode == 'train':
            self.model.train()
        else:
            self.model.eval()

    def evaluate(self, net, data_loader):
        sum_value_loss = 0.0
        n_batches = 1

        for batch_states, targets in data_loader:

            value_targets = targets.to(self.model.device)
            _, out_value = net(batch_states.to(self.model.device))
            

            value_loss = F.mse_loss(out_value, value_targets)
            

            sum_value_loss += value_loss.item()
            n_batches += 1
        
        total_loss_epoch = sum_value_loss / n_batches

        return total_loss_epoch
    
    def export_to_jit(self):
        torch.save(self.model.state_dict(), f"{self.checkpoint_models}/model_{self.iteration}.pt")
        torch.save(self.optimizer.state_dict(), f"{self.checkpoint_models}/optimizer_{self.iteration}.pt")

        self.writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], self.iteration)

        model_path = f"C_project/models/model_{self.iteration}.ts"
        model_path2 = f"C_project/models/model_{self.iteration}.pt"

        if self.iteration == 0:
            example = torch.rand(1, 3, 6, 7).to(self.model.device)
            traced_script_module = torch.jit.trace(self.model, example)

            torch.jit.save(traced_script_module, model_path2)

            compile_spec = {"inputs": [torch_tensorrt.Input(min_shape=[1, 3, 6, 7],
                                                            opt_shape=[self.batch_size_quantization, 3, 6, 7],
                                                            max_shape=[2048, 3, 6, 7])],
                            "enabled_precisions": torch.half,
                            "workspace_size" : 1 << 22
                            }
            
            #model_copy = copy.deepcopy(self.model)
            #model_copy.half()
            trt_base = torch_tensorrt.compile(self.model, **compile_spec)
            print(f'Saving model path: {model_path}')
            torch.jit.save(trt_base, model_path)


        elif self.iteration > 0:
            with torch.no_grad():
                data = iter(self.quantization_dataloader)
                states, _ = next(data)
                jit_model = torch.jit.trace(self.model, states.to(self.model.device))
                torch.jit.save(jit_model, model_path2)

            model_jit = torch.jit.load(model_path2).eval()
            calibrator = torch_tensorrt.ptq.DataLoaderCalibrator(self.quantization_dataloader,
                                                                 use_cache=False,
                                                                 algo_type=torch_tensorrt.ptq.CalibrationAlgo.MINMAX_CALIBRATION,
                                                                 device=self.model.device)
            compile_spec = {
                    "inputs": [torch_tensorrt.Input(min_shape=[1, 3, 6, 7],
                                                    opt_shape=[self.batch_size_quantization, 3, 6, 7],
                                                    max_shape=[2048, 3, 6, 7])],
                    "enabled_precisions": torch.int8,
                    "calibrator": calibrator,
                    "truncate_long_and_double": True    
            }

            print("Compiling INT8 model...")
            trt_ptq = torch_tensorrt.compile(model_jit, **compile_spec)

            print("Bentchmarks...")
            loss_quantized = self.evaluate(trt_ptq, self.quantization_dataloader)
            loss_original = self.evaluate(model_jit, self.quantization_dataloader)
            print(f"Model loss: {loss_original:.4f}. Quantized model loss: {loss_quantized:.4f}")

            self.writer.add_scalar('diff loss quantized', loss_quantized - loss_original, self.iteration)

            print(f'Saving model path: {model_path}')
            torch.jit.save(trt_ptq, model_path)


        self.iteration += 1

        return model_path
    
    def drop_duplicates(self, memory):
        state_tensors, policy_tensors, q_value_tensors, value_tensors = memory
        _, inv, counts = torch.unique(state_tensors, dim=0, return_inverse=True, return_counts=True)
        idx_unique_states = torch.tensor([torch.where(inv == i)[0][0].item() for i, c, in enumerate(counts)])

        n_memory = state_tensors.shape[0]
        print(f"Memory before drop duplicates: {n_memory}")
        print(f"Memory after drop duplicates: {idx_unique_states.shape[0]}")

        state_tensors = state_tensors[idx_unique_states]
        policy_tensors = policy_tensors[idx_unique_states]
        value_tensors = value_tensors[idx_unique_states]
        q_value_tensors = q_value_tensors[idx_unique_states]

        avg_games = n_memory//alphazero_params.num_par_games
        print(f"Mean steps per match: {avg_games}")

        self.writer.add_scalar('memory length', n_memory, self.iteration)

        return (state_tensors, policy_tensors, q_value_tensors, value_tensors)
    

    def split_data(self, memory):
        state_tensors, policy_tensors, value_tensors, old_log_probs = memory

        indices = torch.randperm(state_tensors.size(0))

        #shuffle before split
        state_tensors = state_tensors[indices]
        policy_tensors = policy_tensors[indices]
        value_tensors = value_tensors[indices]
        old_log_probs = old_log_probs[indices]

        idx_limit = int(state_tensors.shape[0] * 0.8)

        state_tensors_train = state_tensors[:idx_limit]
        state_tensors_val = state_tensors[idx_limit:]

        policy_tensors_train = policy_tensors[:idx_limit]
        policy_tensors_val = policy_tensors[idx_limit:]

        value_tensors_train = value_tensors[:idx_limit]
        value_tensors_val = value_tensors[idx_limit:]

        old_log_probs_train = old_log_probs[:idx_limit]
        old_log_probs_val = old_log_probs[idx_limit:]

        memory_train = (state_tensors_train, policy_tensors_train, value_tensors_train, old_log_probs_train)
        memory_val = (state_tensors_val, policy_tensors_val, value_tensors_val, old_log_probs_val)

        return memory_train, memory_val
    
    def set_winratio_stats(self, memory):
        state_tensors, _, _, value_tensors = memory
        n_memory = state_tensors.shape[0]
        n_wins_x = 0 #wx
        n_wins_o = 0 #wo
        n_draws = 0 #d 

        results = []

        for i in range(n_memory):
            s = state_tensors[i,...].squeeze(0)
            v = value_tensors[i].item()
            player_plane = torch.sum(s[2, ...])

            if v == 1.0:
                #if (n_pieces_rival != n_pieces_ours):
                if (player_plane > 0):
                    n_wins_x += 1
                    results.append("wx")
                else:
                    n_wins_o += 1
                    results.append("wo")
            elif v == -1.0:
                #if (n_pieces_rival != n_pieces_ours):
                if (player_plane > 0):
                    n_wins_o += 1
                    results.append("wo")
                else:
                    n_wins_x += 1
                    results.append("wx")
            else:
                n_draws += 1
                results.append("d")

        #n_results = len(results)
        n_wins = n_wins_x + n_wins_o
        n_wins_x_pct = round(n_wins_x/n_wins, 2)
        n_wins_o_pct = round(n_wins_o/n_wins, 2)

        print(f"% wins X: {n_wins_x_pct} % wins O: {n_wins_o_pct}")

        self.writer.add_scalar('pct x winratio', n_wins_x_pct, self.iteration)
        self.writer.add_scalar('pct o winratio', n_wins_o_pct, self.iteration)

                
    def train(self):
        sum_loss = 0.0
        sum_value_loss = 0.0
        sum_policy_loss = 0.0
        n_batches = 1
        grad_max = 0.0
        grad_means = 0.0
        grad_count = 1

        approx_kl_divs = []

        n_exp_buffer = len(self.buffer)
        n_steps_epoch = int(n_exp_buffer/alphazero_params.batch_size)

        n_steps = min(n_steps_epoch*10, alphazero_params.max_steps_per_iteration)

        model_checkpoint = copy.deepcopy(self.model).to(device)

        for i in range(n_steps):

            batch = self.buffer.sample(alphazero_params.batch_size)
            state, policy_targets, value_targets, _ = batch

            out_policy, out_value = self.model(state)
            
            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value.squeeze(1), value_targets)

            loss = alphazero_params.coef_policy * policy_loss + alphazero_params.coef_value * value_loss

            #approx KL divergence based on stable baselines PPO
            with torch.no_grad():
                out_policy_prior, _ = model_checkpoint(state)
                old_log_prob = F.log_softmax(out_policy_prior, dim=1)

                log_prob = F.log_softmax(out_policy, dim=1)
                log_ratio = log_prob - old_log_prob
                k3 = (torch.exp(log_ratio) - 1) - log_ratio
                approx_kl_div = torch.mean(k3).cpu().numpy()
                approx_kl_divs.append(approx_kl_div)

            self.optimizer.zero_grad()
            loss.backward()

            # Clip grad norm
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), alphazero_params.max_grad_norm)

            self.optimizer.step()

            if self.iteration == 1:
                self.adjust_learning_rate_for_warmup(i, n_steps)
            
            sum_loss += loss.item()
            sum_value_loss += value_loss.item()
            sum_policy_loss += policy_loss.item()
            n_batches += 1

            grad_max = 0.0
            for p in self.model.parameters():
                grad_max = max(grad_max, p.grad.abs().max().item())
                grad_means += (p.grad ** 2).mean().sqrt().item()
                grad_count += 1
        
        if self.iteration > 1:
            self.scheduler.step()

        total_loss_epoch = sum_loss / n_batches
        
        self.writer.add_scalar("loss total train", total_loss_epoch, self.epoch_history)
        self.writer.add_scalar("loss value train", sum_value_loss / n_batches, self.epoch_history)
        self.writer.add_scalar("loss policy train", sum_policy_loss / n_batches, self.epoch_history)

        self.writer.add_scalar("grad_l2", grad_means / grad_count, self.epoch_history)
        self.writer.add_scalar("grad_max", grad_max, self.epoch_history)

        mean_kl_divs = np.mean(approx_kl_divs)
        print(f"Max kl_divs: {np.max(approx_kl_divs)}")
        print(f"Min kl_divs: {np.min(approx_kl_divs)}")
        print(f"Mean kl_divs: {mean_kl_divs}")
        print(f"Median kl_divs: {np.median(approx_kl_divs)}")
        self.writer.add_scalar("kl_div train", mean_kl_divs, self.epoch_history)

        self.epoch_history += 1

        return total_loss_epoch

device = torch.device("cuda")
net = model.ResNet(alphazero_params.resnet_num_blocks, alphazero_params.resnet_hidden, 6, 7, 7, device=device)

writer = SummaryWriter(comment="-" + alphazero_params.run_name)
optimizer = optim.SGD(net.parameters(), lr=alphazero_params.initial_lr, momentum=0.9, weight_decay=alphazero_params.weight_decay)

alphazero_trainer = AlphaZero(net,
                              writer,
                              optimizer)

alphazero_trainer.set_model_mode('eval')
_ = alphazero_trainer.export_to_jit()

@app.post("/train")
async def step(request: Request):
    body = await request.json()

    data_path = body['data_path']
    alphazero_trainer.add_data_to_buffer(data_path)
    alphazero_trainer.set_model_mode('eval')

    alphazero_trainer.set_model_mode('train')
    _ = alphazero_trainer.train()
    
    alphazero_trainer.set_model_mode('eval')
    model_path = alphazero_trainer.export_to_jit()    

    return {'model_path': model_path}

