import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt

class Demon(nn.Module):
    def __init__(self, number_of_inputs, width, width_final, number_of_outputs):
        super(Demon, self).__init__()
        
        #Input layer
        self.fnn1 = nn.Linear(number_of_inputs, width)
        self.norm1 = LayerNorm()
        
        #Hidden layers
        self.fnn2 = nn.Linear(width, width)
        self.norm2 = LayerNorm()
        
        self.fnn3 = nn.Linear(width, width)
        self.norm3 = LayerNorm()
        
        self.fnn4 = nn.Linear(width, width)
        self.norm4 = LayerNorm()
        
        #Final layer
        self.fnn_final = nn.Linear(width, width_final)
        self.norm_final = LayerNorm()
        
        #Output layer
        self.fnn_output = nn.Linear(width_final, number_of_outputs)
        self.norm_output = LayerNorm()
        
    def forward(self, x):
        
        x = torch.tanh(self.norm1(self.fnn1(x)))
        
        x = torch.tanh(self.norm2(self.fnn2(x)))
        x = torch.tanh(self.norm3(self.fnn3(x)))
        x = torch.tanh(self.norm4(self.fnn4(x)))
        
        x = torch.tanh(self.norm_final(self.fnn_final(x)))
        
        output = self.fnn_output(x)
        
        return output       
          
class LayerNorm(nn.Module):
    def __init__(self, eps = 1e-4):
        super(LayerNorm, self).__init__()
        self.eps = eps
        
    def forward(self, x):
        mu_x = x.mean(dim = -1, keepdim = True)
        var_x = x.var(dim = -1, keepdim = True, unbiased = False)
        
        #Normalize
        x = (x - mu_x) / (torch.sqrt(var_x)+self.eps)

        return x

class ising_simulation:
    def __init__(self, L_x, L_y, ising_jay):
        self.L_x = L_x
        self.L_y = L_y
        self.number_of_sites = L_x * L_y
        
        self.neighbors = np.zeros((self.number_of_sites, 4), dtype=int)
        for i in range(self.number_of_sites):
            x1 = i % self.L_x
            y1 = i // self.L_x  
            
            # Neighbors considering periodic boundary conditions
            #Left
            if (x1==0):
                self.neighbors[i, 0]= i + self.L_x - 1
            else:
                self.neighbors[i, 0]= i - 1
            #Right    
            if (x1 == L_x-1):
                self.neighbors[i, 1] = i - (self.L_x - 1)
            else:
                self.neighbors[i, 1] = i + 1
            #Down
            if (y1 == 0):
                self.neighbors[i, 2] = i + self.L_x * (self.L_x - 1)
            else:
                self.neighbors[i, 2] = i - self.L_x
                
            #Up
            if (y1==L_y - 1):
                self.neighbors[i, 3] = i - self.L_x * (self.L_x - 1)
            else:
                self.neighbors[i, 3] = i + self.L_x
                
                
        self.ising_jay = ising_jay
        self.spin = (np.full(self.number_of_sites, -1))
    
    def initialize_spin(self):
        for i in range(self.number_of_sites):
            self.spin[i]=-1  

    def delta_nrg(self, chosen_site, field):
        de = 0.0
        s1 = 1.0 * self.spin[chosen_site]
        
        for i in range(4):
            s2 = 1.0 * self.spin[self.neighbors[chosen_site][i]]
            de += 2.0 * self.ising_jay * s1 * s2
            
        de += 2.0 * field * s1
        
        return de
    
    def lattice_nrg(self, field):
        nrg1 = 0.0
        
        for i in range(self.number_of_sites):
            s1 = self.spin[i]
            nrg1 -= field * s1
            
            for j in range(4):
                n1 = self.neighbors[i][j]
                s2 = self.spin[n1]
                nrg1 -= 0.5 * self.ising_jay * s1 * s2
                
        return nrg1
        
    def mc_step(self, tee, field):
        
        chosen_site = np.random.randint(0, self.number_of_sites)
        
        delta_mag = 0.0
        entprod = 0.0
        de = self.delta_nrg(chosen_site, field)
        prob1 = random.random()
        
        threshold = 100
        exponent = (de / tee)
        if exponent > threshold:
            glauber = 0
        elif exponent < -threshold:
            glauber = 1
        else:
            glauber = 1/(1+np.exp(exponent))
        
        if (prob1 < glauber):
            entprod = - de/tee
            delta_mag = - (2.0 * self.spin[chosen_site] / (1.0 * self.number_of_sites))
            self.spin[chosen_site] = -self.spin[chosen_site]
            
        return delta_mag, entprod, de, glauber
    
    def plot_lattice(self, step):
        lattice = self.spin.reshape(self.L_x, self.L_y)
        plt.imshow(lattice, cmap = 'coolwarm', interpolation = 'nearest', vmin = -1, vmax = 1)
        plt.title(f'Spin Configuration at Step {step}')
        plt.colorbar(label='Spin')
        plt.show()
            
class demon_aMC:
    def __init__(self, model, tee_initial, field_initial, tee_final, field_final, trajectory_length, net_actions, sigma_mutate, epsilon, n_scale):
        self.model = model
        self.tee_initial = tee_initial
        self.field_initial = field_initial
        self.tee_final = tee_final
        self.field_final = field_final
        self.net_actions = net_actions
        self.net_step = trajectory_length / net_actions
        self.shear = [[0.0, 0.0], [0.0, 0.0]]
        self.q_shear = 0
        self.epsilon = epsilon
        self.n_scale = n_scale
        
        self.mutation = {}
        self.mean_mutation = {}
        self.parameter_holder = {}
        
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                self.mean_mutation[name] = torch.zeros_like(param)
                self.mutation[name] = torch.zeros_like(param)
        
        self.sigma_mutate = sigma_mutate
        self.q_ok = 1
        self.consec_rejections = 0
        self.n_reset = 0
        
    def counter_parameters(self):
        total_parameter = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(total_parameter)
    
    def print_demon_param(self):
        for name, param in self.model.named_parameters():
            print(f"{name}: shape {param.size()}")
            print(param.data[:5])
        
    def initialize_net(self):
        for param in self.model.parameters():
            nn.init.zeros_(param)
            
    def run_net(self, tau, magnetization):
        tee = 0.0
        field = 0.0
        
        model_input = torch.tensor([tau, magnetization], dtype = torch.float32)
        model_output = self.model(model_input)
            
        if (self.q_shear==0):
            tee = model_output[0].item()
        else:
            tee = model_output[0].item() + (1.0 - tau) * self.shear[0][0] + tau * self.shear[0][1]
            if (tee < 1e-3): tee = 1e-3
                
        if (self.q_shear==0):
            field = model_output[1].item()
        else:
            field = model_output[1].item() + (1.0 - tau) * self.shear[1][0] + tau * self.shear[1][1] 
                
        return tee, field
                
    def store_net(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                self.parameter_holder[name] = param.clone()
    
    def restore_net(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.parameter_holder:
                    param.data.copy_(self.parameter_holder[name].data)
    
    def gauss_rv(self, shape):
        return torch.randn(shape) * self.sigma_mutate  
    
    #Gaussian random variables using the Box-Muller transform
    # def gauss_rv(self, shape):
    #      r1 = torch.rand(shape)
    #      r2 = torch.rand(shape)
    #      g1 = torch.sqrt((-2.0 * torch.log(r1))) * self.sigma_mutate * torch.cos(2*torch.pi*r2)
    #      return g1
                    
    def mutate_net(self):
        for name, param in self.model.named_parameters():
            self.mutation[name] = self.mean_mutation[name] + self.gauss_rv(param.shape)
        for name, param in self.model.named_parameters():
            param.data = param.data + self.mutation[name]
        
    def scale_mutations(self):
        if (self.q_ok == 1):
            self.consec_rejections = 0
            for name in self.mean_mutation:
                self.mean_mutation[name] += self.epsilon * (self.mutation[name] - self.mean_mutation[name])
        else: 
            self.consec_rejections += 1
            if (self.consec_rejections >= self.n_scale):
                self.consec_rejections = 0
                self.n_reset += 1
                self.sigma_mutate *= 0.95
                for name in self.mean_mutation:
                    self.mean_mutation[name].zero_()
                         
    def calculate_shear(self):
        
        self.q_shear = 0
        
        #Run demon at initial condition
        tee_initial_raw, field_initial_raw = self.run_net(0.0, -1.0)
        
        self.shear[0][0] = self.tee_initial -tee_initial_raw
        self.shear[1][0] = self.field_initial - field_initial_raw
        
        #Run demon at final condition
        tee_final_raw, field_final_raw = self.run_net(1.0, 1.0)
        
        self.shear[0][1] = self.tee_final -tee_final_raw
        self.shear[1][1] = self.field_final - field_final_raw    
        
        self.q_shear = 1
    
    def save_trained_demon(self, i):
        filepath=f'/home/chad0723/Demon Project/demon-whitelam/trained_demons/aMC_0217/aMC_sig_0.1_{i}'
        torch.save(self.model.state_dict(), filepath)
        print(f"Model parameters saved to {filepath}")
    
    def load_trained_demon(self, filepath):
        self.model.load_state_dict(torch.load(filepath))
        self.model.eval()
        print(f"Model parameters loaded from {filepath}")
        

class demon_GA:
    def __init__(self, model, tee_initial, field_initial, tee_final, field_final, trajectory_length, net_actions, sigma_mutate, epsilon, n_scale):
        self.model = model
        self.tee_initial = tee_initial
        self.field_initial = field_initial
        self.tee_final = tee_final
        self.field_final = field_final
        self.net_actions = net_actions
        self.net_step = trajectory_length / net_actions
        self.shear = [[0.0, 0.0], [0.0, 0.0]]
        self.q_shear = 0
        self.epsilon = epsilon
        self.n_scale = n_scale
        
        self.mutation = {}
        self.mean_mutation = {}
        self.parameter_holder = {}
        
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                self.mean_mutation[name] = torch.zeros_like(param)
                self.mutation[name] = torch.zeros_like(param)
        
        self.sigma_mutate = sigma_mutate
        self.q_ok = 1
        self.consec_rejections = 0
        self.n_reset = 0
        
    def counter_parameters(self):
        total_parameter = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(total_parameter)
    
    def print_demon_param(self):
        for name, param in self.model.named_parameters():
            print(f"{name}: shape {param.size()}")
            print(param.data[:5])
        
    def initialize_net(self):
        for param in self.model.parameters():
            nn.init.zeros_(param)
            
    def run_net(self, tau, magnetization):
        tee = 0.0
        field = 0.0
        
        model_input = torch.tensor([tau, magnetization], dtype = torch.float32)
        model_output = self.model(model_input)
            
        if (self.q_shear==0):
            tee = model_output[0].item()
        else:
            tee = model_output[0].item() + (1.0 - tau) * self.shear[0][0] + tau * self.shear[0][1]
            if (tee < 1e-3): tee = 1e-3
                
        if (self.q_shear==0):
            field = model_output[1].item()
        else:
            field = model_output[1].item() + (1.0 - tau) * self.shear[1][0] + tau * self.shear[1][1] 
                
        return tee, field
                
    def store_net(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                self.parameter_holder[name] = param.clone()
    
    def restore_net(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.parameter_holder:
                    param.data.copy_(self.parameter_holder[name].data)
    
    def gauss_rv(self, shape):
        return torch.randn(shape) * self.sigma_mutate  
    
    #Gaussian random variables using the Box-Muller transform
    # def gauss_rv(self, shape):
    #      r1 = torch.rand(shape)
    #      r2 = torch.rand(shape)
    #      g1 = torch.sqrt((-2.0 * torch.log(r1))) * self.sigma_mutate * torch.cos(2*torch.pi*r2)
    #      return g1
                    
    def mutate_net(self):
        for name, param in self.model.named_parameters():
            self.mutation[name] = self.mean_mutation[name] + self.gauss_rv(param.shape)
        for name, param in self.model.named_parameters():
            param.data = param.data + self.mutation[name]
        
    def scale_mutations(self):
        if (self.q_ok == 1):
            self.consec_rejections = 0
            for name in self.mean_mutation:
                self.mean_mutation[name] += self.epsilon * (self.mutation[name] - self.mean_mutation[name])
        else: 
            self.consec_rejections += 1
            if (self.consec_rejections >= self.n_scale):
                self.consec_rejections = 0
                self.n_reset += 1
                self.sigma_mutate *= 0.95
                for name in self.mean_mutation:
                    self.mean_mutation[name].zero_()
                         
    def calculate_shear(self):
        
        self.q_shear = 0
        
        #Run demon at initial condition
        tee_initial_raw, field_initial_raw = self.run_net(0.0, -1.0)
        
        self.shear[0][0] = self.tee_initial -tee_initial_raw
        self.shear[1][0] = self.field_initial - field_initial_raw
        
        #Run demon at final condition
        tee_final_raw, field_final_raw = self.run_net(1.0, 1.0)
        
        self.shear[0][1] = self.tee_final -tee_final_raw
        self.shear[1][1] = self.field_final - field_final_raw    
        
        self.q_shear = 1