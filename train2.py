import datetime
import numpy as np
import demon2 as d2
import matplotlib.pyplot as plt
import multiprocessing
current_date = datetime.datetime.now().strftime("%Y%m%d")

#Ising_Model_Simulator
L_x = 32
L_y = 32
number_of_sites = L_x * L_y
ising_jay = 1.0

#Demon class
number_of_inputs = 2
width = 4
width_final = 10
number_of_inputs = 2
number_of_outputs= 2


mc_sweeps = 1000

#demon_mutation class
tee_initial = 0.65
tee_final = 0.65
field_final = 1.0
field_initial = -1.0
trajectory_length = mc_sweeps * L_x * L_y
net_actions = 1000
net_step = trajectory_length / net_actions
sigma_mutate = 1.0
epsilon = 0.01
n_scale = 25

#trajectories instances
number_of_mutation = 5000 
number_of_trajectories = 50
k_m = 1.0
k_sigma = 1e-4

tee_critical = 1.0/(0.5*np.log(1.0+np.sqrt(2.0)))

#Parallel Computing
number_of_cores = 10

ising = d2.ising_simulation(L_x, L_y, ising_jay)
demon=d2.Demon(number_of_inputs, width, width_final, number_of_outputs)
amc = d2.demon_aMC(demon, tee_initial, field_initial, tee_final, field_final, trajectory_length, net_actions, sigma_mutate, epsilon, n_scale)

def action_trajectory_w_magnetization(tee_trajectory, field_trajectory, magnetization_trajectory, total_entprod, final_magnetization, check_interval = 50, cross_size = 0.05):
    plt.figure(figsize=(10, 6))
    
    scatter = plt.scatter(tee_trajectory, field_trajectory, c=magnetization_trajectory, cmap='coolwarm', vmin=-1, vmax=1)
    
    for i in range(0, len(tee_trajectory), check_interval):
        plt.text(tee_trajectory[i], field_trajectory[i], str(i), fontsize=9, color='black')

        x = tee_trajectory[i]
        y = field_trajectory[i]
        plt.plot([x - cross_size, x + cross_size], [y, y], color='black', lw=1)    
        plt.plot([x, x], [y - cross_size, y + cross_size], color='black', lw=1)
    
    plt.subplots_adjust(bottom=0.2)
        
    plt.title(f'Total EntProd: {total_entprod}, Final Mag: {final_magnetization}')   
    plt.xlabel('Tee Value')
    plt.ylabel('Field Value')
    plt.colorbar(scatter, label='Magnetization')
    plt.show()
    
def action_trajectory_w_magnetization_install(success, tee_trajectory, field_trajectory, magnetization_trajectory, total_entprod, final_magnetization, check_interval = 50, cross_size = 0.05):
    plt.figure(figsize=(10, 6))
    
    scatter = plt.scatter(tee_trajectory, field_trajectory, c=magnetization_trajectory, cmap='coolwarm', vmin=-1, vmax=1)
    
    for i in range(0, len(tee_trajectory), check_interval):
        plt.text(tee_trajectory[i], field_trajectory[i], str(i), fontsize=9, color='black')

        x = tee_trajectory[i]
        y = field_trajectory[i]
        plt.plot([x - cross_size, x + cross_size], [y, y], color='black', lw=1)    
        plt.plot([x, x], [y - cross_size, y + cross_size], color='black', lw=1)
    
    plt.subplots_adjust(bottom=0.2)
        
    plt.title(f'#{success}, Total EntProd: {total_entprod}, Final Mag: {final_magnetization}')   
    plt.xlabel('Tee Value')
    plt.ylabel('Field Value')
    plt.colorbar(scatter, label='Magnetization')
    file_path = f'/home/chad0723/Demon Project/demon-whitelam/data_plot/{success}_mag_aMC_sig_{sigma_mutate}_0217.png'
    plt.savefig(file_path)


    
def action_trajectory_w_entprod(tee_trajectory, field_trajectory, entprod_trajectory, total_entprod, final_magnetization, check_interval = 50, cross_size = 0.05):
    plt.figure(figsize=(10, 6))
    
    scatter = plt.scatter(tee_trajectory, field_trajectory, c=entprod_trajectory, cmap='viridis')
    
    for i in range(0, len(tee_trajectory), check_interval):
        plt.text(tee_trajectory[i], field_trajectory[i], str(i), fontsize=9, color='black')

        x = tee_trajectory[i]
        y = field_trajectory[i]
        plt.plot([x - cross_size, x + cross_size], [y, y], color='black', lw=1)    
        plt.plot([x, x], [y - cross_size, y + cross_size], color='black', lw=1)
        
    plt.subplots_adjust(bottom=0.2)
        
    plt.title(f'Total EntProd: {total_entprod}, Final Mag: {final_magnetization}')   
    plt.xlabel('Tee Value')
    plt.ylabel('Field Value')
    plt.colorbar(scatter, label='Entropy production')
    plt.show()

def action_trajectory_w_entprod_install(success, tee_trajectory, field_trajectory, entprod_trajectory, total_entprod, final_magnetization, check_interval = 50, cross_size = 0.05):
    plt.figure(figsize=(10, 6))
    
    scatter = plt.scatter(tee_trajectory, field_trajectory, c=entprod_trajectory, cmap='viridis')
    
    for i in range(0, len(tee_trajectory), check_interval):
        plt.text(tee_trajectory[i], field_trajectory[i], str(i), fontsize=9, color='black')

        x = tee_trajectory[i]
        y = field_trajectory[i]
        plt.plot([x - cross_size, x + cross_size], [y, y], color='black', lw=1)    
        plt.plot([x, x], [y - cross_size, y + cross_size], color='black', lw=1)
    
    plt.subplots_adjust(bottom=0.2)
        
    plt.title(f'#{success}, Total EntProd: {total_entprod}, Final Mag: {final_magnetization}')   
    plt.xlabel('Tee Value')
    plt.ylabel('Field Value')
    plt.colorbar(scatter, label='Entropy production')
    file_path = f'/home/chad0723/Demon Project/demon-whitelam/data_plot/{success}_entprod_aMC_sig_{sigma_mutate}_0217.png'
    plt.savefig(file_path)

    
def run_trajectory():
    tau = 0.0
    total_entprod = 0.0
    magnetization = 0.0
    tee = 0.0
    field = 0.0
    
    amc.calculate_shear()
    
    #set total spin state to -1
    ising.initialize_spin()
    magnetization = - 1.0
    
    tee, field = amc.run_net(tau, magnetization)
    
    e1 = ising.lattice_nrg(field)
    
    for i in range(trajectory_length):
        if (i%net_step == 0):
            tee, field = amc.run_net(tau, magnetization)
        delta_mag, entprod, _, _ = ising.mc_step(tee, field)
        
        magnetization += delta_mag
        total_entprod += entprod
        
        tau += 1.0 / (1.0 * trajectory_length)
        
    tee = tee_final
    field = field_final
    e2 = ising.lattice_nrg(field)
    total_entprod += (e2-e1)/tee_final
    
    return total_entprod, magnetization

def run_trajectory_test():
    tau = 0.0
    total_entprod = 0.0
    magnetization = 0.0
    delta_mag = 0.0
    entprod = 0.0
    tee = 0.0
    field = 0.0
    
    tee_trajectory = [0 for _ in range(net_actions)]
    field_trajectory = [0 for _ in range(net_actions)]
    magnetization_trajectory = [0 for _ in range(net_actions)]
    entprod_trajectory = [0 for _ in range(net_actions)]
    
    amc.calculate_shear()
    
    #set total spin state to -1
    ising.initialize_spin()
    magnetization = - 1.0
    
    tee, field = amc.run_net(tau, magnetization)
    
    e1 = ising.lattice_nrg(field)
    
    for i in range(trajectory_length):
        if (i%net_step == 0):
            tee_trajectory[int(i//net_step)] = tee
            field_trajectory[int(i//net_step)] = field
            magnetization_trajectory[int(i//net_step)] = magnetization
            entprod_trajectory[int(i//net_step)] = entprod
            print(f'{i/net_step}th action')
            print('Input tee: ', tee)
            print('Input field: ', field)
            print('Input magnetization:', magnetization)
            tee, field = amc.run_net(tau, magnetization)
            
        delta_mag, entprod, _, _ = ising.mc_step(tee, field)
        
        magnetization += delta_mag
        total_entprod += entprod
        
        tau += 1.0 / (1.0 * trajectory_length)
        
    tee = tee_final
    field = field_final
    e2 = ising.lattice_nrg(field)
    total_entprod += (e2-e1)/tee_final
    
    return total_entprod, magnetization, tee_trajectory, field_trajectory, magnetization_trajectory, entprod_trajectory
        
def run_trajectory_average():
    
    mean_mag = 0.0
    mean_ep = 0.0
    
    for _ in range(number_of_trajectories):
        total_entprod, magnetization = run_trajectory()
        mean_mag += magnetization
        mean_ep += total_entprod
        
    mean_mag /= number_of_trajectories
    mean_ep /= number_of_trajectories
        
    order_param = k_m * np.abs(mean_mag - 1.0) + k_sigma * mean_ep
    
    return order_param, mean_mag, mean_ep

def worker(_):
    return run_trajectory()

def run_trajectory_average_parallel():
    mean_mag = 0.0
    mean_ep = 0.0

    for _ in range(int(number_of_trajectories/number_of_cores)):
        with multiprocessing.Pool() as pool:
            results = pool.map(worker, range(number_of_cores))
        
        for total_entprod, magnetization in results:
            mean_mag += magnetization
            mean_ep += total_entprod

    mean_mag /= number_of_trajectories
    mean_ep /= number_of_trajectories

    order_param = k_m * np.abs(mean_mag - 1.0) + k_sigma * mean_ep

    return order_param, mean_mag, mean_ep

def train():
    order_param_trajectory = [0 for _ in range(number_of_mutation)]
    final_mag_trajectory = [0 for _ in range(number_of_mutation)]
    mean_ep_trajectory = [0 for _ in range(number_of_mutation)]
    
    success=0
    amc.initialize_net()
    phi, _, _= run_trajectory_average()
    
    
    for generation in range(number_of_mutation):
        print('Generation #', generation)
        amc.store_net()
        amc.mutate_net()
        
        order_param, mean_mag, mean_ep = run_trajectory_average()
        order_param_trajectory[generation] = order_param
        final_mag_trajectory[generation] = mean_mag
        mean_ep_trajectory[generation] = mean_ep
        
        if(order_param <= phi):
            success += 1
            amc.q_ok = 1
            phi = order_param
            print(f'#{success} evolution complete')
            
        else:
            amc.q_ok = 0
            amc.restore_net()
            
        amc.scale_mutations()
        
    return order_param_trajectory, final_mag_trajectory, mean_ep_trajectory

def train_parallel():
    order_param_trajectory = [0 for _ in range(number_of_mutation)]
    final_mag_trajectory = [0 for _ in range(number_of_mutation)]
    mean_ep_trajectory = [0 for _ in range(number_of_mutation)]
    
    success=0
    amc.initialize_net()
    phi, _, _= run_trajectory_average_parallel()
    
    
    for generation in range(number_of_mutation):
        print('Generation #', generation)
        amc.store_net()
        amc.mutate_net()
        
        order_param, mean_mag, mean_ep = run_trajectory_average_parallel()
        order_param_trajectory[generation] = order_param
        final_mag_trajectory[generation] = mean_mag
        mean_ep_trajectory[generation] = mean_ep
        
        if(order_param <= phi):
            success += 1
            amc.q_ok = 1
            phi = order_param
            print(f'#{success} evolution complete')
            amc.save_trained_demon(success)
            total_entprod, magnetization, tee_trajectory, field_trajectory, magnetization_trajectory, entprod_trajectory = run_trajectory_test()
            action_trajectory_w_magnetization_install(success, tee_trajectory, field_trajectory, magnetization_trajectory, total_entprod, magnetization, 50, 0.05)
            action_trajectory_w_entprod_install(success, tee_trajectory, field_trajectory, entprod_trajectory, total_entprod, magnetization, 50, 0.05)
        else:
            amc.q_ok = 0
            amc.restore_net()
            
        amc.scale_mutations()
        
    return order_param_trajectory, final_mag_trajectory, mean_ep_trajectory
        
def train_sandbox():
    order_param_trajectory = [0 for _ in range(number_of_mutation)]
    final_mag_trajectory = [0 for _ in range(number_of_mutation)]
    mean_ep_trajectory = [0 for _ in range(number_of_mutation)]
    
    success=0
    amc.initialize_net()
    phi, _, _= run_trajectory_average()
    
    
    for generation in range(number_of_mutation):
        print('Generation #', generation)
        amc.store_net()
        amc.mutate_net()
        
        order_param, mean_mag, mean_ep = run_trajectory_average()
        order_param_trajectory[generation] = order_param
        final_mag_trajectory[generation] = mean_mag
        mean_ep_trajectory[generation] = mean_ep
        
        if(order_param <= phi):
            success += 1
            amc.q_ok = 1
            phi = order_param
            print(f'#{success} evolution complete')
            total_entprod, magnetization, tee_trajectory, field_trajectory, magnetization_trajectory, entprod_trajectory = run_trajectory_test()
            action_trajectory_w_magnetization(tee_trajectory, field_trajectory, magnetization_trajectory, total_entprod, magnetization, 50, 0.05)
            action_trajectory_w_entprod(tee_trajectory, field_trajectory, entprod_trajectory, total_entprod, magnetization, 50, 0.05)
            
        else:
            amc.q_ok = 0
            amc.restore_net()
            
        amc.scale_mutations()
        
    return order_param_trajectory, final_mag_trajectory, mean_ep_trajectory

def run_plot():
    total_entprod, magnetization, tee_trajectory, field_trajectory, magnetization_trajectory, entprod_trajectory = run_trajectory_test()
    action_trajectory_w_magnetization(tee_trajectory, field_trajectory, magnetization_trajectory, total_entprod, magnetization, 50, 0.05)
    action_trajectory_w_entprod(tee_trajectory, field_trajectory, entprod_trajectory, total_entprod, magnetization, 50, 0.05)
    
def train_sandbox_parallel():
    order_param_trajectory = [0 for _ in range(number_of_mutation)]
    final_mag_trajectory = [0 for _ in range(number_of_mutation)]
    mean_ep_trajectory = [0 for _ in range(number_of_mutation)]
    
    success=0
    amc.initialize_net()
    phi, _, _= run_trajectory_average()
    
    
    for generation in range(number_of_mutation):
        print('Generation #', generation)
        amc.store_net()
        amc.mutate_net()
        
        order_param, mean_mag, mean_ep = run_trajectory_average_parallel()
        order_param_trajectory[generation] = order_param
        final_mag_trajectory[generation] = mean_mag
        mean_ep_trajectory[generation] = mean_ep
        
        if(order_param <= phi):
            success += 1
            amc.q_ok = 1
            phi = order_param
            print(f'#{success} evolution complete')
            run_plot()
            
        else:
            amc.q_ok = 0
            amc.restore_net()
            
        amc.scale_mutations()
        
    return order_param_trajectory, final_mag_trajectory, mean_ep_trajectory       


    
    
    
    