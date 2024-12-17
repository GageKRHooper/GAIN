"""
Created on September 10, 2024

@author: GageKRHooper
"""

import numpy as np
import matplotlib.pyplot as plt

# Define initial parameters
grid_size = (2, 2)  # Example grid size
Neurons = grid_size[0] * grid_size[1]  # Total number of neurons
SimTime = 250  # Total Simulation Time in ms
dt = .01  # Time Step
time = np.arange(0, SimTime, dt)  # Time array
T = 10  # period
epochs = 1
spike_times = []

TrainI = [[0, 0], [1, 0], [0, 1], [1, 1]]
TrainO = [[0], [1], [1], [0]]

upper_bound = 255

num_interations = 1

# Initialize variables
absolute_refractory_period = (SimTime * dt) * 2.0  # in ms
sigma = np.random.uniform(-10, 15, Neurons).astype(np.float32)
sum_weights = np.zeros(Neurons).astype(np.float32)

def sigmoid(x, k):
    return (1/(1 + np.exp((-x)*k)))

def reset_variables():
    # Vars
    V = np.zeros((Neurons, len(time)), dtype=np.float32)  # Membrane potentials for each neuron
    u = np.zeros((Neurons, len(time)), dtype=np.float32)  # Recovery variables
    I = np.zeros((Neurons, len(time)), dtype=np.float32)  # Input from Synapses
    Input_Current = np.zeros((Neurons, len(time)), dtype=np.float32)  # Input Current from User

    # Initialize neuron properties
    resting_current = -np.random.uniform(65, 70, Neurons).astype(np.float32)
    max_mV = np.random.uniform(30, 40, Neurons).astype(np.float32)

    a_values = np.zeros((Neurons, len(time)), dtype=np.float32)
    b_values = np.zeros((Neurons, len(time)), dtype=np.float32)

    # STP parameters
    Ut = np.zeros((Neurons, len(time)), dtype=np.float32)  # STP
    Rt = np.zeros((Neurons, len(time)), dtype=np.float32)
    T_f = np.random.uniform(10, 100, Neurons).astype(np.float32)  # Time constant for facilitation (ms)
    T_d = np.random.uniform(50, 300, Neurons).astype(np.float32)  # Time constant for depression (ms)

    # Other variables
    dV = np.zeros((Neurons, len(time)), dtype=np.float32)  # Change in Membrane potentials
    du = np.zeros((Neurons, len(time)), dtype=np.float32)  # Change in Recovery variables
    W_SYN = np.random.uniform(-.1, .1, (Neurons, Neurons)).astype(np.float32)  # Synaptic weights

    # Initial conditions for other parameters
    R_0 = np.random.uniform(0, 0.2, Neurons).astype(np.float32)
    u_0 = np.random.uniform(0, 0.2, Neurons).astype(np.float32)
    W_0 = np.random.uniform(-.01, .05, (Neurons, Neurons)).astype(np.float32)
    a_0 = np.random.uniform(0.02, 0.1, Neurons).astype(np.float32)
    b_0 = np.random.uniform(0.2, 0.5, Neurons).astype(np.float32)
    o = np.random.uniform(0, .1, Neurons).astype(np.float32)
    y = np.random.uniform(0, .1, Neurons).astype(np.float32)
    U = .1

    T_p = np.random.uniform(10, 40, Neurons).astype(np.float32)
    T_m = np.random.uniform(10, 40, Neurons).astype(np.float32)  # ms avg 20
    A_p = np.random.uniform(0.001, 0.1, Neurons).astype(np.float32)
    A_m = np.random.uniform(0.001, 0.1, Neurons).astype(np.float32)
    
    n_p = np.random.uniform(0.001, 0.01, Neurons).astype(np.float32)
    n_m = np.random.uniform(0.001, 0.01, Neurons).astype(np.float32)
    
    stpr = np.random.uniform(0.9, 1.0, Neurons).astype(np.float32)  # Initialize stpr with values close to 1 (indicating little depression at start)
    stpu = np.random.uniform(0.1, 0.5, Neurons).astype(np.float32)  # Initialize stpu with values that can be scaled

    return (V, u, I, Input_Current, resting_current, max_mV, a_values, b_values,
            Ut, Rt, T_f, T_d, dV, du, W_SYN, R_0, u_0, W_0, a_0, b_0, o, y, U, T_p, T_m, A_p, A_m, n_p, n_m, stpr, stpu)


def get_neighbors(index, rows, cols):
    neighbors = []
    row, col = divmod(index, cols)

    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:  # Skip the neuron itself
                continue
            r, c = row + dr, col + dc
            if 0 <= r < rows and 0 <= c < cols:  # Check if within bounds
                neighbors.append(r * cols + c)
    return neighbors

def print_neuron_variables(neuron_index, currentT, **variables):
    print(f"\nVariables for Neuron {neuron_index} at time {currentT/100}:")
    for var_name, var_data in variables.items():
        try:
            if isinstance(var_data, np.ndarray):
                if var_data.ndim == 2:
                    # For 2D arrays, access neuron and time indices
                    print(f"{var_name}[{neuron_index}, {currentT-1}]: {var_data[neuron_index, currentT-1]}")
                elif var_data.ndim == 1:
                    # For 1D arrays, access neuron index
                    print(f"{var_name}[{neuron_index}]: {var_data[neuron_index]}")
                else:
                    print(f"{var_name}: Data shape {var_data.shape} not handled.")
            else:
                # Handle non-array variables (e.g., scalars or lists)
                print(f"{var_name}: {var_data}")
        except IndexError:
            print(f"{var_name}: IndexError encountered. Data shape {var_data.shape if isinstance(var_data, np.ndarray) else 'N/A'}.")
        except Exception as e:
            print(f"{var_name}: Error encountered - {str(e)}")


def update_weights(last_spike_time, current_time, V, dW, W_0, A_p, A_m, T_p, T_m, T_d, T_f, R_0, u_0, Neurons):
    for neuron in range(Neurons):
        if last_spike_time[neuron] != -np.inf:  # Only update if the neuron spiked before
            for pre_neuron in range(Neurons):
                if pre_neuron != neuron:
                    delta_t = current_time - last_spike_time[pre_neuron]
                    
                    # Calculate weight change using STDP
                    if delta_t > 0:
                        weight_change = A_p * np.exp(-delta_t / T_p[neuron])
                        W_SYN[pre_neuron, neuron] += weight_change[neuron]  # Update weight directly
                        dW[pre_neuron, neuron] = weight_change[neuron]  # Store weight change for debugging
                        
                    elif delta_t < 0:
                        weight_change = -A_m * np.exp(delta_t / T_m[neuron])
                        W_SYN[pre_neuron, neuron] += weight_change[neuron]   # Update weight directly
                        dW[pre_neuron, neuron] = weight_change[neuron]   # Store weight change for debugging

    return dW  # Return the dW for debugging if necessary

def update_stp(last_spike_time, current_time, stpu, stpr, T_f, T_d, dt, Neurons):
    # Update STP for facilitation (stpu) and depression (stpr) based on time since last spike
    for neuron in range(Neurons):
        time_since_last_spike = current_time - last_spike_time[neuron]
        
        # Facilitation dynamics (increases with time)
        if time_since_last_spike < T_f[neuron] and not time_since_last_spike == np.inf:
            stpu[neuron] += (1 - stpu[neuron]) * (1 - np.exp(-time_since_last_spike / T_f[neuron]))  # Facilitation
        else:
            stpu[neuron] *= np.exp(-time_since_last_spike / T_f[neuron])  # Decay
        
        # Depression dynamics (decays with time)
        if time_since_last_spike < T_d[neuron] and not time_since_last_spike == np.inf:
            stpr[neuron] -= stpr[neuron] * (1 - np.exp(-time_since_last_spike / T_d[neuron]))  # Depression
        else:
            stpr[neuron] *= np.exp(-time_since_last_spike / T_d[neuron])  # Decay

    return stpu, stpr


def sum_neighboring_weights(W_SYN, grid_size, Neurons):
    row_size, col_size = grid_size
    sum_weights = np.zeros(Neurons, dtype=np.float32)

    for neuron in range(Neurons):
        neighbors = get_neighbors(neuron, row_size, col_size)
        
        # Sum weights from the neighbors
        neighbor_weights_sum = 0
        for neighbor in neighbors:
            neighbor_weights_sum += W_SYN[neuron, neighbor]
        
        # Store the sum of neighbor weights for the current neuron
        sum_weights[neuron] = neighbor_weights_sum
    
    return sum_weights

def binary_to_current(binary_input, min_current=5.0, max_current=20.0):
    # Convert the binary input to a decimal number
    decimal_value = int("".join(str(bit) for bit in binary_input), 2)
    
    # Scale the current based on the decimal value
    # Map the decimal range [0, 255] to the current range [min_current, max_current]
    current_value = min_current + (decimal_value / (np.max(TrainI))) * (max_current - min_current)
    
    return current_value

def normalize_spike_count(spike_times, max_spikes=255):
    # Count spikes
    spike_count = len(spike_times)

    # Normalize the count (scale it between 0 and 255)
    normalized_value = int((spike_count / max_spikes) * upper_bound)
    normalized_value = np.clip(normalized_value, 0, upper_bound)  # Ensure the value stays within 0-255
    return np.float64(normalized_value / upper_bound)

def run_simulation(iteration, V, u, I, Input_Current, resting_current, max_mV, a_values, b_values,
                   Ut, Rt, T_f, T_d, dV, du, W_SYN, R_0, u_0, W_0, a_0, b_0, o, y, U, T_p, T_m, A_p, A_m, n_p, n_m, stpr, stpu):

    last_spike_time = np.full(Neurons, -np.inf, dtype=np.float32)  # Track last spike time for each neuron
    spike_times_per_neuron = [[] for _ in range(Neurons)]  # List to store spike times for each neuron
    for t in range(1, len(time)):
        percent = 10
        if (t / SimTime) % percent == 0:
            print(f'Percent Complete: {t / SimTime}')
        
        sigma[:] = 0 #np.random.uniform(-10, 25) + logistic_map(time[t] / SimTime)
        
        #Input_Current[:, t] = ranIn[:]
        Input_Current[:, t] = 20
        # Calculate the sum of the weights for each neuron’s neighbors
        sum_weights[:] = sum_neighboring_weights(W_SYN[:], grid_size, Neurons)
        
        # Check for broken neurons
        if np.any(V <= -150) or np.any(np.isnan(V)) or np.any(V == -np.inf):
            print("Variables Exploded, breaking...")
            print("Broken Neurons:")
        
            # Find indices of broken neurons
            broken_neurons = np.where((V[:] <= -150) | np.isnan(V[:]) | (V[:] == -np.inf))[0]
        
            for neuron in broken_neurons:
                print(f"/n Neuron {neuron} is broken.")
                print_neuron_variables(
                    neuron,
                    currentT=t,
                    V=V,
                    u=u,
                    I=I,
                    Input_Current=Input_Current,
                    resting_current=resting_current,
                    max_mV=max_mV,
                    a_values=a_values,
                    b_values=b_values,
                    Ut=Ut,
                    Rt=Rt,
                    T_f=T_f,
                    T_d=T_d,
                    dV=dV,
                    du=du,
                    W_SYN=W_SYN,
                    R_0=R_0,
                    u_0=u_0,
                    W_0=W_0,
                    a_0=a_0,
                    b_0=b_0,
                    o=o,
                    y=y,
                    stpr=stpr,
                    stpu=stpu,
                )
        
            break  # Break out of the loop to avoid further computations

        for neuron in range(Neurons):
            
            # Update STP after each time step
            stpu, stpr = update_stp(last_spike_time, time[t], stpu, stpr, T_f, T_d, dt, Neurons)

            neighbors = get_neighbors(neuron, grid_size[0], grid_size[1])
            
            # Interact with neighbors
            neighbor_interaction = np.sum(np.dot(sum_weights[neuron], (V[neighbors, t-1] - resting_current[neighbors])))

            # Check if the neuron is in the refractory period
            time_since_last_spike = time[t] - last_spike_time[neuron]
            if time_since_last_spike < absolute_refractory_period:
                continue

            # Update current based on neighbor interaction and input current
            w = 1
            I[neuron, t] =  np.clip(((neighbor_interaction*w)), -1, 5) + Input_Current[neuron, t]


        # Update neuron dynamics (as per your model)
        a_values[:, t] = a_0[:] + (o[:] * I[:, t])
        b_values[:, t] = b_0[:] + (y[:] * V[:, t])
        dV[:, t] = (((0.04 * V[:, t-1]**2) + (5 * V[:, t-1]) + 140 - u[:, t-1]) + I[:, t] + (sigma[:] * np.random.uniform(0, 2, Neurons)))
        du[:, t] = a_values[:, t] * (b_values[:, t] - u[:, t-1])
        dW[:] = update_weights(last_spike_time, time[t], V[:,t], dW[:], W_0[:], A_p[:], A_m[:], T_p[:], T_m[:], T_d[:], T_f[:], R_0[:], u_0[:], Neurons)

        V[:, t] = V[:, t-1] + dV[:, t] * dt
        u[:, t] = u[:, t-1] + du[:, t] * dt

        # Handle spiking
        spiking_neurons = V[:, t] > max_mV[:]
        if np.any(spiking_neurons):
            for neuron in np.where(spiking_neurons)[0]:
                spike_times_per_neuron[neuron].append(time[t])
                last_spike_time[neuron] = time[t]
        
        V[spiking_neurons, t] = resting_current[spiking_neurons]
        u[spiking_neurons, t] += u_0[spiking_neurons]
        
        # Correct W_SYN update
        W_SYN[:, :] = (W_0[:] + dW[:]) #* stpu[:] * stpr[:] #(u_0[:].reshape(-1, 1) * R_0[:].reshape(1, -1)) + dW[:]
        
    def plt1():
        # Plotting section
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
        plt.subplots_adjust(left=0.1, bottom=0.1)
        
        # Clear previous plots
        ax1.clear()
        ax2.clear()
        for neuron in range(Neurons):
            ax1.plot(time, V[neuron, :], color="black")
            
        ax1.set_title('Membrane Potential of Neurons')
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Membrane Potential (mV)')
        
        # Create a colormap for membrane potential
        # Normalize membrane potential for colormap
        v_min = -80
        v_max = 40
                    
        ax2.clear()  # Clear previous content
        ax2.imshow(V, aspect='auto', extent=[0, SimTime, Neurons, 0], cmap='viridis', vmin=v_min, vmax=v_max, origin='upper')
                    
                    # Set titles and labels
        ax2.set_title('Membrane Potential Heatmap')
        ax2.set_xlabel('Time (ms)')
        ax2.set_ylabel('Neuron Index')
            
            # Adjust axes limits
        ax1.set_xlim([0, SimTime])
        ax1.set_ylim([np.min(V) - 2, np.max(V) + 10])
        ax2.set_xlim([0, SimTime])
        ax2.set_ylim([0, Neurons])  # Adjust to display all neurons
            
        plt.tight_layout()
        plt.show()
                
    def plt2():
        t=2
        difference = W_SYN[:] - W_0[:]
        if t == 1:
            plt.figure(figsize=(10, 5))
            for neuron in range(Neurons):
                plt.plot(difference[neuron] , alpha=0.5)
            plt.xlabel("Time Step")
            plt.ylabel("Current (arbitrary units)")
            plt.title("Predictable Random Input Current")
            plt.show()
        if t == 2:
            plt.hist(difference, bins=10, edgecolor='black')
            plt.title('Histogram of Differences of dW')
            plt.xlabel('Difference')
            plt.ylabel('Frequency')
            plt.show()
           
    def plt3():
        # After the simulation ends
       weight_change = W_SYN - W_0  # Calculate change in weights
       weight_change_reshaped = weight_change.reshape(grid_size[0], grid_size[1], -1).mean(axis=2)  # Average over time if needed
    
       # Plotting section for weight change
       plt.figure(figsize=(8, 6))
       plt.imshow(weight_change_reshaped, cmap='hot', interpolation='nearest')
       plt.colorbar(label='Weight Change')
       plt.title('Change in Synaptic Weights')
       plt.xlabel('Neuron Index (Column)')
       plt.ylabel('Neuron Index (Row)')
       plt.xticks(ticks=np.arange(grid_size[1]), labels=np.arange(grid_size[1]))
       plt.yticks(ticks=np.arange(grid_size[0]), labels=np.arange(grid_size[0]))
       plt.show()
    plt1()
    #plt2()
    plt3()  # Show the histogram
    
def logistic_map(x, r=3.9):
    return r * x * (1 - x)

for e in range(epochs):
    #np.random.seed(42)
    for i in range(num_interations):
        
        (V, u, I, Input_Current, resting_current, max_mV, a_values, b_values,
         Ut, Rt, T_f, T_d, dV, du, W_SYN, R_0, u_0, W_0, a_0, b_0, o, y, U, T_p, T_m, A_p, A_m, n_p, n_m, stpr, stpu) = reset_variables()
        
        # Reset all variables for the current iteration
        dW = np.zeros((Neurons, Neurons), dtype=np.float32)  # STDP weight changes
        
        V[:, 0] = resting_current[:]
        u[:, 0] = b_0 + (y * V[:, 0])
        
        du[:, 0] = 0
        dV[:, 0] = 0
        
        a_values[:, 0] = a_0 
        b_values[:, 0] = b_0 
        
        run_simulation(i, V, u, I, Input_Current, resting_current, max_mV, a_values, b_values,
                       Ut, Rt, T_f, T_d, dV, du, W_SYN, R_0, u_0, W_0, a_0, b_0, o, y, U, T_p, T_m, A_p, A_m, n_p, n_m, stpr, stpu)
