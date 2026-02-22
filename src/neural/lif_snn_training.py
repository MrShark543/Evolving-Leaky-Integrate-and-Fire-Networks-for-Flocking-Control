"""
LIF SNN Training System for Boids 

"""

import os
import json
import torch
import copy
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple
import pickle

# Import your existing classes
from src.simulation.environment import FlockingEnvironment
from src.boids.simple_snn_boid import SimpleSNNBoid
from src.neural.network import SimpleSNN

class LIFSNNTrainer:
    """
    Evolutionary trainer specifically for LIF Spiking Neural Networks
    """
    
    def __init__(self, population_size: int = 20, mutation_rate: float = 0.15, 
                 save_dir: str = "lif_trained_models"):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.save_dir = save_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"LIF SNN Trainer using device: {self.device}")
        # Create save directory
        self.create_save_directory()
        
        # Training state
        self.best_weights = None
        self.best_fitness = float('-inf')
        self.training_history = []
        
        # LIF-specific parameters
        self.spike_rate_weight = 0.2  # Weight for spike rate diversity
        self.temporal_weight = 0.15   # Weight for temporal dynamics
        
        # Flocking fitness parameters
        self.target_cohesion = 50.0
        self.min_separation = 18.0
        self.target_alignment = 0.65
        
        # Load existing weights if available
        self.load_best_weights()
    
    def create_save_directory(self):
        """Create directory for saving LIF models"""
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            print(f"Created LIF model directory: {self.save_dir}")
    
    def save_best_weights(self, network: SimpleSNN, fitness: float, metadata: dict = None):
        """Save the best LIF SNN weights"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save network weights
        weights_file = os.path.join(self.save_dir, "best_lif_snn_weights.pt")
        torch.save(network.state_dict(), weights_file)
        
        # Save metadata
        metadata_file = os.path.join(self.save_dir, "best_lif_snn_metadata.json")
        metadata_info = {
            "fitness": float(fitness),
            "timestamp": timestamp,
            "network_type": "LIF_SNN",
            "architecture": {
                "input_size": 8,
                "hidden_size": 8,
                "output_size": 3
            },
            "lif_parameters": {
                "hidden_beta": 0.7,
                "hidden_threshold": 0.5,
                "output_beta": 0.6,
                "output_threshold": 0.4
            },
            "training_version": "lif_snn_v1_fixed"
        }
        
        if metadata:
            for key, value in metadata.items():
                if hasattr(value, 'item'):
                    metadata_info[key] = value.item()
                elif isinstance(value, (np.integer, np.floating)):
                    metadata_info[key] = value.item()
                else:
                    metadata_info[key] = value
        
        try:
            with open(metadata_file, 'w') as f:
                json.dump(metadata_info, f, indent=2)
        except Exception as e:
            print(f"Could not save metadata: {e}")
        
        print(f"Saved LIF SNN weights (fitness: {fitness:.4f}) to {self.save_dir}")
        
        self.best_weights = copy.deepcopy(network)
        self.best_fitness = float(fitness)
    
    def load_best_weights(self) -> bool:
        """Load previously saved LIF SNN weights"""
        weights_file = os.path.join(self.save_dir, "best_lif_snn_weights.pt")
        metadata_file = os.path.join(self.save_dir, "best_lif_snn_metadata.json")
        
        if not os.path.exists(weights_file) or not os.path.exists(metadata_file):
            print("No previously trained LIF SNN weights found")
            return False
        
        try:
            # Load metadata
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
         
            if metadata.get("network_type") != "LIF_SNN":
                print("Found weights but not for LIF SNN")
                return False
            
            
            arch = metadata["architecture"]
            network = SimpleSNN(
                input_size=arch["input_size"],
                hidden_size=arch["hidden_size"],
                output_size=arch["output_size"]
            )
            
            # Load weights
            state_dict = torch.load(weights_file, map_location='cpu')
            network.load_state_dict(state_dict)
            
            self.best_weights = network
            self.best_fitness = metadata["fitness"]
            
            version = metadata.get("training_version", "unknown")
            print(f"Loaded {version} LIF SNN weights (fitness: {self.best_fitness:.4f})")
            return True
            
        except Exception as e:
            print(f"Failed to load LIF SNN weights: {e}")
            return False
    
    def safe_get_output_rates(self, network):
        """
        Safely get output rates with proper tensor handling
        """
        try:
            output_rates = network.get_output_rates()
            
            # Handle different tensor shapes
            if output_rates is None:
                return np.array([0.3, 0.5, 0.4])
            
            # Convert to numpy safely
            if isinstance(output_rates, torch.Tensor):
                # Ensure proper shape before converting
                if len(output_rates.shape) == 0:  # Scalar tensor
                    output_rates = output_rates.unsqueeze(0).unsqueeze(0)
                elif len(output_rates.shape) == 1:  # 1D tensor
                    output_rates = output_rates.unsqueeze(0)
                
                # Convert to numpy
                rates_np = output_rates.cpu().squeeze().detach().numpy()
            else:
                rates_np = np.array(output_rates)
            
            # Handle scalar case
            if rates_np.ndim == 0:
                rates_np = np.array([rates_np.item()])
            
            # Ensure we have at least 3 outputs
            if len(rates_np) < 3:
                # Pad to 3 outputs
                padded = np.array([0.3, 0.5, 0.4])
                padded[:len(rates_np)] = rates_np
                rates_np = padded
            elif len(rates_np) > 3:
                # Take first 3
                rates_np = rates_np[:3]
            
            return rates_np
            
        except Exception as e:
            print(f"    Warning: Error getting output rates: {e}")
            return np.array([0.3, 0.5, 0.4])  # Default fallback
    
   

    def evaluate_lif_snn_fitness(self, network: SimpleSNN, num_boids: int = 12, 
                             steps: int = 300) -> Tuple[float, Dict]:
        """
        Evaluate fitness of a LIF SNN with emphasis on spiking dynamics
        """

        base_area = 500 * 400  # 200,000 pixels for 20 boids
        target_pixels_per_boid = base_area / 20  # 10,000 pixels per boid
        
        total_area_needed = num_boids * target_pixels_per_boid
        scale_factor = (total_area_needed / base_area) ** 0.5
        
        # Scale dimensions proportionally
        width = int(500 * scale_factor)
        height = int(400 * scale_factor)
        
        # Ensure minimum size and reasonable limits
        width = max(800, min(width, 2000))   # 600-2000 pixel range
        height = max(600, min(height, 1600)) # 480-1600 pixel range
        
        # Create SCALED environment
        env = FlockingEnvironment(width=width, height=height, wrap_boundaries=False)

        # Store device and network references for the nested class
        trainer_device = self.device
        test_network = copy.deepcopy(network).to(trainer_device)
        test_network.reset_state()

        # Create temporary boid class with this LIF SNN
        class TempLIFSNNBoid(SimpleSNNBoid):
            def __init__(self, x, y, vx, vy, boid_id):
                self.device = trainer_device
                self._skip_network_init = True  # Flag to skip network creation
                
                # Call parent constructor
                super().__init__(x, y, vx, vy, boid_id, use_trained_weights=True)
                
                #  Test network 
                self.network = copy.deepcopy(test_network)
                self.network.reset_state()
            
            def _initialize_separation_focused_weights(self):
                # Override to prevent weight initialization since we're using test network
                if hasattr(self, '_skip_network_init') and self._skip_network_init:
                    pass  # Skip initialization
                else:
                    super()._initialize_separation_focused_weights()
            
            def update(self, neighbors, dt=1.0):
                """Update with error handling and CUDA support"""
                try:
                    # Get enhanced inputs
                    sensor_inputs = self._get_enhanced_inputs(neighbors)
                    
                    # Process through SNN - MOVE TO DEVICE
                    input_tensor = torch.tensor(sensor_inputs, dtype=torch.float32).to(self.device)
                    output_spikes = self.network(input_tensor)
                    
                    # CRITICAL: Move to CPU before numpy conversion
                    output_rates = self.network.get_output_rates()
                    if isinstance(output_rates, torch.Tensor):
                        output_rates = output_rates.squeeze().detach().cpu().numpy()
                    else:
                        output_rates = np.array(output_rates)
                    
                    # Get enhanced force calculation (rest of SimpleSNNBoid logic)
                    local_neighbors = self._get_local_neighbors(neighbors)
                    
                    if len(local_neighbors) > 0:
                        # Calculate classical forces
                        classical_separation = self._separation_force(local_neighbors)
                        classical_alignment = self._alignment_force(local_neighbors)
                        classical_cohesion = self._cohesion_force(local_neighbors)
                        
                        # Get SNN weights
                        if len(output_rates) >= 3:
                            sep_weight = np.clip(output_rates[0], 0.2, 2.0)
                            align_weight = np.clip(output_rates[1], 0.1, 1.0) 
                            cohesion_weight = np.clip(output_rates[2], 0.1, 1.0)
                        else:
                            sep_weight, align_weight, cohesion_weight = 1.5, 0.5, 0.5
                        
                        # Apply forces
                        total_force = (
                            sep_weight * classical_separation +
                            align_weight * classical_alignment +
                            cohesion_weight * classical_cohesion
                        )
                    else:
                        total_force = np.array([0.0, 0.0])
                    
                    # Apply force and update
                    self._apply_force(total_force, dt)
                    
                except Exception as e:
                    # Fallback to simple movement if SNN fails
                    self.position += self.velocity * dt
        
        # Add boids with the test network
        boids_created = 0
        for i in range(num_boids):
            try:
                x = np.random.uniform(50, width-50)
                y = np.random.uniform(50, height-50)
                vx = np.random.uniform(-1, 1)
                vy = np.random.uniform(-1, 1)
                boid = TempLIFSNNBoid(x, y, vx, vy, i)
                env.add_boid(boid)
                boids_created += 1
            except Exception as e:
                print(f"    Warning: Failed to create boid {i}: {e}")
        
        if boids_created == 0:
            return 0.0, {'error': 'no_boids_created'}
        
        # Track metrics and neural activity
        flocking_metrics = []
        spike_activity_history = []
        network_states = []
        
        steps_completed = 0
        
        try:
            for step in range(steps):
                try:
                    env.update()
                    steps_completed = step + 1
                    
                    # Collect flocking metrics every 10 steps after settling
                    if step > 30 and step % 10 == 0:
                        try:
                            metrics = env.get_flock_metrics()
                            flocking_metrics.append(metrics)
                        except Exception as e:
                            print(f"    Warning: Metrics error at step {step}: {e}")
                    
                    # Sample LIF neuron activity every 15 steps
                    if step % 15 == 0 and env.boids and len(env.boids) > 0:
                        try:
                            test_boid = env.boids[0]
                            
                            # Use safe output rate extraction
                            output_rates = self.safe_get_output_rates(test_boid.network)
                            spike_activity_history.append(output_rates)
                            
                        except Exception as e:
                            print(f"    Warning: Neural activity error at step {step}: {e}")
                    
                    # Early termination for failed simulations
                    if step > 60 and step % 40 == 0:
                        try:
                            positions = np.array([boid.position for boid in env.boids])
                            speeds = [np.linalg.norm(boid.velocity) for boid in env.boids]
                            
                            # Check if boids are stuck or scattered
                            if np.mean(speeds) < 0.05:  # Too slow
                                print(f"    Early termination: boids stuck at step {step}")
                                break
                            
                            # Check if all boids in corner
                            x_spread = np.max(positions[:, 0]) - np.min(positions[:, 0])
                            y_spread = np.max(positions[:, 1]) - np.min(positions[:, 1])
                            if x_spread < 30 and y_spread < 30:
                                print(f"    Early termination: corner clustering at step {step}")
                                break
                        except Exception as e:
                            print(f"    Warning: Early termination check error: {e}")
                
                except Exception as e:
                    print(f"    Warning: Step {step} simulation error: {e}")
                    break
        
        except Exception as e:
            print(f"    Evaluation error: {e}")
            return 0.0, {'error': str(e)}
        
        # Calculate comprehensive fitness
        try:
            return self.calculate_lif_fitness(flocking_metrics, spike_activity_history, 
                                            network_states, steps_completed)
        except Exception as e:
            print(f"    Fitness calculation error: {e}")
            return 0.0, {'error': str(e)}
    
    def calculate_lif_fitness(self, flocking_metrics: List[Dict], 
                             spike_history: List[np.ndarray],
                             network_states: List[np.ndarray], 
                             steps_completed: int) -> Tuple[float, Dict]:
        """
         Calculate fitness specifically for LIF SNNs
        """
        if not flocking_metrics:
            return 0.0, {'error': 'no_flocking_data'}
        
        try:
            # 1. FLOCKING BEHAVIOR FITNESS
            avg_cohesion = np.mean([m['cohesion'] for m in flocking_metrics])
            avg_alignment = np.mean([m['alignment'] for m in flocking_metrics])
            min_separation = min([m['separation'] for m in flocking_metrics])
            
            # Flocking scores
            cohesion_score = max(0, 1.0 - abs(avg_cohesion - self.target_cohesion) / self.target_cohesion)
            alignment_score = min(avg_alignment / self.target_alignment, 1.0)
            separation_score = min(min_separation / self.min_separation, 1.0) if min_separation > 0 else 0
            
            # 2. LIF SPIKING DYNAMICS FITNESS
            spike_fitness = 0.0
            spike_details = {}
            
            if spike_history and len(spike_history) > 0:
                try:
                    # Handle different array shapes safely
                    valid_rates = []
                    for rates in spike_history:
                        if isinstance(rates, np.ndarray) and rates.size > 0:
                            # Ensure 1D array with exactly 3 elements
                            if rates.ndim == 0:
                                rates_1d = np.array([rates.item()])
                            else:
                                rates_1d = rates.flatten()
                            
                            # Pad or truncate to 3 elements
                            if len(rates_1d) < 3:
                                padded = np.array([0.3, 0.5, 0.4])
                                padded[:len(rates_1d)] = rates_1d
                                valid_rates.append(padded)
                            else:
                                valid_rates.append(rates_1d[:3])
                    
                    if valid_rates:
                        spike_rates = np.array(valid_rates)
                        
                        # Check for active spiking
                        avg_spike_rates = np.mean(spike_rates, axis=0)
                        spike_details['avg_spike_rates'] = avg_spike_rates.tolist()
                        
                        # Penalize dead neurons
                        dead_neurons = np.sum(avg_spike_rates < 0.02)
                        spike_details['dead_neurons'] = int(dead_neurons)
                        
                        # Reward spike rate diversity 
                        spike_diversity = np.std(avg_spike_rates)
                        spike_details['spike_diversity'] = float(spike_diversity)
                        
                        # Check temporal dynamics 
                        temporal_variance = np.mean(np.var(spike_rates, axis=0))
                        spike_details['temporal_variance'] = float(temporal_variance)
                        
                        # Calculate spike fitness
                        if dead_neurons == 0:
                            activity_score = 1.0
                        elif dead_neurons == 1:
                            activity_score = 0.7
                        elif dead_neurons == 2:
                            activity_score = 0.3
                        else:
                            activity_score = 0.0
                        
                        diversity_score = min(spike_diversity / 0.2, 1.0)  # Target diversity ~0.2
                        temporal_score = min(temporal_variance / 0.1, 1.0)  # Target variance ~0.1
                        
                        spike_fitness = (0.5 * activity_score + 
                                       0.3 * diversity_score + 
                                       0.2 * temporal_score)
                    else:
                        spike_details['dead_neurons'] = 3
                        spike_details['error'] = 'no_valid_spike_data'
                
                except Exception as e:
                    print(f"    Warning: Spike analysis error: {e}")
                    spike_details['dead_neurons'] = 3
                    spike_details['error'] = str(e)
            else:
                spike_details['dead_neurons'] = 3
                spike_details['error'] = 'no_spike_history'
            
            # 3. NETWORK STATE FITNESS 
            membrane_fitness = 0.0
            if network_states and len(network_states) > 0:
                try:
                    # Check that membrane potentials are reasonable and dynamic
                    membrane_data = np.array(network_states)
                    if len(membrane_data) > 0:
                        membrane_variance = np.mean(np.var(membrane_data, axis=0))
                        membrane_fitness = min(membrane_variance / 0.5, 1.0)
                except Exception as e:
                    print(f"    Warning: Membrane fitness error: {e}")
            
            # 4. STABILITY FITNESS
            stability_score = min(steps_completed / 250.0, 1.0)
            
            # 5. COMBINED FITNESS with LIF-specific weighting
            weights = {
                'flocking': 0.60,      # Primary: good flocking
                'spike_dynamics': 0.25, # Proper LIF spiking
                'membrane_state': 0.05, # Minor: membrane dynamics
                'stability': 0.10       # Important: simulation completion
            }
            
            flocking_fitness = (0.45 * cohesion_score + 
                               0.35 * alignment_score + 
                               0.2 * separation_score)
            
            total_fitness = (weights['flocking'] * flocking_fitness +
                            weights['spike_dynamics'] * spike_fitness +
                            weights['membrane_state'] * membrane_fitness +
                            weights['stability'] * stability_score)
            
            # Bonus for excellent performance
            dead_neurons = spike_details.get('dead_neurons', 3)
            if (flocking_fitness > 0.7 and spike_fitness > 0.6 and dead_neurons == 0):
                total_fitness += 0.1  # Excellence bonus
            
            details = {
                'flocking_fitness': flocking_fitness,
                'spike_fitness': spike_fitness,
                'membrane_fitness': membrane_fitness,
                'stability_score': stability_score,
                'cohesion_score': cohesion_score,
                'alignment_score': alignment_score,
                'separation_score': separation_score,
                'avg_cohesion': avg_cohesion,
                'avg_alignment': avg_alignment,
                'min_separation': min_separation,
                'steps_completed': steps_completed,
                **spike_details
            }
            
            return max(0.0, total_fitness), details
        
        except Exception as e:
            print(f"    Error in fitness calculation: {e}")
            return 0.0, {'error': str(e)}
    
    def mutate_lif_network(self, network: SimpleSNN, mutation_strength: float = 0.1):
        """
        Mutation specifically designed for LIF SNNs
        """
        network = network.to(self.device)
        with torch.no_grad():
            # Standard weight mutation
            for param in network.parameters():
                if np.random.random() < self.mutation_rate:
                    mutation = torch.randn_like(param) * mutation_strength
                    param.add_(mutation)
                    # Keep weights in reasonable range for LIF dynamics
                    param.clamp_(-1.0, 1.0)
            
            # LIF-specific mutations
            
            # 1. Boost weak output connections
            if np.random.random() < 0.2:
                try:
                    output_idx = torch.randint(0, network.fc2.weight.shape[0], (1,)).item()
                    strong_connections = torch.randperm(network.fc2.weight.shape[1])[:2]
                    network.fc2.weight[output_idx, strong_connections] += torch.randn(2) * 0.3
                    network.fc2.bias[output_idx] += 0.1
                except Exception as e:
                    pass  # Skip if error
            
            # 2. Adjust input-to-hidden weights
            if np.random.random() < 0.15:
                try:
                    hidden_idx = torch.randint(0, network.fc1.weight.shape[0], (1,)).item()
                    input_connections = torch.randperm(network.fc1.weight.shape[1])[:2]
                    network.fc1.weight[hidden_idx, input_connections] += torch.randn(2) * 0.2
                except Exception as e:
                    pass  # Skip if error
            
            # 3. Bias adjustment 
            if np.random.random() < 0.1:
                try:
                    # Adjust hidden biases
                    bias_indices = torch.randperm(network.fc1.bias.shape[0])[:2]
                    network.fc1.bias[bias_indices] += torch.randn(2) * 0.15
                    
                    # Adjust output biases
                    output_bias_idx = torch.randint(0, network.fc2.bias.shape[0], (1,)).item()
                    network.fc2.bias[output_bias_idx] += torch.randn(1) * 0.1
                except Exception as e:
                    pass  # Skip if error
    
    def crossover_networks(self, network1: SimpleSNN, network2: SimpleSNN, 
                          alpha: float = 0.5) -> SimpleSNN:
        """Create offspring by combining two LIF SNNs"""
        new_network = copy.deepcopy(network1)
        new_network = new_network.to(self.device)
        
        try:
            with torch.no_grad():
                for param1, param2, new_param in zip(
                    network1.parameters(), 
                    network2.parameters(), 
                    new_network.parameters()
                ):
                    new_param.data = alpha * param1.data + (1 - alpha) * param2.data
        except Exception as e:
            print(f"    Warning: Crossover error: {e}")
            # Return one parent if crossover fails
            return copy.deepcopy(network1)
        
        return new_network
    
    def tournament_selection(self, fitness_values: List[float], 
                           tournament_size: int = 4) -> int:
        """Tournament selection for parent selection"""
        try:
            tournament_indices = np.random.choice(len(fitness_values), tournament_size, replace=False)
            tournament_fitness = [fitness_values[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            return winner_idx
        except Exception as e:
            # Fallback to random selection
            return np.random.randint(0, len(fitness_values))
    
    def train_lif_snn_population(self, generations: int = 25, 
                                evaluation_steps: int = 300) -> Dict:
        """
         Main training loop for LIF SNN population
        """
        print(f"Training LIF SNN Population for Boid Flocking")
        print(f"Generations: {generations}")
        print(f"Population: {self.population_size}")
        print(f"Evaluation steps: {evaluation_steps}")
        print(f"Architecture: 6 inputs → 8 LIF hidden → 3 LIF outputs")
        print(f"Target: Cohesion={self.target_cohesion}, Alignment≥{self.target_alignment}")
        
        # Create initial population of LIF SNNs
        population = []
        for i in range(self.population_size):
            try:
                # Create base LIF SNN
                network = SimpleSNN(input_size=8, hidden_size=8, output_size=3)
                network = network.to(self.device)
                # Initialize with LIF-friendly weights
                with torch.no_grad():
                    # Moderate random initialization
                    for param in network.parameters():
                        param.data = torch.randn_like(param) * 0.4
                    
                    # Ensure positive biases for easier spiking
                    network.fc1.bias.data = torch.abs(network.fc1.bias.data) * 0.5 + 0.2
                    network.fc2.bias.data = torch.abs(network.fc2.bias.data) * 0.3 + 0.15
                    
                    # Ensure each output has some strong connections
                    for output_idx in range(network.fc2.weight.shape[0]):
                        strong_connections = torch.randperm(network.fc2.weight.shape[1])[:2]
                        network.fc2.weight.data[output_idx, strong_connections] = torch.abs(
                            network.fc2.weight.data[output_idx, strong_connections]
                        ) + 0.3
                
                population.append(network)
            
            except Exception as e:
                print(f"    Warning: Failed to create network {i}: {e}")
                # Create a simpler fallback network
                try:
                    network = SimpleSNN(input_size=8, hidden_size=8, output_size=3)
                    population.append(network)
                except Exception as e2:
                    print(f"    Error: Cannot create any network: {e2}")
                    return {'error': 'network_creation_failed'}
        
        if len(population) == 0:
            return {'error': 'no_networks_created'}
        
        fitness_history = []
        best_generation_fitness = 0.0
        
        for generation in range(generations):
            print(f"\n🔄 Generation {generation + 1}/{generations}")
            
            # Evaluate population
            generation_fitness = []
            generation_details = []
            
            for i, network in enumerate(population):
                try:
                    fitness, details = self.evaluate_lif_snn_fitness(
                        network, 
                        num_boids=10, 
                        steps=evaluation_steps
                    )
                    generation_fitness.append(fitness)
                    generation_details.append(details)
                    
                    if i % 5 == 0:
                        dead_neurons = details.get('dead_neurons', 'unknown')
                        print(f"  Individual {i+1}/{self.population_size}: "
                              f"fitness = {fitness:.3f}, dead_neurons = {dead_neurons}")
                
                except Exception as e:
                    print(f"  Individual {i+1}: Evaluation failed: {e}")
                    generation_fitness.append(0.0)
                    generation_details.append({'error': str(e)})
            
            # Track statistics
            if generation_fitness:
                best_gen_fitness = max(generation_fitness)
                avg_gen_fitness = np.mean(generation_fitness)
                best_idx = generation_fitness.index(best_gen_fitness)
                best_details = generation_details[best_idx]
                
                print(f"Generation {generation + 1} Results:")
                print(f"Best fitness: {best_gen_fitness:.3f}")
                print(f"Average fitness: {avg_gen_fitness:.3f}")
                print(f"Best details: flocking={best_details.get('flocking_fitness', 0):.3f}, "
                      f"spikes={best_details.get('spike_fitness', 0):.3f}")
                print(f"Dead neurons in best: {best_details.get('dead_neurons', 'unknown')}")
                
                fitness_history.append({
                    'generation': generation + 1,
                    'best_fitness': best_gen_fitness,
                    'avg_fitness': avg_gen_fitness,
                    'best_details': best_details
                })
                
                # Save if improved
                if best_gen_fitness > self.best_fitness:
                    try:
                        self.save_best_weights(
                            population[best_idx], 
                            best_gen_fitness,
                            {
                                "generation": generation + 1,
                                "population_size": self.population_size,
                                "method": "evolutionary_lif_snn",
                                "evaluation_steps": evaluation_steps,
                                **best_details
                            }
                        )
                        print(f"New global best: {best_gen_fitness:.3f}")
                        best_generation_fitness = best_gen_fitness
                    except Exception as e:
                        print(f"     Warning: Failed to save weights: {e}")
                
                # Early stopping
                if best_gen_fitness > 0.95:
                    print(f"Early stopping - excellent LIF SNN achieved!")
                    break
            else:
                print("No valid fitness values this generation")
                break
            
            # Create next generation
            if generation < generations - 1 and generation_fitness:
                try:
                    new_population = []
                    
                    # Sort by fitness
                    sorted_indices = np.argsort(generation_fitness)[::-1]
                    
                    # Strong elitism - keep top 25%
                    elite_count = max(2, self.population_size // 4)
                    for i in range(elite_count):
                        idx = sorted_indices[i]
                        if generation_fitness[idx] > 0:  # Only keep good individuals
                            new_population.append(copy.deepcopy(population[idx]))
                    
                    # Generate offspring
                    while len(new_population) < self.population_size:
                        try:
                            # Select parents with valid fitness
                            valid_indices = [i for i, f in enumerate(generation_fitness) if f > 0]
                            if len(valid_indices) >= 2:
                                parent1_idx = self.tournament_selection([generation_fitness[i] for i in valid_indices], tournament_size=min(5, len(valid_indices)))
                                parent2_idx = self.tournament_selection([generation_fitness[i] for i in valid_indices], tournament_size=min(5, len(valid_indices)))
                                
                                # Map back to original indices
                                parent1_idx = valid_indices[parent1_idx]
                                parent2_idx = valid_indices[parent2_idx]
                                
                                offspring = self.crossover_networks(
                                    population[parent1_idx], 
                                    population[parent2_idx],
                                    alpha=np.random.uniform(0.3, 0.7)
                                )
                            else:
                                # If no valid parents, create new random network
                                offspring = SimpleSNN(input_size=8, hidden_size=8, output_size=3)
                            
                            # Adaptive mutation
                            mut_strength = 0.15 if avg_gen_fitness < 0.3 else 0.1
                            self.mutate_lif_network(offspring, mutation_strength=mut_strength)
                            new_population.append(offspring)
                            
                        except Exception as e:
                            print(f"Warning: Offspring creation error: {e}")
                            # Add a copy of best individual as fallback
                            if generation_fitness and max(generation_fitness) > 0:
                                best_idx = generation_fitness.index(max(generation_fitness))
                                new_population.append(copy.deepcopy(population[best_idx]))
                            else:
                                # Last resort: create new random network
                                try:
                                    new_network = SimpleSNN(input_size=8, hidden_size=8, output_size=3)
                                    new_population.append(new_network)
                                except:
                                    break  # 
                    
                    population = new_population
                    
                except Exception as e:
                    print(f"     Error creating next generation: {e}")
                    break
        
        return {
            'best_fitness': self.best_fitness,
            'best_weights': self.best_weights,
            'fitness_history': fitness_history,
            'generations': generations,
            'final_best': best_generation_fitness,
            'network_type': 'LIF_SNN'
        }
    
    def has_trained_weights(self) -> bool:
        """Check if trained LIF SNN weights are available"""
        return self.best_weights is not None
    
    def apply_trained_weights_to_boid(self, snn_boid: SimpleSNNBoid) -> bool:
        """Apply trained LIF SNN weights to a boid"""
        if not self.has_trained_weights():
            return False
        
        try:
            with torch.no_grad():
                for param, best_param in zip(snn_boid.network.parameters(), 
                                           self.best_weights.parameters()):
                    param.data = best_param.data.clone()
            # Reset network state for clean start
            snn_boid.network.reset_state()
            return True
        except Exception as e:
            print(f"Failed to apply LIF SNN weights: {e}")
            return False
    
    def create_trained_lif_flock(self, num_boids: int, 
                                environment_bounds: Tuple[int, int] = (800, 600)) -> List[SimpleSNNBoid]:
        """Create a flock of boids with trained LIF SNN weights"""
        if not self.has_trained_weights():
            return []
        
        width, height = environment_bounds
        trained_boids = []
        
        for i in range(num_boids):
            try:
                # Start boids in center region for better flocking
                x = np.random.uniform(width*0.2, width*0.8)
                y = np.random.uniform(height*0.2, height*0.8)
                vx = np.random.uniform(-1.2, 1.2)
                vy = np.random.uniform(-1.2, 1.2)
                
                boid = SimpleSNNBoid(x, y, vx, vy, boid_id=i)
                if self.apply_trained_weights_to_boid(boid):
                    trained_boids.append(boid)
                else:
                    print(f"Warning: Failed to apply weights to boid {i}")
            except Exception as e:
                print(f"Warning: Failed to create trained boid {i}: {e}")
        
        return trained_boids


# Global trainer instance
_lif_trainer = None

def get_lif_trainer() -> LIFSNNTrainer:
    """Get global LIF SNN trainer instance"""
    global _lif_trainer
    if _lif_trainer is None:
        _lif_trainer = LIFSNNTrainer()
    return _lif_trainer

def train_lif_snn_boids(generations: int = 25) -> bool:
    """Train LIF SNN boids - main training function"""
    trainer = get_lif_trainer()
    
    print(f"🚀 Starting LIF SNN Training for Boid Flocking")
    print(f"   This will train actual spiking neural networks!")
    
    try:
        results = trainer.train_lif_snn_population(
            generations=generations,
            evaluation_steps=300
        )
        
        if 'error' in results:
            print(f"Training failed: {results['error']}")
            return False
        
        success = results['best_fitness'] > 0.4
        if success:
            print(f"\nLIF SNN Training Successful!")
            print(f"   Best fitness: {results['best_fitness']:.4f}")
            print(f"   Trained LIF SNNs ready for flocking!")
        else:
            print(f"\nLIF SNN Training completed but low fitness: {results['best_fitness']:.4f}")
            print(f"   Consider running more generations or adjusting parameters")
        
        return success
        
    except Exception as e:
        print(f"LIF SNN Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_trained_lif_flock(num_boids: int) -> List[SimpleSNNBoid]:
    """Create a flock of boids with trained LIF SNN weights"""
    trainer = get_lif_trainer()
    return trainer.create_trained_lif_flock(num_boids)

def get_lif_training_status() -> Dict:
    """Get LIF SNN training status"""
    trainer = get_lif_trainer()
    return {
        "has_trained_weights": trainer.has_trained_weights(),
        "best_fitness": trainer.best_fitness if trainer.has_trained_weights() else None,
        "save_directory": trainer.save_dir,
        "network_type": "LIF_SNN",
        "target_cohesion": trainer.target_cohesion,
        "target_alignment": trainer.target_alignment
    }

def test_lif_snn_training():
    """Test function to verify LIF SNN training setup"""
    print("Testing LIF SNN Training Setup")
    
    # Test network creation
    try:
        network = SimpleSNN(input_size=8, hidden_size=8, output_size=3)
        print("LIF SNN network creation: OK")
        
        # Test forward pass
        test_input = torch.randn(1, 6)
        output = network(test_input)
        print(f"LIF SNN forward pass: OK (output shape: {output.shape})")
        
        # Test output rates
        rates = network.get_output_rates()
        print(f"LIF SNN output rates: OK (rates shape: {rates.shape})")
        
    except Exception as e:
        print(f"LIF SNN network creation failed: {e}")
        return False
    
    # Test boid creation
    try:
        boid = SimpleSNNBoid(100, 100, 1, 1, boid_id=0)
        print("SimpleSNNBoid creation: OK")
        
        # Test boid update
        boid.update([], dt=1.0)
        print("SimpleSNNBoid update: OK")
        
    except Exception as e:
        print(f"SimpleSNNBoid creation failed: {e}")
        return False
    
    # Test trainer creation
    try:
        trainer = LIFSNNTrainer(population_size=5)
        print("LIF SNN trainer creation: OK")
        
        # Test safe output rates function
        test_network = SimpleSNN(input_size=8, hidden_size=8, output_size=3)
        rates = trainer.safe_get_output_rates(test_network)
        print(f"Safe output rates: OK (shape: {rates.shape})")
        
    except Exception as e:
        print(f"LIF SNN trainer creation failed: {e}")
        return False
    
    print("LIF SNN training setup is ready!")
    print("Run train_lif_snn_boids(generations=20) to start training")
    return True

# Additional utility functions for debugging
def debug_tensor_shapes():
    """Debug function to check tensor shapes in the network"""
    print("🔍 Debugging tensor shapes...")
    
    try:
        network = SimpleSNN(input_size=8, hidden_size=8, output_size=3)
        
        # Test with different input shapes
        test_inputs = [
            torch.randn(6),           # 1D
            torch.randn(1, 6),        # 2D batch
            torch.randn(2, 6),        # 2D multi-batch
        ]
        
        for i, test_input in enumerate(test_inputs):
            print(f"  Test {i+1}: Input shape {test_input.shape}")
            try:
                output = network(test_input)
                rates = network.get_output_rates()
                print(f"Output shape: {output.shape}")
                print(f"Rates shape: {rates.shape}")
                print(f"Success")
            except Exception as e:
                print(f"Failed: {e}")
        
    except Exception as e:
        print(f"Debug failed: {e}")

if __name__ == "__main__":
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

    print("LIF SNN Training System - Fixed Version")
    print("=" * 50)
    
    # Run tests
    if test_lif_snn_training():
        print("\nRunning tensor shape debugging...")
        debug_tensor_shapes()
        
        print(f"\nReady to train LIF SNNs!")
        print("   Use: train_lif_snn_boids(generations=25)")
    else:
        print("\nSetup verification failed")
        print("   Check your project structure and imports")