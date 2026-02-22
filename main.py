"""
LIF SNN Training
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
import copy
import pickle
import json
import multiprocessing as mp
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from functools import partial
import time

# Add path t
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Force CPU-only mode
os.environ['CUDA_VISIBLE_DEVICES'] = ''
torch.set_num_threads(1)  # Prevent thread oversubscription in multiprocessing

# Import your existing LIF training system
try:
    from src.neural.lif_snn_training import LIFSNNTrainer, get_lif_trainer
    from src.neural.network import SimpleSNN
    from src.neural.neurons import LIFNeuron
    from src.simulation.environment import FlockingEnvironment
    from src.boids.simple_snn_boid import SimpleSNNBoid
    IMPORTS_OK = True
    print("Successfully imported existing LIF SNN system")
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the correct directory")
    IMPORTS_OK = False

# Global function for parallel evaluation
def evaluate_network_parallel(args):
    """
    Evaluate a single network (for parallel processing)
    Module-level function for multiprocessing
    """
    network_idx, network_state_dict, eval_params = args
    
    # Reconstruct network from state dict
    network = SimpleSNN(
        input_size=eval_params['input_size'],
        hidden_size=eval_params['hidden_size'], 
        output_size=eval_params['output_size']
    )
    network.load_state_dict(network_state_dict)
    network.eval()  # Set to evaluation mode
    
    # Create evaluator instance
    evaluator = NetworkEvaluator(eval_params)
    
    # Evaluate
    fitness, details = evaluator.evaluate_network(network)
    
    return network_idx, fitness, details



class ComprehensiveEvaluator:
    """Comprehensive evaluation and data collection for trained models"""
    
    def __init__(self, save_dir: str = "evaluation_results"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.evaluation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def evaluate_scalability(self, network: SimpleSNN, 
                            boid_counts: List[int] = [15, 25, 50, 75, 100],
                            steps_per_eval: int = 500,
                            trials_per_count: int = 3) -> Dict:
        """Evaluate model performance at different scales"""
        
        print(f"\nEvaluating Scalability across {boid_counts} boids...")
        scalability_results = {
            'boid_counts': boid_counts,
            'trials': trials_per_count,
            'steps': steps_per_eval,
            'results': {}
        }
        
        for num_boids in boid_counts:
            print(f"\n  Testing with {num_boids} boids...")
            trial_results = []
            
            for trial in range(trials_per_count):
                # Create evaluator with adaptive environment size
                eval_params = self._get_eval_params(num_boids)
                evaluator = NetworkEvaluator(eval_params)
                
                # Run evaluation
                start_time = time.time()
                fitness, details = evaluator.evaluate_network(network)
                eval_time = time.time() - start_time
                
                # Collect detailed metrics
                trial_data = {
                    'trial': trial,
                    'fitness': fitness,
                    'eval_time': eval_time,
                    'cohesion_score': details.get('cohesion_score', 0),
                    'alignment_score': details.get('alignment_score', 0),
                    'separation_score': details.get('separation_score', 0),
                    'avg_cohesion': details.get('avg_cohesion', 0),
                    'avg_alignment': details.get('avg_alignment', 0),
                    'min_separation': details.get('min_separation', 0),
                    'collisions': details.get('collisions', 0),
                    'collision_rate': details.get('collisions', 0) / (steps_per_eval / 10),  # per metric collection
                    'dead_neurons': details.get('dead_neurons', 0),
                    'spike_fitness': details.get('spike_fitness', 0),
                    'flocking_fitness': details.get('flocking_fitness', 0)
                }
                trial_results.append(trial_data)
                
                print(f"    Trial {trial+1}: fitness={fitness:.3f}, collisions={trial_data['collisions']}")
            
            # Aggregate results
            scalability_results['results'][num_boids] = {
                'trials': trial_results,
                'mean_fitness': np.mean([t['fitness'] for t in trial_results]),
                'std_fitness': np.std([t['fitness'] for t in trial_results]),
                'mean_collisions': np.mean([t['collisions'] for t in trial_results]),
                'mean_collision_rate': np.mean([t['collision_rate'] for t in trial_results]),
                'mean_eval_time': np.mean([t['eval_time'] for t in trial_results]),
                'mean_cohesion': np.mean([t['avg_cohesion'] for t in trial_results]),
                'mean_alignment': np.mean([t['avg_alignment'] for t in trial_results]),
                'mean_separation': np.mean([t['min_separation'] for t in trial_results])
            }
        
        # Save results
        filename = f"scalability_results_{self.evaluation_id}.pkl"
        filepath = os.path.join(self.save_dir, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(scalability_results, f)
        
        # Also save as JSON for easy reading
        json_filename = f"scalability_results_{self.evaluation_id}.json"
        json_filepath = os.path.join(self.save_dir, json_filename)
        json_safe_results = self._make_json_safe(scalability_results)
        with open(json_filepath, 'w') as f:
            json.dump(json_safe_results, f, indent=2)
        
        print(f"\n✅ Scalability evaluation complete. Saved to {filepath}")
        return scalability_results
    
    def evaluate_collision_patterns(self, network: SimpleSNN,
                                   num_boids: int = 50,
                                   steps: int = 1000,
                                   detailed_tracking: bool = True) -> Dict:
        """Detailed collision analysis over time"""
        
        print(f"\nAnalyzing Collision Patterns...")
        
        # Setup environment
        eval_params = self._get_eval_params(num_boids)
        env_width = eval_params['env_width']
        env_height = eval_params['env_height']
        
        # Create environment and boids
        env = FlockingEnvironment(width=env_width, height=env_height, wrap_boundaries=False)
        
        # Create boids with the network
        for i in range(num_boids):
            x = np.random.uniform(50, env_width-50)
            y = np.random.uniform(50, env_height-50)
            vx = np.random.uniform(-1, 1)
            vy = np.random.uniform(-1, 1)
            
            boid = SimpleSNNBoid(x, y, vx, vy, i, use_trained_weights=True)
            with torch.no_grad():
                for param, test_param in zip(boid.network.parameters(), network.parameters()):
                    param.data = test_param.data.clone()
            boid.network.reset_state()
            env.add_boid(boid)
        
        # Track collisions over time
        collision_data = {
            'time_steps': [],
            'collision_counts': [],
            'near_miss_counts': [],
            'min_distances': [],
            'collision_pairs': [],
            'spatial_distribution': []
        }
        
        collision_threshold = 20.0
        near_miss_threshold = 30.0
        
        for step in range(steps):
            env.update()
            
            if step % 10 == 0:  # Sample every 10 steps
                # Calculate pairwise distances
                positions = np.array([b.position for b in env.boids])
                
                collision_count = 0
                near_miss_count = 0
                min_dist = float('inf')
                collision_pairs_step = []
                
                for i in range(len(env.boids)):
                    for j in range(i+1, len(env.boids)):
                        dist = np.linalg.norm(positions[i] - positions[j])
                        min_dist = min(min_dist, dist)
                        
                        if dist < collision_threshold:
                            collision_count += 1
                            if detailed_tracking:
                                collision_pairs_step.append((i, j, dist))
                        elif dist < near_miss_threshold:
                            near_miss_count += 1
                
                collision_data['time_steps'].append(step)
                collision_data['collision_counts'].append(collision_count)
                collision_data['near_miss_counts'].append(near_miss_count)
                collision_data['min_distances'].append(min_dist)
                
                if detailed_tracking:
                    collision_data['collision_pairs'].append(collision_pairs_step)
                    
                    # Spatial distribution (grid-based density)
                    grid_size = 50
                    grid_x = int(env_width / grid_size) + 1
                    grid_y = int(env_height / grid_size) + 1
                    density_grid = np.zeros((grid_x, grid_y))
                    
                    for pos in positions:
                        gx = min(int(pos[0] / grid_size), grid_x - 1)
                        gy = min(int(pos[1] / grid_size), grid_y - 1)
                        density_grid[gx, gy] += 1
                    
                    collision_data['spatial_distribution'].append(density_grid)
        
        # Calculate summary statistics
        collision_data['summary'] = {
            'total_collisions': sum(collision_data['collision_counts']),
            'avg_collisions_per_step': np.mean(collision_data['collision_counts']),
            'max_collisions_per_step': max(collision_data['collision_counts']),
            'collision_free_steps': sum(1 for c in collision_data['collision_counts'] if c == 0),
            'avg_min_distance': np.mean(collision_data['min_distances']),
            'total_near_misses': sum(collision_data['near_miss_counts'])
        }
        
        # Save results
        filename = f"collision_analysis_{num_boids}boids_{self.evaluation_id}.pkl"
        filepath = os.path.join(self.save_dir, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(collision_data, f)
        
        print(f"Collision analysis complete. Total collisions: {collision_data['summary']['total_collisions']}")
        return collision_data
    
    def analyze_neural_dynamics(self, network: SimpleSNN,
                               test_scenarios: int = 5,
                               steps_per_scenario: int = 200) -> Dict:
        """Analyze neural activity patterns in different scenarios"""
        
        print(f"\nAnalyzing Neural Dynamics...")
        
        neural_data = {
            'scenarios': [],
            'spike_rates_over_time': [],
            'membrane_potentials': [],
            'input_output_correlations': [],
            'dead_neuron_evolution': []
        }
        
        scenarios = [
            {'name': 'sparse', 'num_boids': 10, 'area_scale': 1.5},
            {'name': 'normal', 'num_boids': 25, 'area_scale': 1.0},
            {'name': 'dense', 'num_boids': 40, 'area_scale': 0.8},
            {'name': 'very_dense', 'num_boids': 60, 'area_scale': 0.7},
            {'name': 'extreme', 'num_boids': 80, 'area_scale': 0.6}
        ]
        
        for scenario in scenarios[:test_scenarios]:
            print(f"  Testing scenario: {scenario['name']}")
            
            # Setup test environment
            base_width, base_height = 800, 600
            width = int(base_width * scenario['area_scale'])
            height = int(base_height * scenario['area_scale'])
            
            env = FlockingEnvironment(width=width, height=height, wrap_boundaries=False)
            
            # Create test boid with network
            test_boid = SimpleSNNBoid(width/2, height/2, 1, 0, 0, use_trained_weights=True)
            with torch.no_grad():
                for param, test_param in zip(test_boid.network.parameters(), network.parameters()):
                    param.data = test_param.data.clone()
            test_boid.network.reset_state()
            env.add_boid(test_boid)
            
            # Add other boids
            for i in range(1, scenario['num_boids']):
                x = np.random.uniform(50, width-50)
                y = np.random.uniform(50, height-50)
                boid = SimpleSNNBoid(x, y, np.random.uniform(-1,1), np.random.uniform(-1,1), i)
                env.add_boid(boid)
            
            # Collect neural data
            scenario_data = {
                'name': scenario['name'],
                'spike_rates': [],
                'inputs': [],
                'outputs': [],
                'dead_neurons_count': []
            }
            
            for step in range(steps_per_scenario):
                env.update()
                
                if step % 5 == 0:  # Sample every 5 steps
                    # Get neural activity
                    output_rates = test_boid.network.get_output_rates()
                    if isinstance(output_rates, torch.Tensor):
                        output_rates = output_rates.squeeze().detach().cpu().numpy()
                    
                    # Get inputs
                    neighbors = [b for b in env.boids if b.id != test_boid.id]
                    inputs = test_boid._get_enhanced_inputs(neighbors)
                    
                    scenario_data['spike_rates'].append(output_rates)
                    scenario_data['inputs'].append(inputs)
                    scenario_data['outputs'].append(output_rates)
                    
                    # Count dead neurons
                    dead_count = sum(1 for r in output_rates if r < 0.02)
                    scenario_data['dead_neurons_count'].append(dead_count)
            
            # Calculate correlations
            if len(scenario_data['inputs']) > 0:
                inputs_array = np.array(scenario_data['inputs'])
                outputs_array = np.array(scenario_data['outputs'])
                
                correlations = {}
                for i in range(inputs_array.shape[1]):  # For each input
                    for j in range(outputs_array.shape[1]):  # For each output
                        if outputs_array[:, j].std() > 0 and inputs_array[:, i].std() > 0:
                            corr = np.corrcoef(inputs_array[:, i], outputs_array[:, j])[0, 1]
                            correlations[f'input_{i}_to_output_{j}'] = corr
                
                scenario_data['correlations'] = correlations
            
            neural_data['scenarios'].append(scenario_data)
        
        # Save results
        filename = f"neural_dynamics_{self.evaluation_id}.pkl"
        filepath = os.path.join(self.save_dir, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(neural_data, f)
        
        print(f"Neural dynamics analysis complete.")
        return neural_data
    
    def evaluate_computational_efficiency(self, network: SimpleSNN,
                                        boid_counts: List[int] = [10, 25, 50, 100],
                                        steps: int = 100,
                                        num_workers_list: List[int] = [1, 2, 4, 8]) -> Dict:
        """Measure computational performance metrics"""
        
        print(f"\nEvaluating Computational Efficiency...")
        
        efficiency_data = {
            'boid_counts': boid_counts,
            'serial_times': {},
            'parallel_times': {},
            'speedup_factors': {},
            'memory_usage': {}
        }
        
        for num_boids in boid_counts:
            print(f"\n  Testing {num_boids} boids...")
            
            # Serial evaluation
            eval_params = self._get_eval_params(num_boids)
            evaluator = NetworkEvaluator(eval_params)
            
            start_time = time.time()
            evaluator.evaluate_network(network)
            serial_time = time.time() - start_time
            efficiency_data['serial_times'][num_boids] = serial_time
            
            # Parallel evaluation timing (simulated)
            parallel_times = {}
            for num_workers in num_workers_list:
                if num_workers <= mp.cpu_count():
                    # Estimate parallel time
                    estimated_time = serial_time / (num_workers * 0.8)  # 80% efficiency
                    parallel_times[num_workers] = estimated_time
            
            efficiency_data['parallel_times'][num_boids] = parallel_times
            
            # Calculate speedup
            speedups = {w: serial_time / t for w, t in parallel_times.items()}
            efficiency_data['speedup_factors'][num_boids] = speedups
            
            print(f"    Serial time: {serial_time:.2f}s")
            print(f"    Estimated speedups: {speedups}")
        
        # Save results
        filename = f"efficiency_analysis_{self.evaluation_id}.pkl"
        filepath = os.path.join(self.save_dir, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(efficiency_data, f)
        
        print(f"Efficiency analysis complete.")
        return efficiency_data
    
    def _get_eval_params(self, num_boids: int) -> Dict:
        """Get evaluation parameters for given boid count"""
        base_area = 500 * 400
        target_pixels_per_boid = base_area / 20
        total_area_needed = num_boids * target_pixels_per_boid
        scale_factor = (total_area_needed / base_area) ** 0.5
        
        env_width = max(800, min(int(500 * scale_factor), 2000))
        env_height = max(600, min(int(400 * scale_factor), 1600))
        
        return {
            'num_boids': num_boids,
            'steps': 500,
            'target_cohesion': 35.0,
            'target_alignment': 0.7,
            'min_separation': 30.0,
            'env_width': env_width,
            'env_height': env_height,
            'input_size': 8,
            'hidden_size': 12,
            'output_size': 3
        }
    
    def _make_json_safe(self, data):
        """Convert numpy types to JSON-safe types"""
        if isinstance(data, dict):
            return {k: self._make_json_safe(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._make_json_safe(v) for v in data]
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, (np.integer, np.floating)):
            return data.item()
        else:
            return data


def run_comprehensive_evaluation(weights_path: str = "optimized_lif_models/best_lif_snn_weights.pt"):
    """Run complete evaluation suite on trained model"""
    
    print("=" * 70)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("=" * 70)
    
    # Load trained network
    network = SimpleSNN(input_size=8, hidden_size=12, output_size=3)
    network.load_state_dict(torch.load(weights_path, map_location='cpu'))
    network.eval()
    
    # Create evaluator
    evaluator = ComprehensiveEvaluator()
    
    # 1. Scalability Analysis
    scalability_results = evaluator.evaluate_scalability(
        network,
        boid_counts=[15, 25, 35, 50, 75, 100],
        steps_per_eval=500,
        trials_per_count=3
    )
    
    # 2. Collision Analysis
    collision_results = evaluator.evaluate_collision_patterns(
        network,
        num_boids=50,
        steps=1000,
        detailed_tracking=True
    )
    
    # 3. Neural Dynamics
    neural_results = evaluator.analyze_neural_dynamics(
        network,
        test_scenarios=5,
        steps_per_scenario=200
    )
    
    # 4. Computational Efficiency
    efficiency_results = evaluator.evaluate_computational_efficiency(
        network,
        boid_counts=[10, 25, 50, 100],
        steps=100
    )
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print(f"Results saved to: {evaluator.save_dir}")
    print("=" * 70)
    
    return {
        'scalability': scalability_results,
        'collisions': collision_results,
        'neural': neural_results,
        'efficiency': efficiency_results
    }



class NetworkEvaluator:
    """Separate class for network evaluation """
    
    def __init__(self, eval_params: Dict):
        self.num_boids = eval_params['num_boids']
        self.steps = eval_params['steps']
        self.target_cohesion = eval_params['target_cohesion']
        self.target_alignment = eval_params['target_alignment']
        self.min_separation = eval_params['min_separation']
        self.env_width = eval_params.get('env_width', 800)
        self.env_height = eval_params.get('env_height', 600)
    
    def evaluate_network(self, network: SimpleSNN) -> Tuple[float, Dict]:
        """Evaluate a single network """
        try:
            # Create environment
            env = FlockingEnvironment(
                width=self.env_width, 
                height=self.env_height, 
                wrap_boundaries=False
            )
            
            # Create temporary boid class
            class TempLIFBoid(SimpleSNNBoid):
                def __init__(self, x, y, vx, vy, boid_id):
                    # Override device to ensure CPU only
                    self.device = torch.device('cpu')
                    super().__init__(x, y, vx, vy, boid_id, use_trained_weights=True)
                    # Replace network
                    self.network = copy.deepcopy(network)
                    self.network.eval()
                    self.network.reset_state()
                
                def update(self, neighbors, dt=1.0):
                    """CPU-only update"""
                    try:
                        sensor_inputs = self._get_enhanced_inputs(neighbors)
                        input_tensor = torch.tensor(sensor_inputs, dtype=torch.float32)
                        output_spikes = self.network(input_tensor)
                        
                        # Get output rates
                        output_rates = self.network.get_output_rates()
                        if isinstance(output_rates, torch.Tensor):
                            output_rates = output_rates.squeeze().detach().numpy()
                        else:
                            output_rates = np.array(output_rates)
                        
                        # Apply forces 
                        local_neighbors = self._get_local_neighbors(neighbors)
                        
                        if len(local_neighbors) > 0:
                            classical_separation = self._separation_force(local_neighbors)
                            classical_alignment = self._alignment_force(local_neighbors)
                            classical_cohesion = self._cohesion_force(local_neighbors)
                            
                            if len(output_rates) >= 3:
                                sep_weight = np.clip(output_rates[0], 0.2, 2.0)
                                align_weight = np.clip(output_rates[1], 0.1, 1.0)
                                cohesion_weight = np.clip(output_rates[2], 0.1, 1.0)
                            else:
                                sep_weight, align_weight, cohesion_weight = 1.5, 0.5, 0.5
                            
                            total_force = (
                                sep_weight * classical_separation +
                                align_weight * classical_alignment +
                                cohesion_weight * classical_cohesion
                            )
                        else:
                            total_force = np.array([0.0, 0.0])
                        
                        self._apply_force(total_force, dt)
                        
                    except Exception as e:
                        # Fallback
                        self.position += self.velocity * dt
            
            # Add boids
            for i in range(self.num_boids):
                x = np.random.uniform(50, self.env_width-50)
                y = np.random.uniform(50, self.env_height-50)
                vx = np.random.uniform(-1, 1)
                vy = np.random.uniform(-1, 1)
                boid = TempLIFBoid(x, y, vx, vy, i)
                env.add_boid(boid)
            
            # Run simulation
            flocking_metrics = []
            spike_activity_history = []
            
            for step in range(self.steps):
                env.update()
                
                # Collect metrics
                if step > 30 and step % 10 == 0:
                    metrics = env.get_flock_metrics()
                    flocking_metrics.append(metrics)
                
                # Sample neural activity
                if step % 15 == 0 and env.boids:
                    boid = env.boids[0]
                    output_rates = self._safe_get_output_rates(boid.network)
                    spike_activity_history.append(output_rates)
            
            # Calculate fitness
            return self._calculate_fitness(flocking_metrics, spike_activity_history)
            
        except Exception as e:
            return 0.0, {'error': str(e)}
    
    def _safe_get_output_rates(self, network):
        """Safely extract output rates"""
        try:
            output_rates = network.get_output_rates()
            if output_rates is None:
                return np.array([0.3, 0.5, 0.4])
            
            if isinstance(output_rates, torch.Tensor):
                rates_np = output_rates.squeeze().detach().numpy()
            else:
                rates_np = np.array(output_rates)
            
            if rates_np.ndim == 0:
                rates_np = np.array([rates_np.item()])
            
            if len(rates_np) < 3:
                padded = np.array([0.3, 0.5, 0.4])
                padded[:len(rates_np)] = rates_np
                rates_np = padded
            
            return rates_np[:3]
        except:
            return np.array([0.3, 0.5, 0.4])


    def _calculate_fitness(self, flocking_metrics: List[Dict], 
                            spike_history: List[np.ndarray]) -> Tuple[float, Dict]:
        """Calculate fitness score with all expected metrics"""
        if not flocking_metrics:
            return 0.0, {'error': 'no_metrics'}
        
        # Flocking metrics
        avg_cohesion = np.mean([m['cohesion'] for m in flocking_metrics])
        avg_alignment = np.mean([m['alignment'] for m in flocking_metrics])
        min_separation = min([m['separation'] for m in flocking_metrics])
        
        # Collision count
        collision_threshold = 20.0
        collisions = sum(1 for m in flocking_metrics if m['separation'] < collision_threshold)
        
        # Fitness scores
        cohesion_score = max(0, 1.0 - abs(avg_cohesion - self.target_cohesion) / max(self.target_cohesion, 1))
        alignment_score = min(avg_alignment / self.target_alignment, 1.0) if self.target_alignment > 0 else 0
        
        if min_separation < collision_threshold:
            separation_score = 0.0
        elif min_separation < self.min_separation:
            separation_score = 0.4 * (min_separation / self.min_separation)
        else:
            separation_score = 0.8 + 0.2 * min(min_separation / (self.min_separation * 1.5), 1.0)
        
        # Neural activity
        spike_fitness = 0.6  # Default
        dead_neurons = 0
        avg_rates = 0.5
        
        if spike_history:
            try:
                all_rates = []
                for rates in spike_history:
                    if isinstance(rates, np.ndarray) and rates.size > 0:
                        all_rates.extend(rates.flatten()[:3])
                
                if all_rates:
                    avg_rates = np.mean(all_rates)
                    # Counting how many neurons are consistently dead 
                    dead_count = 0
                    for i in range(3):  # 3 output neurons
                        neuron_rates = [rates[i] for rates in spike_history 
                                    if isinstance(rates, np.ndarray) and len(rates) > i]
                        if neuron_rates and np.mean(neuron_rates) < 0.02:
                            dead_count += 1
                    dead_neurons = dead_count
                    
                    # Calculate spike fitness
                    if dead_neurons == 0:
                        spike_fitness = 1.0
                    elif dead_neurons == 1:
                        spike_fitness = 0.7
                    elif dead_neurons == 2:
                        spike_fitness = 0.4
                    else:
                        spike_fitness = 0.2
            except:
                pass  # Keep defaults
        
        # Total fitness
        flocking_fitness = 0.45 * cohesion_score + 0.35 * alignment_score + 0.2 * separation_score
        stability_score = 1.0  # Assume full stability if we get here
        
        total_fitness = 0.65 * flocking_fitness + 0.25 * spike_fitness + 0.1 * stability_score
        
        # Collision penalty
        if collisions > 0:
            collision_penalty = min(collisions / len(flocking_metrics), 0.8)
            total_fitness *= (1.0 - collision_penalty)
        else:
            collision_penalty = 0.0
        
        # Build comprehensive details dictionary
        details = {
            # Core scores
            'cohesion_score': float(cohesion_score),
            'alignment_score': float(alignment_score),
            'separation_score': float(separation_score),
            'flocking_fitness': float(flocking_fitness),
            'spike_fitness': float(spike_fitness),
            'stability_score': float(stability_score),
            
            # Raw metrics
            'avg_cohesion': float(avg_cohesion),
            'avg_alignment': float(avg_alignment),
            'min_separation': float(min_separation),
            
            # Collision info
            'collisions': int(collisions),
            'collision_penalty': float(collision_penalty),
            
            # Neural info
            'dead_neurons': int(dead_neurons),
            'avg_rates': float(avg_rates),
            
            # Additional info
            'steps_completed': len(flocking_metrics) * 10  # Approximate
        }
        
        return max(0.1, total_fitness), details


class OptimizedGradualLIFTrainer:
    """Fully optimized trainer with CPU-only, parallelization, and checkpointing"""
    
    def __init__(self, population_size: int = 25, mutation_rate: float = 0.18,
                 save_dir: str = "optimized_lif_models", checkpoint_frequency: int = 5,
                 num_workers: Optional[int] = None):
        # Core parameters
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.save_dir = save_dir
        self.checkpoint_frequency = checkpoint_frequency
        
        # Set number of workers for parallel evaluation
        if num_workers is None:
            self.num_workers = min(mp.cpu_count() - 1, 8)  # Leave one CPU free
        else:
            self.num_workers = num_workers
            
   
        print(f"Parallel evaluation with {self.num_workers} workers")
        print(f"Checkpointing every {checkpoint_frequency} generations")
        
        # Create directories
        self.create_directories()
        
        # Training state
        self.best_fitness = float('-inf')
        self.best_weights = None
        self.training_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Gradual difficulty parameters
        self.base_target_cohesion = 80.0
        self.base_min_separation = 25.0
        self.base_target_alignment = 0.5
        self.final_target_cohesion = 35.0
        self.final_min_separation = 30.0
        self.final_target_alignment = 0.7
        
        # Current values
        self.target_cohesion = self.base_target_cohesion
        self.min_separation = self.base_min_separation
        self.target_alignment = self.base_target_alignment
        
        # Training history
        self.fitness_history = []
        self.generation_fitness_history = []
        self.current_boid_count = 15
        
    def create_directories(self):
        """Create necessary directories"""
        os.makedirs(self.save_dir, exist_ok=True)
        self.checkpoint_dir = os.path.join(self.save_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        print(f"Save directory: {self.save_dir}")
        print(f"Checkpoint directory: {self.checkpoint_dir}")
    
    def update_targets_for_generation(self, generation: int, total_generations: int):
        """Update difficulty targets based on progress"""
        progress = min(generation / max(total_generations * 0.8, 1), 1.0)
        
        self.target_cohesion = self.base_target_cohesion + progress * (self.final_target_cohesion - self.base_target_cohesion)
        self.min_separation = self.base_min_separation + progress * (self.final_min_separation - self.base_min_separation)
        self.target_alignment = self.base_target_alignment + progress * (self.final_target_alignment - self.base_target_alignment)
        
        if generation % 10 == 0:
            print(f" Updated targets: C<{self.target_cohesion:.1f}, A>{self.target_alignment:.2f}, S>{self.min_separation:.1f}")
    

    def get_adaptive_num_boids(self, generation: int, best_fitness: float, 
                          avg_fitness: float) -> int:
        """Gradual increase with performance gates"""
        
        # Start small
        if generation < 5:
            return 15
        elif generation < 10:
            return 20
        
        # Performance-gated progression
        if best_fitness > 0.65:  # Only increase if performing well
            # Gradual increase based on generation
            target = 20 + (generation // 10) * 5  # +5 every 10 gens
            target = min(target, 50)  # Cap at 50 for stable training
            
            # Smooth increase from current
            if target > self.current_boid_count:
                return self.current_boid_count + 3  # Small steps
        
        # If struggling, maintain current level
        return self.current_boid_count
    
    def evaluate_population_parallel(self, population: List[SimpleSNN], 
                                   num_boids: int, steps: int) -> Tuple[List[float], List[Dict]]:
        """Evaluate entire population in parallel"""
        # Calculate environment size based on boid count
        base_area = 500 * 400
        target_pixels_per_boid = base_area / 20
        total_area_needed = num_boids * target_pixels_per_boid
        scale_factor = (total_area_needed / base_area) ** 0.5
        
        env_width = max(800, min(int(500 * scale_factor), 2000))
        env_height = max(600, min(int(400 * scale_factor), 1600))
        
        # Prepare evaluation parameters
        eval_params = {
            'num_boids': num_boids,
            'steps': steps,
            'target_cohesion': self.target_cohesion,
            'target_alignment': self.target_alignment,
            'min_separation': self.min_separation,
            'env_width': env_width,
            'env_height': env_height,
            'input_size': 8,
            'hidden_size': 12,
            'output_size': 3
        }
        
        # Prepare work items
        work_items = []
        for i, network in enumerate(population):
            state_dict = network.state_dict()
            work_items.append((i, state_dict, eval_params))
        
        # Evaluate in parallel
        print(f" Evaluating {len(population)} networks in parallel with {self.num_workers} workers...")
        start_time = time.time()
        
        with mp.Pool(self.num_workers) as pool:
            results = pool.map(evaluate_network_parallel, work_items)
        
        # Sort results by index
        results.sort(key=lambda x: x[0])
        
        # Extract fitness values and details
        fitness_values = [r[1] for r in results]
        details_list = [r[2] for r in results]
        
        elapsed = time.time() - start_time
        print(f" Parallel evaluation completed in {elapsed:.1f} seconds")
        
        return fitness_values, details_list
    
    def save_checkpoint(self, generation: int, population: List[SimpleSNN],
                       fitness_values: List[float], best_idx: int):
        """Save training checkpoint"""
        checkpoint_filename = f"checkpoint_gen{generation:03d}_{self.training_id}.pkl"
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_filename)
        
        checkpoint = {
            'generation': generation,
            'population_state_dicts': [net.state_dict() for net in population],
            'fitness_values': fitness_values,
            'best_fitness': self.best_fitness,
            'best_weights_state_dict': self.best_weights.state_dict() if self.best_weights else None,
            'current_boid_count': self.current_boid_count,
            'current_targets': {
                'cohesion': self.target_cohesion,
                'alignment': self.target_alignment,
                'separation': self.min_separation
            },
            'fitness_history': self.fitness_history,
            'generation_fitness_history': self.generation_fitness_history,
            'training_params': {
                'population_size': self.population_size,
                'mutation_rate': self.mutation_rate,
                'num_workers': self.num_workers
            },
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint, f)
            
            # Save metadata JSON
            metadata = {
                'generation': generation,
                'best_fitness': float(self.best_fitness),
                'avg_fitness': float(np.mean(fitness_values)),
                'best_idx': best_idx,
                'current_boid_count': self.current_boid_count,
                'timestamp': checkpoint['timestamp']
            }
            
            metadata_path = checkpoint_path.replace('.pkl', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"  💾 Checkpoint saved: generation {generation}")
            
            # Clean up old checkpoints (keep last 3)
            self._cleanup_old_checkpoints()
            
        except Exception as e:
            print(f"  ⚠️  Failed to save checkpoint: {e}")
    
    def _cleanup_old_checkpoints(self, keep_last: int = 3):
        """Remove old checkpoints, keeping only the most recent ones"""
        checkpoints = sorted([f for f in os.listdir(self.checkpoint_dir) 
                             if f.endswith('.pkl') and f.startswith('checkpoint_')])
        
        if len(checkpoints) > keep_last:
            for old_checkpoint in checkpoints[:-keep_last]:
                try:
                    os.remove(os.path.join(self.checkpoint_dir, old_checkpoint))
                    # Also remove metadata
                    metadata = old_checkpoint.replace('.pkl', '_metadata.json')
                    if os.path.exists(os.path.join(self.checkpoint_dir, metadata)):
                        os.remove(os.path.join(self.checkpoint_dir, metadata))
                except:
                    pass
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict:
        """Load a checkpoint to resume training"""
        print(f"📂 Loading checkpoint: {checkpoint_path}")
        
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        # Restore training state
        self.current_boid_count = checkpoint['current_boid_count']
        self.fitness_history = checkpoint['fitness_history']
        self.generation_fitness_history = checkpoint['generation_fitness_history']
        
        # Restore best weights
        if checkpoint['best_weights_state_dict']:
            self.best_weights = SimpleSNN(input_size=8, hidden_size=12, output_size=3)
            self.best_weights.load_state_dict(checkpoint['best_weights_state_dict'])
            self.best_fitness = checkpoint['best_fitness']
        
        print(f"  ✅ Checkpoint loaded: generation {checkpoint['generation']}")
        print(f"     Best fitness: {self.best_fitness:.4f}")
        
        return checkpoint
    
    def find_latest_checkpoint(self) -> Optional[str]:
        """Find the most recent checkpoint"""
        checkpoints = [f for f in os.listdir(self.checkpoint_dir) 
                      if f.endswith('.pkl') and f.startswith('checkpoint_')]
        
        if not checkpoints:
            return None
        
        # Sort by generation number
        checkpoints.sort(key=lambda x: int(x.split('_')[1][3:6]))
        latest = checkpoints[-1]
        
        return os.path.join(self.checkpoint_dir, latest)
    
 

    def save_best_weights(self, network: SimpleSNN, fitness: float, metadata: Dict):
        """Save the best weights with backward-compatible metadata format"""
        self.best_weights = copy.deepcopy(network)
        self.best_fitness = fitness
        
        # Save weights
        weights_path = os.path.join(self.save_dir, "best_lif_snn_weights.pt")
        torch.save(network.state_dict(), weights_path)
        
        # Find generation from fitness history
        current_generation = len(self.fitness_history)
        
        # Save metadata 
        metadata_path = os.path.join(self.save_dir, "best_lif_snn_metadata.json")
        metadata_info = {
            # Required fields for visualizer compatibility
            "fitness": float(fitness),
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"), 
            "network_type": "LIF_SNN",  
            "architecture": {
                "input_size": 8,
                "hidden_size": 12,
                "output_size": 3
            },
            "lif_parameters": {  
                "hidden_beta": 0.7,
                "hidden_threshold": 0.5,
                "output_beta": 0.6,
                "output_threshold": 0.4
            },
            "training_version": "lif_snn_v1_fixed",  
            
            # Training details
            "generation": current_generation,
            "population_size": self.population_size,
            "method": "gradual_evolutionary_lif_snn_fixed",  
            "evaluation_steps": 300,  # Default value
            
            # Targets
            "final_targets": {
                "cohesion": self.final_target_cohesion,
                "alignment": self.final_target_alignment,
                "separation": self.final_min_separation
            },
            "current_targets": {
                "cohesion": self.target_cohesion,
                "alignment": self.target_alignment,
                "separation": self.min_separation
            },
            
            # Performance metrics - extract from metadata
            "flocking_fitness": metadata.get('flocking_fitness', fitness * 0.65),
            "spike_fitness": metadata.get('spike_fitness', 0.6),
            "stability_score": metadata.get('stability_score', 1.0),
            "cohesion_score": metadata.get('cohesion_score', 0.5),
            "alignment_score": metadata.get('alignment_score', 0.5),
            "separation_score": metadata.get('separation_score', 0.5),
            "avg_cohesion": float(metadata.get('avg_cohesion', 50.0)),
            "avg_alignment": float(metadata.get('avg_alignment', 0.5)),
            "min_separation": float(metadata.get('min_separation', 25.0)),
            
            # Additional metrics
            "bonus": 0.0,
            "penalty": 0.0,
            "collisions": int(metadata.get('collisions', 0)),
            "collision_penalty": 0.0,
            "steps_completed": 300,
            
            # Neural metrics
            "dead_neurons": int(metadata.get('dead_neurons', 0)),
            "avg_rates": float(metadata.get('avg_rates', 0.5))
        }
        
        # Handle numpy types in the metadata
        def convert_numpy_types(obj):
            if isinstance(obj, np.generic):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(v) for v in obj]
            return obj
        
        metadata_info = convert_numpy_types(metadata_info)
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata_info, f, indent=2)
        
        print(f" New best weights saved: fitness = {fitness:.4f}")
    
    def mutate_network(self, network: SimpleSNN, mutation_strength: float = 0.1):
        """Mutate network weights"""
        with torch.no_grad():
            for param in network.parameters():
                if np.random.random() < self.mutation_rate:
                    mutation = torch.randn_like(param) * mutation_strength
                    param.add_(mutation)
                    param.clamp_(-1.0, 1.0)
            
            # LIF-specific mutations
            if np.random.random() < 0.2:
                output_idx = np.random.randint(0, network.fc2.weight.shape[0])
                connections = np.random.choice(network.fc2.weight.shape[1], 2, replace=False)
                network.fc2.weight[output_idx, connections] += torch.randn(2) * 0.3
    
    def crossover_networks(self, parent1: SimpleSNN, parent2: SimpleSNN, alpha: float = 0.5) -> SimpleSNN:
        """Create offspring from two parents"""
        child = SimpleSNN(input_size=8, hidden_size=12, output_size=3)
        
        with torch.no_grad():
            for p_child, p1, p2 in zip(child.parameters(), parent1.parameters(), parent2.parameters()):
                p_child.data = alpha * p1.data + (1 - alpha) * p2.data
        
        return child
    
    def train(self, generations: int = 100, evaluation_steps: int = 700,
              resume_from_checkpoint: bool = True) -> Dict:
        """Main training loop with all optimizations"""
        print(f"\nStarting Optimized LIF SNN Training")
        print(f"Generations: {generations}")
        print(f"Population: {self.population_size}")
        print(f"Evaluation steps: {evaluation_steps}")
        print(f"Parallel workers: {self.num_workers}")
        print(f"Checkpoint frequency: every {self.checkpoint_frequency} generations")
        
        # Check for existing checkpoint
        start_generation = 0
        population = None
        
        if resume_from_checkpoint:
            latest_checkpoint = self.find_latest_checkpoint()
            if latest_checkpoint:
                print(f"\nFound checkpoint, resuming training...")
                checkpoint = self.load_checkpoint(latest_checkpoint)
                start_generation = checkpoint['generation'] + 1
                
                # Restore population
                population = []
                for state_dict in checkpoint['population_state_dicts']:
                    net = SimpleSNN(input_size=8, hidden_size=12, output_size=3)
                    net.load_state_dict(state_dict)
                    population.append(net)
        
        # Create initial population if needed
        if population is None:
            population = []
            for i in range(self.population_size):
                network = SimpleSNN(input_size=8, hidden_size=12, output_size=3)
                
                # Initialize with good starting weights
                with torch.no_grad():
                    for param in network.parameters():
                        param.data = torch.randn_like(param) * 0.3
                    
                    network.fc1.bias.data = torch.abs(network.fc1.bias.data) * 0.4 + 0.3
                    network.fc2.bias.data = torch.abs(network.fc2.bias.data) * 0.2 + 0.2
                
                population.append(network)
        
        # Training loop
        total_start_time = time.time()
        
        for generation in range(start_generation, generations):
            gen_start_time = time.time()
            
            # Update targets
            self.update_targets_for_generation(generation, generations)
            
            # Update boid count
            if generation > 0 and self.generation_fitness_history:
                prev_best = max(self.generation_fitness_history[-1])
                prev_avg = np.mean(self.generation_fitness_history[-1])
                self.current_boid_count = self.get_adaptive_num_boids(
                    generation, prev_best, prev_avg
                )
            
            print(f"\nGeneration {generation + 1}/{generations}")
            print(f"Boid count: {self.current_boid_count}")
            
            # Parallel evaluation
            fitness_values, details_list = self.evaluate_population_parallel(
                population, self.current_boid_count, evaluation_steps
            )
            
            # Track results
            self.generation_fitness_history.append(fitness_values)
            
            # Find best
            best_idx = np.argmax(fitness_values)
            best_fitness = fitness_values[best_idx]
            best_details = details_list[best_idx]
            avg_fitness = np.mean(fitness_values)
            
            print(f"Results:")
            print(f"Best fitness: {best_fitness:.4f}")
            print(f"Average fitness: {avg_fitness:.4f}")
            print(f"Min separation: {best_details.get('min_separation', 0):.1f}")
            print(f"Collisions: {best_details.get('collisions', 0)}")
            
            # Track history
            self.fitness_history.append({
                'generation': generation + 1,
                'best_fitness': best_fitness,
                'avg_fitness': avg_fitness,
                'best_details': best_details
            })
            
            # Save best if improved
            if best_fitness > self.best_fitness:
                self.save_best_weights(population[best_idx], best_fitness, best_details)
            
            # Checkpoint
            if (generation + 1) % self.checkpoint_frequency == 0:
                self.save_checkpoint(generation, population, fitness_values, best_idx)
            
            # Create next generation
            if generation < generations - 1:
                new_population = []
                
                # Elitism
                sorted_indices = np.argsort(fitness_values)[::-1]
                elite_count = max(2, self.population_size // 4)
                
                for i in range(elite_count):
                    new_population.append(copy.deepcopy(population[sorted_indices[i]]))
                
                # Generate offspring
                while len(new_population) < self.population_size:
                    # Tournament selection
                    tournament_size = min(5, len(fitness_values))
                    tournament_indices = np.random.choice(len(fitness_values), tournament_size, replace=False)
                    tournament_fitness = [fitness_values[i] for i in tournament_indices]
                    
                    parent1_idx = tournament_indices[np.argmax(tournament_fitness)]
                    parent2_idx = tournament_indices[np.argsort(tournament_fitness)[-2]]
                    
                    # Crossover
                    alpha = np.random.uniform(0.3, 0.7)
                    offspring = self.crossover_networks(
                        population[parent1_idx],
                        population[parent2_idx],
                        alpha
                    )
                    
                    # Mutation
                    mutation_strength = 0.15 if avg_fitness < 0.3 else 0.1
                    self.mutate_network(offspring, mutation_strength)
                    
                    new_population.append(offspring)
                
                population = new_population
            
            # Time tracking
            gen_time = time.time() - gen_start_time
            print(f"Generation time: {gen_time:.1f} seconds")
            
            # Early stopping
            if best_fitness > 0.95:
                print(f"\nEarly stopping - excellent fitness achieved!")
                break
        
        # Training complete
        total_time = time.time() - total_start_time
        print(f"\nTraining completed!")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Best fitness: {self.best_fitness:.4f}")
        print(f"Final generation: {generation + 1}")
        self.save_training_analytics()
    
     
        generation_details_file = os.path.join(self.save_dir, f"generation_details_{self.training_id}.pkl")
        with open(generation_details_file, 'wb') as f:
            pickle.dump({
                'generation_fitness': self.generation_fitness_history,
                'fitness_history': self.fitness_history,
                'boid_count_history': [self.current_boid_count] * len(self.fitness_history)  # Track boid counts
            }, f)
        
        return {
            'best_fitness': self.best_fitness,
            'best_weights': self.best_weights,
            'fitness_history': self.fitness_history,
            'generations_completed': generation + 1 - start_generation,
            'total_time': total_time,
            'final_boid_count': self.current_boid_count,
            'analytics_saved': True
        }
    
    def save_training_analytics(self):
        """Save comprehensive training analytics"""
        analytics_file = os.path.join(self.save_dir, f"training_analytics_{self.training_id}.pkl")
        
        analytics_data = {
            'training_id': self.training_id,
            'hyperparameters': {
                'population_size': self.population_size,
                'mutation_rate': self.mutation_rate,
                'num_workers': self.num_workers,
                'architecture': '8→12→3',
                'initial_boid_count': 15,
                'base_targets': {
                    'cohesion': self.base_target_cohesion,
                    'separation': self.base_min_separation,
                    'alignment': self.base_target_alignment
                },
                'final_targets': {
                    'cohesion': self.final_target_cohesion,
                    'separation': self.final_min_separation,
                    'alignment': self.final_target_alignment
                }
            },
            'fitness_history': self.fitness_history,
            'generation_fitness_history': self.generation_fitness_history,
            'population_diversity': self._calculate_population_diversity(),
            'convergence_metrics': self._calculate_convergence_metrics(),
            'neural_health': self._analyze_neural_health()
        }
        
        with open(analytics_file, 'wb') as f:
            pickle.dump(analytics_data, f)
        
        # Also save a summary JSON
        summary_file = os.path.join(self.save_dir, f"training_summary_{self.training_id}.json")
        summary_data = {
            'training_id': self.training_id,
            'best_fitness': float(self.best_fitness),
            'generations_completed': len(self.fitness_history),
            'final_boid_count': self.current_boid_count,
            'total_evaluations': self.population_size * len(self.fitness_history),
            'convergence_generation': self._find_convergence_point(),
            'architecture': '8→12→3',
            'mutation_rate': self.mutation_rate
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        print(f"📊 Training analytics saved to {analytics_file}")
    
    def _calculate_population_diversity(self) -> Dict:
        """Calculate diversity metrics for the population"""
        if not self.generation_fitness_history:
            return {}
        
        diversity_metrics = []
        for gen_fitness in self.generation_fitness_history:
            if gen_fitness:
                diversity_metrics.append({
                    'mean': np.mean(gen_fitness),
                    'std': np.std(gen_fitness),
                    'min': min(gen_fitness),
                    'max': max(gen_fitness),
                    'range': max(gen_fitness) - min(gen_fitness),
                    'coefficient_of_variation': np.std(gen_fitness) / np.mean(gen_fitness) if np.mean(gen_fitness) > 0 else 0
                })
        
        return diversity_metrics
    

    def _calculate_convergence_metrics(self) -> Dict:
        """Calculate convergence metrics for the training process"""
        
        if len(self.fitness_history) < 5:
            return {
                'converged': False,
                'convergence_generation': -1,
                'improvement_rates': [],
                'plateau_points': [],
                'stability_score': 0.0
            }
        
        # Extract best fitness history
        best_fitness_history = [h['best_fitness'] for h in self.fitness_history]
        avg_fitness_history = [h['avg_fitness'] for h in self.fitness_history]
        
        # Calculate improvement rates between generations
        improvement_rates = []
        for i in range(1, len(best_fitness_history)):
            rate = best_fitness_history[i] - best_fitness_history[i-1]
            improvement_rates.append(rate)
        
        # Find plateau points (where improvement < threshold)
        plateau_threshold = 0.01  # Less than 1% improvement
        plateau_points = []
        for i, rate in enumerate(improvement_rates):
            if abs(rate) < plateau_threshold:
                plateau_points.append(i + 1)  # +1 because we start from generation 1
        
        # Find convergence point (5 consecutive generations with minimal improvement)
        convergence_generation = -1
        window_size = 5
        
        if len(best_fitness_history) >= window_size:
            for i in range(window_size, len(best_fitness_history)):
                window = best_fitness_history[i-window_size:i]
                window_improvement = max(window) - min(window)
                
                if window_improvement < plateau_threshold * window_size:
                    convergence_generation = i - window_size + 1
                    break
        
        # Calculate stability score 
        stability_score = 0.0
        if convergence_generation > 0:
            post_convergence = best_fitness_history[convergence_generation:]
            if len(post_convergence) > 1:
                stability_score = 1.0 - np.std(post_convergence)
                stability_score = max(0.0, stability_score)  # Ensure non-negative
        
        # Calculate fitness variance over time
        fitness_variance_over_time = []
        if self.generation_fitness_history:
            for gen_fitnesses in self.generation_fitness_history:
                if gen_fitnesses and len(gen_fitnesses) > 0:
                    variance = np.var(gen_fitnesses)
                    fitness_variance_over_time.append(variance)
        
        # Determine if truly converged
        converged = convergence_generation > 0
        
        # Calculate average improvement rate
        avg_improvement_rate = np.mean(improvement_rates) if improvement_rates else 0.0
        
        # Find the generation with maximum fitness
        max_fitness_gen = np.argmax(best_fitness_history) + 1 if best_fitness_history else 0
        
        # Calculate convergence speed 
        final_fitness = best_fitness_history[-1] if best_fitness_history else 0
        target_fitness = final_fitness * 0.9
        convergence_speed = -1
        
        for i, fitness in enumerate(best_fitness_history):
            if fitness >= target_fitness:
                convergence_speed = i + 1
                break
        
        return {
            'converged': converged,
            'convergence_generation': convergence_generation,
            'convergence_speed': convergence_speed,  # Generations to 90% of final
            'improvement_rates': improvement_rates,
            'avg_improvement_rate': float(avg_improvement_rate),
            'plateau_points': plateau_points,
            'first_plateau': plateau_points[0] if plateau_points else -1,
            'stability_score': float(stability_score),
            'fitness_variance_over_time': fitness_variance_over_time,
            'max_fitness_generation': max_fitness_gen,
            'final_fitness': float(best_fitness_history[-1]) if best_fitness_history else 0.0,
            'fitness_improvement': float(best_fitness_history[-1] - best_fitness_history[0]) if len(best_fitness_history) > 1 else 0.0
        }
    
    def _analyze_neural_health(self) -> Dict:
        """Analyze neural network health metrics"""
        if not self.fitness_history:
            return {}
        
        # Extract neural metrics from best individuals
        dead_neuron_history = []
        spike_fitness_history = []
        
        for gen_data in self.fitness_history:
            if 'best_details' in gen_data:
                details = gen_data['best_details']
                dead_neuron_history.append(details.get('dead_neurons', 0))
                spike_fitness_history.append(details.get('spike_fitness', 0))
        
        return {
            'dead_neuron_history': dead_neuron_history,
            'spike_fitness_history': spike_fitness_history,
            'avg_dead_neurons': np.mean(dead_neuron_history) if dead_neuron_history else 0,
            'final_dead_neurons': dead_neuron_history[-1] if dead_neuron_history else 0
        }
    
    def _find_convergence_point(self) -> int:
        """Find the generation where fitness converged"""
        if len(self.fitness_history) < 10:
            return -1
        
        best_fitness_history = [h['best_fitness'] for h in self.fitness_history]
        
        # Look for point where improvement becomes minimal
        window_size = 5
        threshold = 0.02
        
        for i in range(window_size, len(best_fitness_history)):
            window = best_fitness_history[i-window_size:i]
            improvement = max(window) - min(window)
            if improvement < threshold:
                return i
        
        return len(best_fitness_history)



def main():
    """Main function to run optimized training"""
    print("LIF SNN Training System")
    print("=" * 70)
    
    # Training options
    print("\nOptions:")
    print("1. Quick test (10 generations, 300 steps)")
    print("2. Standard training (100 generations, 500 steps)")
    print("3. Extended training (100 generations, 700 steps)")
    print("4. Custom training settings")
    print("5. Run comprehensive evaluation on trained model")  # NEW OPTION
    print("6. Compare mutation rates experiment")  # NEW OPTION
    
    choice = input("\nEnter choice (1-6): ").strip()
    
    if choice in ['1', '2', '3', '4']:
        # EXISTING TRAINING CODE
        if choice == '1':
            generations = 10
            steps = 300
            checkpoint_freq = 10
        elif choice == '2':
            generations = 100
            steps = 500
            checkpoint_freq = 5
        elif choice == '3':
            generations = 100
            steps = 700
            checkpoint_freq = 5
        elif choice == '4':
            generations = int(input("Enter number of generations: "))
            steps = int(input("Enter evaluation steps: "))
            checkpoint_freq = int(input("Enter checkpoint frequency: "))
        
        # Ask about workers
        num_workers = int(input(f"\nNumber of parallel workers (1-{mp.cpu_count()}, default={min(mp.cpu_count()-1, 8)}): ") or min(mp.cpu_count()-1, 8))
        
        # Ask about resuming
        resume = input("\nResume from checkpoint if available? (y/n, default=y): ").strip().lower() != 'n'
        
        print(f"\nStarting training with:")
        print(f"Generations: {generations}")
        print(f"Evaluation steps: {steps}")
        print(f"Checkpoint frequency: every {checkpoint_freq} generations")
        print(f"Parallel workers: {num_workers}")
        print(f"Resume from checkpoint: {resume}")
        
        # Create trainer
        trainer = OptimizedGradualLIFTrainer(
            population_size=25,
            mutation_rate=0.18,
            checkpoint_frequency=checkpoint_freq,
            num_workers=num_workers
        )
        
        # Train
        results = trainer.train(
            generations=generations,
            evaluation_steps=steps,
            resume_from_checkpoint=resume
        )
        
        print("\nTraining complete!")
        print(f"Best fitness achieved: {results['best_fitness']:.4f}")
        print(f"Time taken: {results['total_time']/60:.1f} minutes")
        print(f"Generations completed: {results['generations_completed']}")
        
        # Performance summary
        if results['best_fitness'] > 0.8:
            print("\nExcellent performance! Your LIF SNN should exhibit great flocking behavior!")
        elif results['best_fitness'] > 0.6:
            print("\nGood performance! Your LIF SNN shows decent flocking behavior.")
        else:
            print("\nModerate performance. Consider training for more generations.")
        
  
        if results['best_fitness'] > 0.5:
            run_eval = input("\n🔬 Run comprehensive evaluation on trained model? (y/n): ").strip().lower()
            if run_eval == 'y':
                run_comprehensive_evaluation_suite()
    
    elif choice == '5':
        run_comprehensive_evaluation_suite()
    
    elif choice == '6':
        run_mutation_rate_experiment_menu()
    
    else:
        print("Invalid choice, exiting.")


def run_comprehensive_evaluation_suite():
    """Run comprehensive evaluation on trained model"""
    print("\n" + "=" * 70)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("=" * 70)
    
    # Check if trained model exists
    weights_path = "optimized_lif_models/best_lif_snn_weights.pt"
    if not os.path.exists(weights_path):
        print("No trained model found! Please train a model first (options 1-4).")
        return
    
    # Load metadata to show model info
    metadata_path = "optimized_lif_models/best_lif_snn_metadata.json"
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"Model Info:")
        print(f"Fitness: {metadata.get('fitness', 'unknown'):.4f}")
        print(f"Generation: {metadata.get('generation', 'unknown')}")
        print(f"Architecture: {metadata.get('architecture', {})}")
    
    print("\nEvaluation Options:")
    print("1. Full evaluation suite (all tests)")
    print("2. Scalability analysis only")
    print("3. Collision analysis only")
    print("4. Neural dynamics analysis only")
    print("5. Computational efficiency only")
    
    eval_choice = input("\nEnter choice (1-5): ").strip()
    
    # Create evaluator instance 
    evaluator = ComprehensiveEvaluator()
    
    # Load trained network
    network = SimpleSNN(input_size=8, hidden_size=12, output_size=3)
    network.load_state_dict(torch.load(weights_path, map_location='cpu'))
    network.eval()
    
    if eval_choice == '1':
        # Full evaluation
        print("\nRunning full evaluation suite (this may take several minutes)...")
        
        # 1. Scalability
        print("\n[1/4] Evaluating scalability...")
        scalability_results = evaluator.evaluate_scalability(
            network,
            boid_counts=[15, 25, 35, 50, 75, 100],
            steps_per_eval=500,
            trials_per_count=3
        )
        
        # 2. Collisions
        print("\n[2/4] Analyzing collision patterns...")
        collision_results = evaluator.evaluate_collision_patterns(
            network,
            num_boids=50,
            steps=1000,
            detailed_tracking=True
        )
        
        # 3. Neural dynamics
        print("\n[3/4] Analyzing neural dynamics...")
        neural_results = evaluator.analyze_neural_dynamics(
            network,
            test_scenarios=5,
            steps_per_scenario=200
        )
        
        # 4. Efficiency
        print("\n[4/4] Evaluating computational efficiency...")
        efficiency_results = evaluator.evaluate_computational_efficiency(
            network,
            boid_counts=[10, 25, 50, 100],
            steps=100
        )
        
    elif eval_choice == '2':
        print("\nRunning scalability analysis...")
        scalability_results = evaluator.evaluate_scalability(
            network,
            boid_counts=[15, 25, 35, 50, 75, 100],
            steps_per_eval=500,
            trials_per_count=3
        )
        
    elif eval_choice == '3':
        print("\nRunning collision analysis...")
        num_boids = int(input("Number of boids for collision test (default=50): ") or 50)
        collision_results = evaluator.evaluate_collision_patterns(
            network,
            num_boids=num_boids,
            steps=1000,
            detailed_tracking=True
        )
        
    elif eval_choice == '4':
        print("\nRunning neural dynamics analysis...")
        neural_results = evaluator.analyze_neural_dynamics(
            network,
            test_scenarios=5,
            steps_per_scenario=200
        )
        
    elif eval_choice == '5':
        print("\nRunning computational efficiency analysis...")
        efficiency_results = evaluator.evaluate_computational_efficiency(
            network,
            boid_counts=[10, 25, 50, 100],
            steps=100
        )
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print(f"Results saved to: {evaluator.save_dir}/")
  


def run_mutation_rate_experiment_menu():
    """Run mutation rate comparison experiment"""
    print("\n" + "=" * 70)
    print("MUTATION RATE COMPARISON EXPERIMENT")
    print("=" * 70)
    
    # Define mutation rates to test
    mutation_rates = [0.10, 0.15, 0.18, 0.20, 0.25]
    
    print(f"\nAvailable mutation rates: {mutation_rates}")
    print("\nOptions:")
    print("1. Test mutation rate 0.10")
    print("2. Test mutation rate 0.15")
    print("3. Test mutation rate 0.18 (default)")
    print("4. Test mutation rate 0.20")
    print("5. Test mutation rate 0.25")
    print("6. Custom mutation rate")
    
    choice = input("\nEnter choice (1-6): ").strip()
    
    if choice in ['1', '2', '3', '4', '5']:
        selected_rate = mutation_rates[int(choice) - 1]
    elif choice == '6':
        selected_rate = float(input("Enter custom mutation rate (0.0-1.0): "))
    else:
        print("Invalid choice")
        return
    
    generations = int(input(f"\nNumber of generations (default=50): ") or 50)
    population_size = int(input(f"Population size (default=25): ") or 25)
    
    print(f"\nStarting experiment with mutation rate: {selected_rate}")
    
    # Create trainer with specific mutation rate
    save_dir = f"mutation_exp_{selected_rate:.2f}"
    trainer = OptimizedGradualLIFTrainer(
        population_size=population_size,
        mutation_rate=selected_rate,
        save_dir=save_dir,
        checkpoint_frequency=10,
        num_workers=4
    )
    
    # Run training
    results = trainer.train(
        generations=generations,
        evaluation_steps=500,
        resume_from_checkpoint=False  # Start fresh for fair comparison
    )
    
    # Save experiment metadata
    experiment_data = {
        'mutation_rate': selected_rate,
        'generations': generations,
        'population_size': population_size,
        'best_fitness': float(results['best_fitness']),
        'generations_completed': results['generations_completed'],
        'final_boid_count': results['final_boid_count'],
        'fitness_progression': [float(h['best_fitness']) for h in trainer.fitness_history],
        'avg_fitness_progression': [float(h['avg_fitness']) for h in trainer.fitness_history]
    }
    
    # Save experiment data
    experiment_file = os.path.join(save_dir, f"mutation_experiment_{selected_rate:.2f}.json")
    with open(experiment_file, 'w') as f:
        json.dump(experiment_data, f, indent=2)
    
    print(f"\nMutation rate {selected_rate} experiment complete")
    print(f"   Best fitness: {results['best_fitness']:.4f}")
    print(f"   Data saved to: {save_dir}/")

if __name__ == "__main__":
    if IMPORTS_OK:
        main()
    else:
        print("Cannot run - import errors")
        print("Please ensure all required modules are in the correct location")