

import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class NeuralArchitectureComparison:
    """Compare neural network architectures with different hidden layer sizes."""
    
    def __init__(self, data_paths: Dict[int, Dict[str, str]]):
        """
        Initialize the comparison with paths to data files for each architecture.
        
        
        Args:
            data_paths: Dictionary mapping hidden nodes to file paths
                       {8: {'neural': 'path', 'scalability': 'path', 'collision': 'path'}, ...}
        """
        self.data_paths = data_paths
        self.architectures = list(data_paths.keys())
        self.data = {}
        self.load_all_data()
        
    def diagnose_data_structure(self):
        """Diagnose and print the structure of loaded data for debugging."""
        print("\n" + "="*80)
        print(" DATA STRUCTURE DIAGNOSIS ".center(80))
        print("="*80)
        
        for hidden_nodes in self.architectures:
            print(f"\n{hidden_nodes} Hidden Nodes Architecture:")
            print("-"*40)
            
            # Neural data
            neural_data = self.data[hidden_nodes]['neural']
            print(f"  Neural data type: {type(neural_data)}")
            if neural_data:
                if isinstance(neural_data, dict):
                    print(f"    Keys: {list(neural_data.keys())[:5]}...")
                    if 'scenarios' in neural_data:
                        print(f"    Scenarios type: {type(neural_data['scenarios'])}")
                        if isinstance(neural_data['scenarios'], dict):
                            print(f"    Scenario names: {list(neural_data['scenarios'].keys())}")
                elif isinstance(neural_data, list):
                    print(f"    List length: {len(neural_data)}")
                    if neural_data:
                        print(f"    First item type: {type(neural_data[0])}")
            
            # Scalability data
            scale_data = self.data[hidden_nodes]['scalability']
            print(f"  Scalability data type: {type(scale_data)}")
            if scale_data and isinstance(scale_data, dict):
                print(f"    Keys: {list(scale_data.keys())[:5]}...")
            
            # Collision data
            coll_data = self.data[hidden_nodes]['collision']
            print(f"  Collision data type: {type(coll_data)}")
            if coll_data and isinstance(coll_data, dict):
                print(f"    Keys: {list(coll_data.keys())[:5]}...")
    
    def load_all_data(self):
        """Load all data files for each architecture."""
        for hidden_nodes in self.architectures:
            self.data[hidden_nodes] = {
                'neural': self.load_pickle(self.data_paths[hidden_nodes]['neural']),
                'scalability': self.load_json_or_pickle(self.data_paths[hidden_nodes]['scalability']),
                'collision': self.load_pickle(self.data_paths[hidden_nodes]['collision'])
            }
            print(f"Loaded data for {hidden_nodes} hidden nodes architecture")
        
        # Run diagnosis after loading
        self.diagnose_data_structure()
    
    def load_pickle(self, filepath: str):
        """Load pickle file."""
        try:
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def load_json_or_pickle(self, filepath: str):
        """Try to load JSON first, then pickle if JSON fails."""
        json_path = filepath.replace('.pkl', '.json')
        try:
            with open(json_path, 'r') as f:
                return json.load(f)
        except:
            return self.load_pickle(filepath)
    
    def analyze_neural_dynamics(self) -> pd.DataFrame:
        """Analyze neural dynamics across architectures."""
        results = []
        
        for hidden_nodes in self.architectures:
            neural_data = self.data[hidden_nodes]['neural']
            if not neural_data:
                continue
            
            # Handle neural_data that might be a dict or have scenarios
            if isinstance(neural_data, dict) and 'scenarios' in neural_data:
                scenarios = neural_data['scenarios']
                # Handle both dict and list formats
                if isinstance(scenarios, dict):
                    scenarios_to_process = scenarios.items()
                elif isinstance(scenarios, list):
                    # If it's a list, create enumerated pairs
                    scenarios_to_process = enumerate(scenarios)
                else:
                    print(f"Unknown scenarios format for {hidden_nodes} hidden nodes")
                    continue
                    
                for scenario_name, scenario_data in scenarios_to_process:
                    if isinstance(scenario_data, dict):
                        spike_rates = scenario_data.get('spike_rates', [])
                        dead_neurons = scenario_data.get('dead_neurons_count', [])
                        correlations = scenario_data.get('correlations', {})
                    else:
                        spike_rates = getattr(scenario_data, 'spike_rates', [])
                        dead_neurons = getattr(scenario_data, 'dead_neurons_count', [])
                        correlations = getattr(scenario_data, 'correlations', {})
                    
                    # Calculate metrics
                    if spike_rates:
                        avg_spike_rate = np.mean([np.mean(sr) for sr in spike_rates if len(sr) > 0])
                        std_spike_rate = np.std([np.mean(sr) for sr in spike_rates if len(sr) > 0])
                    else:
                        avg_spike_rate = 0
                        std_spike_rate = 0
                    
                    if dead_neurons:
                        avg_dead = np.mean(dead_neurons)
                        max_dead = np.max(dead_neurons)
                    else:
                        avg_dead = 0
                        max_dead = 0
                    
                    avg_correlation = np.mean(list(correlations.values())) if correlations else 0
                    
                    results.append({
                        'Architecture': f"{hidden_nodes} Hidden",
                        'Scenario': str(scenario_name),
                        'Avg_Spike_Rate': avg_spike_rate,
                        'Std_Spike_Rate': std_spike_rate,
                        'Avg_Dead_Neurons': avg_dead,
                        'Max_Dead_Neurons': max_dead,
                        'Avg_Correlation': avg_correlation
                    })
            
            elif isinstance(neural_data, list):
                # Process as a single scenario
                spike_rates = []
                dead_neurons = []
                for item in neural_data:
                    if isinstance(item, dict):
                        if 'spike_rates' in item:
                            spike_rates.extend(item['spike_rates'])
                        if 'dead_neurons_count' in item:
                            dead_neurons.extend(item['dead_neurons_count'])
                
                if spike_rates:
                    avg_spike_rate = np.mean(spike_rates)
                    std_spike_rate = np.std(spike_rates)
                else:
                    avg_spike_rate = 0
                    std_spike_rate = 0
                    
                avg_dead = np.mean(dead_neurons) if dead_neurons else 0
                max_dead = np.max(dead_neurons) if dead_neurons else 0
                
                results.append({
                    'Architecture': f"{hidden_nodes} Hidden",
                    'Scenario': 'default',
                    'Avg_Spike_Rate': avg_spike_rate,
                    'Std_Spike_Rate': std_spike_rate,
                    'Avg_Dead_Neurons': avg_dead,
                    'Max_Dead_Neurons': max_dead,
                    'Avg_Correlation': 0
                })
            else:
                print(f"Unknown neural_data structure for {hidden_nodes} hidden nodes: {type(neural_data)}")
        
        return pd.DataFrame(results)
    
    def analyze_scalability(self) -> pd.DataFrame:
        """Analyze scalability metrics across architectures."""
        results = []
        
        for hidden_nodes in self.architectures:
            scalability_data = self.data[hidden_nodes]['scalability']
            if not scalability_data:
                continue
            
            # Handle both dict and object formats
            if isinstance(scalability_data, dict):
                results_data = scalability_data.get('results', {})
            else:
                results_data = getattr(scalability_data, 'results', {})
            
            for boid_count, metrics in results_data.items():
                # Convert string keys to int 
                boid_count = int(boid_count) if isinstance(boid_count, str) else boid_count
                
                results.append({
                    'Architecture': f"{hidden_nodes} Hidden",
                    'Boid_Count': boid_count,
                    'Mean_Fitness': metrics.get('mean_fitness', 0),
                    'Std_Fitness': metrics.get('std_fitness', 0),
                    'Mean_Collisions': metrics.get('mean_collisions', 0),
                    'Collision_Rate': metrics.get('mean_collision_rate', 0),
                    'Eval_Time': metrics.get('mean_eval_time', 0),
                    'Mean_Cohesion': metrics.get('mean_cohesion', 0),
                    'Mean_Alignment': metrics.get('mean_alignment', 0),
                    'Mean_Separation': metrics.get('mean_separation', 0)
                })
        
        return pd.DataFrame(results)
    
    def analyze_collisions(self) -> pd.DataFrame:
        """Analyze collision patterns across architectures."""
        results = []
        
        for hidden_nodes in self.architectures:
            collision_data = self.data[hidden_nodes]['collision']
            if not collision_data:
                continue
            
            summary = collision_data.get('summary', {})
            
            # Time series analysis
            collision_counts = collision_data.get('collision_counts', [])
            near_miss_counts = collision_data.get('near_miss_counts', [])
            min_distances = collision_data.get('min_distances', [])
            
            # Calculate statistics
            if collision_counts:
                collision_trend = np.polyfit(range(len(collision_counts)), collision_counts, 1)[0]
            else:
                collision_trend = 0
            
            results.append({
                'Architecture': f"{hidden_nodes} Hidden",
                'Total_Collisions': summary.get('total_collisions', 0),
                'Avg_Collisions_Per_Step': summary.get('avg_collisions_per_step', 0),
                'Max_Collisions_Per_Step': summary.get('max_collisions_per_step', 0),
                'Collision_Free_Steps': summary.get('collision_free_steps', 0),
                'Avg_Min_Distance': summary.get('avg_min_distance', 0),
                'Total_Near_Misses': summary.get('total_near_misses', 0),
                'Collision_Trend': collision_trend,
                'Mean_Near_Misses': np.mean(near_miss_counts) if near_miss_counts else 0,
                'Min_Distance_Std': np.std(min_distances) if min_distances else 0
            })
        
        return pd.DataFrame(results)
    
    def plot_comparison(self):
        """Create comprehensive comparison plots."""
        # Set up the plotting style
        plt.style.use('seaborn-v0_8-darkgrid')
        fig = plt.figure(figsize=(20, 12))
        
        # Get data
        neural_df = self.analyze_neural_dynamics()
        scalability_df = self.analyze_scalability()
        collision_df = self.analyze_collisions()
        
        # 1. Neural Dynamics Comparison
        ax1 = plt.subplot(3, 4, 1)
        if not neural_df.empty:
            neural_summary = neural_df.groupby('Architecture')['Avg_Spike_Rate'].mean()
            neural_summary.plot(kind='bar', ax=ax1, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            ax1.set_title('Average Spike Rate by Architecture', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Architecture')
            ax1.set_ylabel('Spike Rate')
            ax1.tick_params(axis='x', rotation=45)
        
        # 2. Dead Neurons Comparison
        ax2 = plt.subplot(3, 4, 2)
        if not neural_df.empty:
            dead_neurons = neural_df.groupby('Architecture')['Avg_Dead_Neurons'].mean()
            dead_neurons.plot(kind='bar', ax=ax2, color=['#95E1D3', '#F38181', '#FCE38A'])
            ax2.set_title('Average Dead Neurons', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Architecture')
            ax2.set_ylabel('Dead Neurons')
            ax2.tick_params(axis='x', rotation=45)
        
        # 3. Fitness vs Boid Count
        ax3 = plt.subplot(3, 4, 3)
        if not scalability_df.empty:
            for arch in scalability_df['Architecture'].unique():
                arch_data = scalability_df[scalability_df['Architecture'] == arch]
                ax3.plot(arch_data['Boid_Count'], arch_data['Mean_Fitness'], 
                        marker='o', label=arch, linewidth=2)
            ax3.set_title('Fitness vs Boid Count', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Number of Boids')
            ax3.set_ylabel('Mean Fitness')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Collision Rate vs Boid Count
        ax4 = plt.subplot(3, 4, 4)
        if not scalability_df.empty:
            for arch in scalability_df['Architecture'].unique():
                arch_data = scalability_df[scalability_df['Architecture'] == arch]
                ax4.plot(arch_data['Boid_Count'], arch_data['Collision_Rate'], 
                        marker='s', label=arch, linewidth=2)
            ax4.set_title('Collision Rate vs Boid Count', fontsize=12, fontweight='bold')
            ax4.set_xlabel('Number of Boids')
            ax4.set_ylabel('Collision Rate')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # 5. Evaluation Time Scaling
        ax5 = plt.subplot(3, 4, 5)
        if not scalability_df.empty:
            for arch in scalability_df['Architecture'].unique():
                arch_data = scalability_df[scalability_df['Architecture'] == arch]
                ax5.plot(arch_data['Boid_Count'], arch_data['Eval_Time'], 
                        marker='^', label=arch, linewidth=2)
            ax5.set_title('Computation Time Scaling', fontsize=12, fontweight='bold')
            ax5.set_xlabel('Number of Boids')
            ax5.set_ylabel('Evaluation Time (s)')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # 6. Total Collisions Comparison
        ax6 = plt.subplot(3, 4, 6)
        if not collision_df.empty:
            collision_df.plot(x='Architecture', y='Total_Collisions', kind='bar', 
                            ax=ax6, color=['#A8E6CF', '#FFD3B6', '#FFAAA5'])
            ax6.set_title('Total Collisions (50 Boids)', fontsize=12, fontweight='bold')
            ax6.set_xlabel('Architecture')
            ax6.set_ylabel('Total Collisions')
            ax6.tick_params(axis='x', rotation=45)
            ax6.legend().remove()
        
        # 7. Cohesion Performance
        ax7 = plt.subplot(3, 4, 7)
        if not scalability_df.empty:
            for arch in scalability_df['Architecture'].unique():
                arch_data = scalability_df[scalability_df['Architecture'] == arch]
                ax7.plot(arch_data['Boid_Count'], arch_data['Mean_Cohesion'], 
                        marker='D', label=arch, linewidth=2)
            ax7.set_title('Cohesion Performance', fontsize=12, fontweight='bold')
            ax7.set_xlabel('Number of Boids')
            ax7.set_ylabel('Mean Cohesion Distance')
            ax7.legend()
            ax7.grid(True, alpha=0.3)
        
        # 8. Alignment Performance
        ax8 = plt.subplot(3, 4, 8)
        if not scalability_df.empty:
            for arch in scalability_df['Architecture'].unique():
                arch_data = scalability_df[scalability_df['Architecture'] == arch]
                ax8.plot(arch_data['Boid_Count'], arch_data['Mean_Alignment'], 
                        marker='p', label=arch, linewidth=2)
            ax8.set_title('Alignment Performance', fontsize=12, fontweight='bold')
            ax8.set_xlabel('Number of Boids')
            ax8.set_ylabel('Mean Alignment')
            ax8.legend()
            ax8.grid(True, alpha=0.3)
        
        # 9. Fitness Stability (Standard Deviation)
        ax9 = plt.subplot(3, 4, 9)
        if not scalability_df.empty:
            for arch in scalability_df['Architecture'].unique():
                arch_data = scalability_df[scalability_df['Architecture'] == arch]
                ax9.plot(arch_data['Boid_Count'], arch_data['Std_Fitness'], 
                        marker='*', label=arch, linewidth=2, markersize=10)
            ax9.set_title('Fitness Stability (Lower is Better)', fontsize=12, fontweight='bold')
            ax9.set_xlabel('Number of Boids')
            ax9.set_ylabel('Fitness Std Dev')
            ax9.legend()
            ax9.grid(True, alpha=0.3)
        
        # 10. Collision-Free Steps
        ax10 = plt.subplot(3, 4, 10)
        if not collision_df.empty:
            collision_df.plot(x='Architecture', y='Collision_Free_Steps', kind='bar',
                            ax=ax10, color=['#B4E7CE', '#95B8D1', '#EACBD2'])
            ax10.set_title('Collision-Free Steps', fontsize=12, fontweight='bold')
            ax10.set_xlabel('Architecture')
            ax10.set_ylabel('Number of Steps')
            ax10.tick_params(axis='x', rotation=45)
            ax10.legend().remove()
        
        # 11. Average Min Distance
        ax11 = plt.subplot(3, 4, 11)
        if not collision_df.empty:
            collision_df.plot(x='Architecture', y='Avg_Min_Distance', kind='bar',
                            ax=ax11, color=['#D4A5A5', '#A5C9D4', '#C9D4A5'])
            ax11.set_title('Average Minimum Distance', fontsize=12, fontweight='bold')
            ax11.set_xlabel('Architecture')
            ax11.set_ylabel('Distance')
            ax11.tick_params(axis='x', rotation=45)
            ax11.legend().remove()
        
        # 12. Performance Efficiency (Fitness/Time)
        ax12 = plt.subplot(3, 4, 12)
        if not scalability_df.empty:
            scalability_df['Efficiency'] = scalability_df['Mean_Fitness'] / (scalability_df['Eval_Time'] + 0.001)
            for arch in scalability_df['Architecture'].unique():
                arch_data = scalability_df[scalability_df['Architecture'] == arch]
                ax12.plot(arch_data['Boid_Count'], arch_data['Efficiency'], 
                        marker='h', label=arch, linewidth=2)
            ax12.set_title('Performance Efficiency (Fitness/Time)', fontsize=12, fontweight='bold')
            ax12.set_xlabel('Number of Boids')
            ax12.set_ylabel('Efficiency Score')
            ax12.legend()
            ax12.grid(True, alpha=0.3)
        
        plt.suptitle('Neural Network Architecture Comparison: 8 vs 12 vs 20 Hidden Nodes', 
                     fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def generate_summary_report(self) -> Dict:
        """Summary report."""
        neural_df = self.analyze_neural_dynamics()
        scalability_df = self.analyze_scalability()
        collision_df = self.analyze_collisions()
        
        report = {}
        
        for hidden_nodes in self.architectures:
            arch_name = f"{hidden_nodes} Hidden Nodes"
            report[arch_name] = {}
            
            # Neural dynamics summary
            if not neural_df.empty:
                arch_neural = neural_df[neural_df['Architecture'] == f"{hidden_nodes} Hidden"]
                if not arch_neural.empty:
                    report[arch_name]['Neural Dynamics'] = {
                        'Avg Spike Rate': arch_neural['Avg_Spike_Rate'].mean(),
                        'Avg Dead Neurons': arch_neural['Avg_Dead_Neurons'].mean(),
                        'Max Dead Neurons': arch_neural['Max_Dead_Neurons'].max()
                    }
            
            # Scalability summary
            if not scalability_df.empty:
                arch_scale = scalability_df[scalability_df['Architecture'] == f"{hidden_nodes} Hidden"]
                if not arch_scale.empty:
                    report[arch_name]['Scalability'] = {
                        'Best Fitness': arch_scale['Mean_Fitness'].max(),
                        'Worst Fitness': arch_scale['Mean_Fitness'].min(),
                        'Avg Collision Rate': arch_scale['Collision_Rate'].mean(),
                        'Max Eval Time': arch_scale['Eval_Time'].max(),
                        'Fitness at 100 Boids': arch_scale[arch_scale['Boid_Count'] == 100]['Mean_Fitness'].values[0] if not arch_scale[arch_scale['Boid_Count'] == 100].empty else None
                    }
            
            # Collision summary
            if not collision_df.empty:
                arch_coll = collision_df[collision_df['Architecture'] == f"{hidden_nodes} Hidden"]
                if not arch_coll.empty:
                    report[arch_name]['Collisions'] = {
                        'Total Collisions': arch_coll['Total_Collisions'].values[0] if len(arch_coll) > 0 else 0,
                        'Avg Per Step': arch_coll['Avg_Collisions_Per_Step'].values[0] if len(arch_coll) > 0 else 0,
                        'Collision Free Steps': arch_coll['Collision_Free_Steps'].values[0] if len(arch_coll) > 0 else 0
                    }
        
        return report
    
    def print_summary(self):
        """Print a formatted summary of the comparison."""
        report = self.generate_summary_report()
        
        print("\n" + "="*80)
        print(" NEURAL NETWORK ARCHITECTURE COMPARISON SUMMARY ".center(80))
        print("="*80)
        
        for arch_name, metrics in report.items():
            print(f"\n{arch_name}")
            print("-"*40)
            
            for category, values in metrics.items():
                print(f"\n  {category}:")
                for metric, value in values.items():
                    if value is not None:
                        if isinstance(value, float):
                            print(f"    • {metric}: {value:.4f}")
                        else:
                            print(f"    • {metric}: {value}")
                    else:
                        print(f"    • {metric}: N/A")
        
      
        
        best_arch = self.determine_best_architecture(report)
        print(f"\nBest Overall Architecture: {best_arch['name']}")
        print(f"Reasoning: {best_arch['reason']}")
    
    def determine_best_architecture(self, report: Dict) -> Dict:
        """Determine the best architecture based on multiple criteria."""
        scores = {}
        
        for arch_name in report.keys():
            score = 0
            metrics = report[arch_name]
            
            # Fitness score (higher is better)
            if 'Scalability' in metrics and metrics['Scalability'].get('Best Fitness'):
                score += metrics['Scalability']['Best Fitness'] * 100
            
            # Collision score (lower is better)
            if 'Collisions' in metrics and metrics['Collisions'].get('Total Collisions'):
                score -= metrics['Collisions']['Total Collisions'] * 0.5
            
            # Dead neurons (lower is better)
            if 'Neural Dynamics' in metrics and metrics['Neural Dynamics'].get('Avg Dead Neurons'):
                score -= metrics['Neural Dynamics']['Avg Dead Neurons'] * 10
            
            scores[arch_name] = score
        
        best_arch = max(scores, key=scores.get)
        
        reasons = []
        if report[best_arch].get('Scalability', {}).get('Best Fitness'):
            reasons.append(f"high fitness ({report[best_arch]['Scalability']['Best Fitness']:.3f})")
        if report[best_arch].get('Collisions', {}).get('Total Collisions'):
            reasons.append(f"collision management ({report[best_arch]['Collisions']['Total Collisions']} total)")
        
        return {
            'name': best_arch,
            'score': scores[best_arch],
            'reason': "Best balance of " + " and ".join(reasons)
        }


# Example usage
def main():
  
    data_paths = {
        12: {
            'neural': r'C:\Users\ss2658\Desktop\Alife\evaluation_results\neural_dynamics_20250803_214534.pkl',
            'scalability': r'C:\Users\ss2658\Desktop\Alife\evaluation_results\scalability_results_20250803_214534.pkl',
            'collision': r'C:\Users\ss2658\Desktop\Alife\evaluation_results\collision_analysis_50boids_20250803_214534.pkl'
        },
        20: {
            'neural': r'C:\Users\ss2658\Desktop\Alife\experiment_results2\neural_dynamics_20250804_014923.pkl',
            'scalability': r'C:\Users\ss2658\Desktop\Alife\experiment_results2\scalability_results_20250804_014923.pkl',
            'collision': r'C:\Users\ss2658\Desktop\Alife\experiment_results2\collision_analysis_50boids_20250804_014923.pkl'
        },
        8: {
            'neural': r'C:\Users\ss2658\Desktop\Alife\experiment_results3\neural_dynamics_20250804_014942.pkl',
            'scalability': r'C:\Users\ss2658\Desktop\Alife\experiment_results3\scalability_results_20250804_014942.pkl',
            'collision': r'C:\Users\ss2658\Desktop\Alife\experiment_results3\collision_analysis_50boids_20250804_014942.pkl'
        }
    }
    
    
    # Create comparison object
    comparison = NeuralArchitectureComparison(data_paths)
    
    # Generate and display comprehensive analysis
    comparison.print_summary()
    
    # Create visualization
    fig = comparison.plot_comparison()
    
    # Save the figure with high quality
    # Save as PNG (best for presentations/reports)
    fig.savefig('architecture_comparison.png', dpi=300, bbox_inches='tight')
    

    
    print("\nVisualization saved as 'architecture_comparison.png'")
    
    # Get detailed dataframes for further analysis
    neural_df = comparison.analyze_neural_dynamics()
    scalability_df = comparison.analyze_scalability()
    collision_df = comparison.analyze_collisions()
    
    # Save to CSV for further analysis
    neural_df.to_csv('neural_dynamics_comparison.csv', index=False)
    scalability_df.to_csv('scalability_comparison.csv', index=False)
    collision_df.to_csv('collision_comparison.csv', index=False)
    
    print("Analysis complete! Results saved to CSV files.")
    print("Visualization saved (multiple formats available)")
    
    return comparison, neural_df, scalability_df, collision_df


if __name__ == "__main__":
    comparison, neural_df, scalability_df, collision_df = main()