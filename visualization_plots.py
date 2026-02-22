"""
Comprehensive Visualization Script for LIF SNN Training Results

"""

import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Tuple
import pandas as pd
from datetime import datetime

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class ComprehensiveVisualizer:
    """Generate all plots from evaluation data"""
    
    def __init__(self, data_dir: str = "evaluation_results", 
                 save_dir: str = "report_figures"):
        self.data_dir = data_dir
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Set consistent plot parameters
        self.figsize = (10, 6)
        self.dpi = 300
        self.colors = sns.color_palette("husl", 8)
        
    def load_all_data(self, timestamp: str = "20250803_214534"):
        """Load all data files for a specific evaluation run"""
        data = {}
        
        # Load scalability results
        scalability_pkl = os.path.join(self.data_dir, f"scalability_results_{timestamp}.pkl")
        scalability_json = os.path.join(self.data_dir, f"scalability_results_{timestamp}.json")
        
        if os.path.exists(scalability_pkl):
            with open(scalability_pkl, 'rb') as f:
                data['scalability'] = pickle.load(f)
        elif os.path.exists(scalability_json):
            with open(scalability_json, 'r') as f:
                data['scalability'] = json.load(f)
        
        # Load collision analysis
        collision_pkl = os.path.join(self.data_dir, f"collision_analysis_50boids_{timestamp}.pkl")
        if os.path.exists(collision_pkl):
            with open(collision_pkl, 'rb') as f:
                data['collisions'] = pickle.load(f)
        
        # Load neural dynamics if available
        neural_pkl = os.path.join(self.data_dir, f"neural_dynamics_{timestamp}.pkl")
        if os.path.exists(neural_pkl):
            with open(neural_pkl, 'rb') as f:
                data['neural'] = pickle.load(f)
        
        # Load efficiency analysis if available
        efficiency_pkl = os.path.join(self.data_dir, f"efficiency_analysis_{timestamp}.pkl")
        if os.path.exists(efficiency_pkl):
            with open(efficiency_pkl, 'rb') as f:
                data['efficiency'] = pickle.load(f)
        
        return data
    
    def plot_scalability_analysis(self, data: Dict = None):
        """Scalability plotting function"""
        
        # Load the JSON data directly since pickle seems corrupted
        try:
            with open('evaluation_results/scalability_results_20250803_214534.json', 'r') as f: #file name
                scalability_data = json.load(f)
        except FileNotFoundError:
            print("Scalability JSON file not found")
            return
        
        # Create figure
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Extract data properly
        boid_counts = scalability_data['boid_counts']
        results = scalability_data['results']
        
        # Prepare arrays for plotting
        mean_fitness = []
        std_fitness = []
        mean_collisions = []
        mean_collision_rates = []
        mean_cohesion = []
        mean_alignment = []
        mean_separation = []
        mean_eval_times = []
        
        # Extract data for each boid count
        for count in boid_counts:
            if str(count) in results:
                res = results[str(count)]
                mean_fitness.append(res['mean_fitness'])
                std_fitness.append(res['std_fitness'])
                mean_collisions.append(res['mean_collisions'])
                mean_collision_rates.append(res['mean_collision_rate'])
                mean_cohesion.append(res['mean_cohesion'])
                mean_alignment.append(res['mean_alignment'])
                mean_separation.append(res['mean_separation'])
                mean_eval_times.append(res['mean_eval_time'])
        
        print(f"Loaded data for {len(mean_fitness)} boid counts")
        print(f"Fitness values: {mean_fitness}")
        
        if not mean_fitness:
            print("No valid fitness data found")
            return
        
        # 1. Fitness vs Boid Count
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.errorbar(boid_counts, mean_fitness, yerr=std_fitness, 
                    marker='o', capsize=5, capthick=2, linewidth=2, 
                    markersize=8, color=self.colors[0])
        ax1.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Acceptable threshold')
        ax1.set_xlabel('Number of Boids', fontsize=11)
        ax1.set_ylabel('Fitness Score', fontsize=11)
        ax1.set_title('A. Fitness Scaling with Flock Size', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_ylim([0, 0.6])
        
        # 2. Collision Rate vs Boid Count
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(boid_counts, mean_collision_rates, marker='s', 
                linewidth=2, markersize=8, color=self.colors[1])
        ax2.fill_between(boid_counts, 0, mean_collision_rates, alpha=0.3, color=self.colors[1])
        ax2.set_xlabel('Number of Boids', fontsize=11)
        ax2.set_ylabel('Collision Rate (per step)', fontsize=11)
        ax2.set_title('B. Collision Rate Scaling', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. Flocking Metrics vs Boid Count
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(boid_counts, mean_cohesion, marker='o', label='Cohesion', linewidth=2, color=self.colors[2])
        ax3.plot(boid_counts, np.array(mean_alignment)*100, marker='s', label='Alignment (×100)', linewidth=2, color=self.colors[3])
        ax3.plot(boid_counts, mean_separation, marker='^', label='Separation', linewidth=2, color=self.colors[4])
        ax3.set_xlabel('Number of Boids', fontsize=11)
        ax3.set_ylabel('Metric Value', fontsize=11)
        ax3.set_title('C. Flocking Metrics Scaling', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Computation Time Scaling
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.plot(boid_counts, mean_eval_times, marker='d', 
                linewidth=2, markersize=8, color=self.colors[5])
        ax4.set_xlabel('Number of Boids', fontsize=11)
        ax4.set_ylabel('Evaluation Time (seconds)', fontsize=11)
        ax4.set_title('D. Computational Complexity', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # 5. Performance Components Heatmap
        ax5 = fig.add_subplot(gs[1, 1:])
        
        # Create heatmap data
        metrics_matrix = []
        for count in boid_counts:
            if str(count) in results:
                row_data = []
                trials = results[str(count)]['trials']
                for trial in trials[:3]:
                    row_data.extend([
                        trial.get('cohesion_score', 0),
                        trial.get('alignment_score', 0),
                        trial.get('separation_score', 0),
                        trial.get('spike_fitness', 0),
                        trial.get('flocking_fitness', 0)
                    ])
                while len(row_data) < 15:
                    row_data.append(0)
                metrics_matrix.append(row_data[:15])
        
        if metrics_matrix:
            metrics_matrix = np.array(metrics_matrix)
            metric_labels = ['Coh', 'Align', 'Sep', 'Spike', 'Flock'] * 3
            trial_labels = ['T1-']*5 + ['T2-']*5 + ['T3-']*5
            col_labels = [t + m for t, m in zip(trial_labels, metric_labels)]
            
            im = ax5.imshow(metrics_matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
            ax5.set_xticks(range(len(col_labels)))
            ax5.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=8)
            ax5.set_yticks(range(len(boid_counts)))
            ax5.set_yticklabels(boid_counts)
            ax5.set_ylabel('Number of Boids', fontsize=11)
            ax5.set_title('E. Performance Components Across Trials', fontsize=12, fontweight='bold')
            plt.colorbar(im, ax=ax5, label='Score')
        
        # 6. Success Rate Bar Chart
        ax6 = fig.add_subplot(gs[2, :2])
        
        success_rates = []
        for count in boid_counts:
            if str(count) in results:
                trials = results[str(count)]['trials']
                successes = sum(1 for t in trials if t['fitness'] > 0.5)
                success_rates.append(successes / len(trials) * 100)
        
        if success_rates:
            bars = ax6.bar(range(len(boid_counts)), success_rates, color=self.colors[6])
            ax6.set_xticks(range(len(boid_counts)))
            ax6.set_xticklabels(boid_counts)
            ax6.set_xlabel('Number of Boids', fontsize=11)
            ax6.set_ylabel('Success Rate (%)', fontsize=11)
            ax6.set_title('F. Trial Success Rate (Fitness > 0.5)', fontsize=12, fontweight='bold')
            ax6.axhline(y=50, color='r', linestyle='--', alpha=0.5)
            ax6.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for i, (bar, rate) in enumerate(zip(bars, success_rates)):
                height = bar.get_height()
                ax6.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{rate:.0f}%', ha='center', va='bottom')
        
        # 7. Summary Box
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.axis('off')
        
        # Calculate key statistics
        if mean_fitness:
            optimal_idx = np.argmax(mean_fitness)
            optimal_size = boid_counts[optimal_idx]
            max_fitness = mean_fitness[optimal_idx]
            
            # Find specific performances
            fitness_15 = mean_fitness[0] if len(mean_fitness) > 0 else 0
            fitness_25 = mean_fitness[1] if len(mean_fitness) > 1 else 0
            fitness_50 = mean_fitness[3] if len(mean_fitness) > 3 else 0
            fitness_100 = mean_fitness[5] if len(mean_fitness) > 5 else 0
            
            summary_text = f"""SCALABILITY SUMMARY
        
    Optimal Flock Size: {optimal_size} boids
    Peak Fitness: {max_fitness:.3f}

    Performance at Scale:
    • 15 boids: {fitness_15:.3f}
    • 25 boids: {fitness_25:.3f}
    • 50 boids: {fitness_50:.3f}
    • 100 boids: {fitness_100:.3f}

   """
            
            ax7.text(0.1, 0.9, summary_text, transform=ax7.transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('Scalability Analysis - LIF SNN Flocking Performance', 
                    fontsize=14, fontweight='bold', y=1.02)
        
        # Save and show
        filename = os.path.join(self.save_dir, 'scalability_analysis.png')
        plt.savefig(filename, dpi=self.dpi, bbox_inches='tight')
        plt.show()
        print(f"Saved: {filename}")
    
    def plot_collision_analysis(self, data: Dict):
        """Create collision analysis plots"""
        if 'collisions' not in data:
            print("No collision data found")
            return
        
        collisions = data['collisions']
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Collision Analysis - 50 Boids System', fontsize=14, fontweight='bold')
        
        # 1. Collision count over time
        ax1 = axes[0, 0]
        ax1.plot(collisions['time_steps'], collisions['collision_counts'], 
                linewidth=2, color=self.colors[0])
        ax1.fill_between(collisions['time_steps'], 0, collisions['collision_counts'], 
                         alpha=0.3, color=self.colors[0])
        ax1.set_xlabel('Time Step', fontsize=11)
        ax1.set_ylabel('Number of Collisions', fontsize=11)
        ax1.set_title('A. Collision Events Over Time', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 2. Near-miss events
        ax2 = axes[0, 1]
        ax2.plot(collisions['time_steps'], collisions['near_miss_counts'],
                linewidth=2, color=self.colors[1], label='Near Misses')
        ax2.plot(collisions['time_steps'], collisions['collision_counts'],
                linewidth=2, color=self.colors[0], label='Collisions')
        ax2.set_xlabel('Time Step', fontsize=11)
        ax2.set_ylabel('Count', fontsize=11)
        ax2.set_title('B. Collisions vs Near Misses', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Minimum distance evolution
        ax3 = axes[0, 2]
        ax3.plot(collisions['time_steps'], collisions['min_distances'],
                linewidth=2, color=self.colors[2])
        ax3.axhline(y=20, color='r', linestyle='--', alpha=0.5, label='Collision Threshold')
        ax3.axhline(y=30, color='orange', linestyle='--', alpha=0.5, label='Near-Miss Threshold')
        ax3.set_xlabel('Time Step', fontsize=11)
        ax3.set_ylabel('Minimum Distance', fontsize=11)
        ax3.set_title('C. Minimum Inter-Boid Distance', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Collision frequency distribution
        ax4 = axes[1, 0]
        collision_counts = collisions['collision_counts']
        ax4.hist(collision_counts, bins=20, edgecolor='black', color=self.colors[3])
        ax4.set_xlabel('Collisions per Sample', fontsize=11)
        ax4.set_ylabel('Frequency', fontsize=11)
        ax4.set_title('D. Collision Frequency Distribution', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 5. Cumulative collisions
        ax5 = axes[1, 1]
        cumulative_collisions = np.cumsum(collision_counts)
        ax5.plot(collisions['time_steps'], cumulative_collisions,
                linewidth=2, color=self.colors[4])
        ax5.fill_between(collisions['time_steps'], 0, cumulative_collisions,
                         alpha=0.3, color=self.colors[4])
        ax5.set_xlabel('Time Step', fontsize=11)
        ax5.set_ylabel('Cumulative Collisions', fontsize=11)
        ax5.set_title('E. Cumulative Collision Count', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # 6. Summary statistics
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        summary = collisions['summary']
        summary_text = f"""COLLISION STATISTICS
        
Total Collisions: {summary['total_collisions']}
Avg per Step: {summary['avg_collisions_per_step']:.2f}
Max per Step: {summary['max_collisions_per_step']}

Collision-Free Steps: {summary['collision_free_steps']}/100
Success Rate: {summary['collision_free_steps']:.0f}%

Avg Min Distance: {summary['avg_min_distance']:.1f}
Total Near Misses: {summary['total_near_misses']}

Safety Assessment:
{'Good' if summary['collision_free_steps'] > 80 else 'Needs Improvement'}"""
        
        ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.tight_layout()
        
        # Save figure
        filename = os.path.join(self.save_dir, 'collision_analysis.png')
        plt.savefig(filename, dpi=self.dpi, bbox_inches='tight')
        plt.show()
        print(f"Saved: {filename}")
    
    def plot_training_history(self, checkpoint_dir: str = "optimized_lif_models/checkpoints"):
        """Plot training history from checkpoints"""
        
        # Find checkpoints
        if not os.path.exists(checkpoint_dir):
            print(f"Checkpoint directory not found: {checkpoint_dir}")
            # Try alternative path
            checkpoint_dir = "checkpoints"
            if not os.path.exists(checkpoint_dir):
                print("No checkpoint directory found, skipping training history")
                return
        
        try:
            # Find pickle checkpoint files directly
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pkl')]
            
            if not checkpoints:
                print("No checkpoint files found")
                return
            
            # Load the latest checkpoint
            latest = sorted(checkpoints)[-1]
            checkpoint_path = os.path.join(checkpoint_dir, latest)
            
            print(f"Loading checkpoint: {latest}")
            
            with open(checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)
            
            # Create training plots
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('Training History Analysis', fontsize=14, fontweight='bold')
            
            # 1. Fitness progression
            ax1 = axes[0, 0]
            
            if 'fitness_history' in checkpoint and checkpoint['fitness_history']:
                fitness_history = checkpoint['fitness_history']
                generations = range(1, len(fitness_history) + 1)
                
                best_fitness = [h['best_fitness'] for h in fitness_history]
                avg_fitness = [h['avg_fitness'] for h in fitness_history]
                
                ax1.plot(generations, best_fitness, 
                        label='Best', linewidth=2, color=self.colors[0])
                ax1.plot(generations, avg_fitness, 
                        label='Average', linewidth=2, color=self.colors[1])
                ax1.fill_between(generations, avg_fitness, best_fitness, alpha=0.3)
                ax1.set_xlabel('Generation', fontsize=11)
                ax1.set_ylabel('Fitness', fontsize=11)
                ax1.set_title('A. Fitness Evolution', fontsize=12, fontweight='bold')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
            else:
                ax1.text(0.5, 0.5, 'No fitness history', ha='center', va='center', transform=ax1.transAxes)
                ax1.set_title('A. Fitness Evolution', fontsize=12, fontweight='bold')
            
            # 2. Population diversity
            ax2 = axes[0, 1]
            if 'generation_fitness_history' in checkpoint:
                gen_fitness = checkpoint['generation_fitness_history']
                diversity = [np.std(gen) for gen in gen_fitness if gen]
                
                if diversity:
                    ax2.plot(range(1, len(diversity)+1), diversity, 
                            linewidth=2, color=self.colors[2])
                    ax2.set_xlabel('Generation', fontsize=11)
                    ax2.set_ylabel('Population Std Dev', fontsize=11)
                    ax2.set_title('B. Population Diversity', fontsize=12, fontweight='bold')
                    ax2.grid(True, alpha=0.3)
                else:
                    ax2.text(0.5, 0.5, 'No diversity data', ha='center', va='center', transform=ax2.transAxes)
                    ax2.set_title('B. Population Diversity', fontsize=12, fontweight='bold')
            else:
                ax2.text(0.5, 0.5, 'No diversity data', ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('B. Population Diversity', fontsize=12, fontweight='bold')
            
            # 3. Target evolution
            ax3 = axes[1, 0]
            if 'current_targets' in checkpoint:
                targets = checkpoint['current_targets']
                info_text = "Final Targets:\n\n"
                info_text += f"Cohesion: {targets.get('cohesion', 'N/A'):.1f}\n"
                info_text += f"Alignment: {targets.get('alignment', 'N/A'):.2f}\n"
                info_text += f"Separation: {targets.get('separation', 'N/A'):.1f}\n\n"
                info_text += f"Boid Count: {checkpoint.get('current_boid_count', 'N/A')}"
                
                ax3.text(0.1, 0.8, info_text, fontsize=11, transform=ax3.transAxes,
                        verticalalignment='top')
                ax3.axis('off')
                ax3.set_title('C. Training Parameters', fontsize=12, fontweight='bold')
            else:
                ax3.axis('off')
                ax3.text(0.5, 0.5, 'No target data', ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('C. Training Parameters', fontsize=12, fontweight='bold')
            
            # 4. Training summary
            ax4 = axes[1, 1]
            ax4.axis('off')
            
            summary_text = "TRAINING SUMMARY\n\n"
            summary_text += f"Generation: {checkpoint.get('generation', 'N/A')}\n"
            summary_text += f"Best Fitness: {checkpoint.get('best_fitness', 0):.4f}\n"
            
            if 'training_params' in checkpoint:
                params = checkpoint['training_params']
                summary_text += f"Population Size: {params.get('population_size', 'N/A')}\n"
                summary_text += f"Mutation Rate: {params.get('mutation_rate', 'N/A')}\n"
                summary_text += f"Workers: {params.get('num_workers', 'N/A')}\n"
            
            summary_text += f"\nTimestamp: {checkpoint.get('timestamp', 'N/A')[:19] if 'timestamp' in checkpoint else 'N/A'}"
            
            ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
                    fontsize=11, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
            
            plt.tight_layout()
            
            # Save figure
            filename = os.path.join(self.save_dir, 'training_history.png')
            plt.savefig(filename, dpi=self.dpi, bbox_inches='tight')
            plt.show()
            print(f"Saved: {filename}")
            
        except Exception as e:
            print(f"Error plotting training history: {e}")
            import traceback
            traceback.print_exc()
    
    def create_summary_report(self, data: Dict):
        """Create a summary report """
        
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle('LIF SNN Flocking System - Performance Summary Report', 
                    fontsize=16, fontweight='bold')
        
        # Create grid
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # Key metrics summary
        ax_summary = fig.add_subplot(gs[0, :2])
        ax_summary.axis('off')
        
        # Load scalability data directly from JSON for accuracy
        try:
            with open('evaluation_results/scalability_results_20250803_214534.json', 'r') as f:
                scalability_json = json.load(f)
            
            results = scalability_json['results']
            
            # Extract actual fitness values
            fitness_15 = results['15']['mean_fitness'] if '15' in results else 0
            fitness_25 = results['25']['mean_fitness'] if '25' in results else 0
            fitness_35 = results['35']['mean_fitness'] if '35' in results else 0
            fitness_50 = results['50']['mean_fitness'] if '50' in results else 0
            fitness_75 = results['75']['mean_fitness'] if '75' in results else 0
            fitness_100 = results['100']['mean_fitness'] if '100' in results else 0
            
            # Determine scalability assessment
            scalability_small = "Good" if fitness_25 > 0.3 else "Poor"
            scalability_large = "Poor"  
            
            summary_text = f"""
    Performance at Different Scales:
    • 15 Boids:  {fitness_15:.3f} fitness
    • 25 Boids:  {fitness_25:.3f} fitness
    • 35 Boids:  {fitness_35:.3f} fitness
    • 50 Boids:  {fitness_50:.3f} fitness
    • 75 Boids:  {fitness_75:.3f} fitness
    • 100 Boids: {fitness_100:.3f} fitness

    
   """
            
        except Exception as e:
            print(f"Error loading scalability data: {e}")
            summary_text = """"""
        
        ax_summary.text(0.05, 0.95, summary_text, transform=ax_summary.transAxes,
                    fontsize=11, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
        
        # Mini scalability plot
        ax1 = fig.add_subplot(gs[0, 2])
        try:
            boid_counts = scalability_json['boid_counts']
            mean_fitness = []
            for count in boid_counts:
                if str(count) in results:
                    mean_fitness.append(results[str(count)]['mean_fitness'])
            
            if mean_fitness:
                ax1.plot(boid_counts, mean_fitness, 'o-', linewidth=2, markersize=6, color='red')
                ax1.axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='Target')
                ax1.set_xlabel('Number of Boids', fontsize=10)
                ax1.set_ylabel('Fitness', fontsize=10)
                ax1.set_title('Scalability', fontweight='bold', fontsize=11)
                ax1.grid(True, alpha=0.3)
                ax1.set_ylim([0, 0.6])
                ax1.legend(fontsize=8)
        except:
            ax1.text(0.5, 0.5, 'No scalability data', ha='center', va='center')
            ax1.set_title('Scalability', fontweight='bold')
        
        # Mini collision plot
        ax2 = fig.add_subplot(gs[0, 3])
        if 'collisions' in data:
            try:
                collision_counts = data['collisions']['collision_counts'][:50]
                time_steps = range(0, len(collision_counts)*10, 10)
                ax2.plot(time_steps, collision_counts, linewidth=1.5, color='orange')
                ax2.fill_between(time_steps, 0, collision_counts, alpha=0.3, color='orange')
                ax2.set_xlabel('Time Step', fontsize=10)
                ax2.set_ylabel('Collisions', fontsize=10)
                ax2.set_title('Collision Pattern', fontweight='bold', fontsize=11)
                ax2.grid(True, alpha=0.3)
            except:
                ax2.text(0.5, 0.5, 'No collision data', ha='center', va='center')
                ax2.set_title('Collision Pattern', fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'No collision data', ha='center', va='center')
            ax2.set_title('Collision Pattern', fontweight='bold')
        
        # Performance breakdown bar chart
        ax3 = fig.add_subplot(gs[1, :2])
        try:
            # Create performance breakdown for different scales
            scales = ['15', '25', '35', '50', '75', '100']
            cohesion_scores = []
            alignment_scores = []
            separation_scores = []
            
            for scale in scales:
                if scale in results:
                    # Average across trials
                    trials = results[scale]['trials']
                    cohesion_scores.append(np.mean([t.get('cohesion_score', 0) for t in trials]))
                    alignment_scores.append(np.mean([t.get('alignment_score', 0) for t in trials]))
                    separation_scores.append(np.mean([t.get('separation_score', 0) for t in trials]))
            
            if cohesion_scores:
                x = np.arange(len(scales))
                width = 0.25
                
                bars1 = ax3.bar(x - width, cohesion_scores, width, label='Cohesion', color='blue', alpha=0.7)
                bars2 = ax3.bar(x, alignment_scores, width, label='Alignment', color='green', alpha=0.7)
                bars3 = ax3.bar(x + width, separation_scores, width, label='Separation', color='red', alpha=0.7)
                
                ax3.set_xlabel('Number of Boids', fontsize=11)
                ax3.set_ylabel('Score', fontsize=11)
                ax3.set_title('Performance Components by Scale', fontweight='bold', fontsize=12)
                ax3.set_xticks(x)
                ax3.set_xticklabels(scales)
                ax3.legend()
                ax3.grid(True, alpha=0.3, axis='y')
        except:
            ax3.text(0.5, 0.5, 'Performance breakdown unavailable', ha='center', va='center')
            ax3.set_title('Performance Components', fontweight='bold')
        
        # Collision statistics
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.axis('off')
        
        if 'collisions' in data:
            summary = data['collisions']['summary']
            collision_text = f"""COLLISION STATISTICS
            
    Total Collisions: {summary['total_collisions']}
    Avg per Step: {summary['avg_collisions_per_step']:.2f}
    Success Rate: {summary['collision_free_steps']}%

    Min Distance: {summary['avg_min_distance']:.1f}
    Near Misses: {summary['total_near_misses']}"""
        else:
            collision_text = "No collision data available"
        
        ax4.text(0.1, 0.8, collision_text, transform=ax4.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        # Training summary
        ax5 = fig.add_subplot(gs[1, 3])
        ax5.axis('off')
        
        training_text = """TRAINING SUMMARY
        
"""
        
        ax5.text(0.1, 0.8, training_text, transform=ax5.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
        
        # Key findings
        ax6 = fig.add_subplot(gs[2, :])
        ax6.axis('off')
        
        findings_text = """KEY FINDINGS AND LIMITATIONS
        
 """
        
        ax6.text(0.05, 0.9, findings_text, transform=ax6.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        # Remove tight_layout which causes warnings
        # plt.tight_layout()  # Comment out or remove this line
        
        # Add manual spacing adjustment instead
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.3, wspace=0.3)
        
        
        # Save figure
        filename = os.path.join(self.save_dir, 'summary_report.png')
        plt.savefig(filename, dpi=self.dpi, bbox_inches='tight')
        plt.show()
        print(f"Saved: {filename}")
    
    def generate_all_plots(self, timestamp: str = "20250803_214534"):
        """Generate all plots for the report"""
        print("="*60)
        print("GENERATING ALL PLOTS FOR REPORT")
        print("="*60)
        
        # Load all data
        print("\nLoading data...")
        data = self.load_all_data(timestamp)
        print(f"Loaded: {list(data.keys())}")
        
        # Generate each plot type
        print("\nGenerating plots...")
        
        print("\n1. Scalability Analysis...")
        self.plot_scalability_analysis(data)
        
        print("\n2. Collision Analysis...")
        self.plot_collision_analysis(data)
        
        print("\n3. Training History...")
        self.plot_training_history()
        
        print("\n4. Summary Report...")
        self.create_summary_report(data)
        
        print("\n" + "="*60)
        print(f"ALL PLOTS SAVED TO: {self.save_dir}/")
        print("="*60)


def main():
    """Main function to run all visualizations"""
    
    # Create visualizer
    visualizer = ComprehensiveVisualizer(
        data_dir="evaluation_results",
        save_dir="report_figures"
    )
    
    
    print("LIF SNN Comprehensive Data Visualizer")
    print("="*60)
    
    # Check for available data files
    eval_dir = "evaluation_results"
    if os.path.exists(eval_dir):
        files = os.listdir(eval_dir)
        timestamps = set()
        for f in files:
            # Look for timestamp patterns in filenames
            if '.pkl' in f or '.json' in f:
                # Try to extract timestamp (format: YYYYMMDD_HHMMSS)
                import re
                pattern = r'(\d{8}_\d{6})'
                matches = re.findall(pattern, f)
                for match in matches:
                    timestamps.add(match)
        
        if timestamps:
            print("\nAvailable evaluation timestamps:")
            sorted_timestamps = sorted(timestamps)
            for i, ts in enumerate(sorted_timestamps, 1):
                print(f"{i}. {ts}")
            
            print(f"\nDefault (latest): {sorted_timestamps[-1]}")
            choice = input("\nEnter number, timestamp, or press Enter for default: ").strip()
            
            if not choice:  # User pressed Enter
                timestamp = sorted_timestamps[-1]
            elif choice.isdigit() and 1 <= int(choice) <= len(sorted_timestamps):
                timestamp = sorted_timestamps[int(choice)-1]
            elif choice in timestamps:
                timestamp = choice
            else:
                print(f"Invalid choice, using default: {sorted_timestamps[-1]}")
                timestamp = sorted_timestamps[-1]
        else:
            print("\nNo timestamp found in files, using your data timestamp...")
            timestamp = "20250803_214534"  # 
        print(f"\nDirectory '{eval_dir}' not found, using default timestamp...")
        timestamp = "20250803_214534"
    
    print(f"\nUsing timestamp: {timestamp}")
    
    # Generate all plots
    try:
        visualizer.generate_all_plots(timestamp)
        print("\nYou can now use these figures in your report!")
        print("   Check the 'report_figures' directory for all plots.")
    except Exception as e:
        print(f"\nError generating plots: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()