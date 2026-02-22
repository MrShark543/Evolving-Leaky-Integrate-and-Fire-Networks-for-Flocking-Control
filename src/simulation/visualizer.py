import pygame
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import List, Optional
from ..boids.classical_boid import ClassicalBoid

class PygameVisualizer:
    """Real-time pygame visualization for boid simulation"""
    
    def __init__(self, width: int = 800, height: int = 600, fps: int = 60):
        pygame.init()
        self.width = width
        self.height = height
        self.fps = fps
        
        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.BLUE = (0, 100, 255)
        self.RED = (255, 100, 100)
        self.GREEN = (100, 255, 100)
        
        # Setup display
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("SNN-Boids Simulation")
        self.clock = pygame.time.Clock()
        
        # Font for text
        self.font = pygame.font.Font(None, 36)
        
    def draw_boid(self, boid: ClassicalBoid, color: tuple = None):
        """Draw a single boid as a triangle pointing in velocity direction"""
        if color is None:
            color = self.BLUE
            
        pos = boid.position.astype(int)
        vel = boid.velocity
        
        # Calculate triangle points
        if np.linalg.norm(vel) > 0:
            # Normalize velocity for direction
            direction = vel / np.linalg.norm(vel)
            
            # Triangle size
            size = 8
            
            # Calculate triangle points
            tip = pos + direction * size
            left = pos + np.array([-direction[1], direction[0]]) * size * 0.5 - direction * size * 0.3
            right = pos + np.array([direction[1], -direction[0]]) * size * 0.5 - direction * size * 0.3
            
            # Draw triangle
            points = [tip, left, right]
            pygame.draw.polygon(self.screen, color, points)
        else:
            # Draw circle if no velocity
            pygame.draw.circle(self.screen, color, pos, 5)
    
    def draw_perception_circle(self, boid: ClassicalBoid, color: tuple = (100, 100, 100)):
        """Draw perception radius around boid"""
        pos = boid.position.astype(int)
        pygame.draw.circle(self.screen, color, pos, int(boid.perception_radius), 1)
    
    def draw_text(self, text: str, x: int, y: int, color: tuple = None):
        """Draw text on screen"""
        if color is None:
            color = self.WHITE
        text_surface = self.font.render(text, True, color)
        self.screen.blit(text_surface, (x, y))
    
    def update_display(self, boids: List[ClassicalBoid], metrics: dict = None, 
                      show_perception: bool = False):
        """Update the display with current boid positions"""
        # Clear screen
        self.screen.fill(self.BLACK)
        
        
        if show_perception and len(boids) < 20:  # Only for small flocks
            for boid in boids:
                self.draw_perception_circle(boid)
        
        # Draw boids
        for boid in boids:
            # Color-code different types if needed
            if hasattr(boid, 'boid_type'):
                if boid.boid_type == 'snn':
                    color = self.RED
                else:
                    color = self.BLUE
            else:
                color = self.BLUE
            
            self.draw_boid(boid, color)
        
        # Draw metrics if provided
        if metrics:
            y_offset = 10
            self.draw_text(f"Boids: {metrics.get('num_boids', 0)}", 10, y_offset)
            y_offset += 30
            self.draw_text(f"Cohesion: {metrics.get('cohesion', 0):.1f}", 10, y_offset)
            y_offset += 30
            self.draw_text(f"Alignment: {metrics.get('alignment', 0):.2f}", 10, y_offset)
            y_offset += 30
            self.draw_text(f"Separation: {metrics.get('separation', 0):.1f}", 10, y_offset)
        
        # Update display
        pygame.display.flip()
        self.clock.tick(self.fps)
    
    def check_quit(self) -> bool:
        """Check to quit"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return True
        return False
    
    def cleanup(self):
        """Clean up pygame"""
        pygame.quit()

class MatplotlibVisualizer:
    """Static matplotlib visualization for analysis"""
    
    def __init__(self):
        self.fig = None
        self.axes = None
    
    def plot_trajectory(self, positions_history: List[np.ndarray], 
                       title: str = "Boid Trajectories"):
        """Plot trajectories of all boids"""
        plt.figure(figsize=(10, 8))
        
        # Plot trajectories for each boid
        num_boids = positions_history[0].shape[0]
        colors = plt.cm.tab10(np.linspace(0, 1, min(num_boids, 10)))
        
        for boid_id in range(min(num_boids, 10)):  
            trajectory = np.array([pos[boid_id] for pos in positions_history])
            plt.plot(trajectory[:, 0], trajectory[:, 1], 
                    color=colors[boid_id], alpha=0.7, linewidth=1)
            
            # Mark start and end
            plt.scatter(trajectory[0, 0], trajectory[0, 1], 
                       color=colors[boid_id], s=50, marker='o')
            plt.scatter(trajectory[-1, 0], trajectory[-1, 1], 
                       color=colors[boid_id], s=50, marker='s')
        
        plt.title(title)
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        return plt.gcf()
    
    def plot_metrics_comparison(self, classical_metrics: List[dict], 
                              snn_metrics: List[dict] = None):
        """Compare metrics between classical and SNN boids"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Extract classical metrics
        steps = [m['time_step'] for m in classical_metrics]
        classical_cohesion = [m['cohesion'] for m in classical_metrics]
        classical_alignment = [m['alignment'] for m in classical_metrics]
        classical_separation = [m['separation'] for m in classical_metrics]
        
        # Plot classical metrics
        axes[0, 0].plot(steps, classical_cohesion, label='Classical', color='blue')
        axes[0, 1].plot(steps, classical_alignment, label='Classical', color='blue')
        axes[1, 0].plot(steps, classical_separation, label='Classical', color='blue')
        
        # Plot SNN metrics if provided
        if snn_metrics:
            snn_steps = [m['time_step'] for m in snn_metrics]
            snn_cohesion = [m['cohesion'] for m in snn_metrics]
            snn_alignment = [m['alignment'] for m in snn_metrics]
            snn_separation = [m['separation'] for m in snn_metrics]
            
            axes[0, 0].plot(snn_steps, snn_cohesion, label='SNN', color='red')
            axes[0, 1].plot(snn_steps, snn_alignment, label='SNN', color='red')
            axes[1, 0].plot(snn_steps, snn_separation, label='SNN', color='red')
        
        # Set titles and labels
        axes[0, 0].set_title('Cohesion Over Time')
        axes[0, 0].set_ylabel('Average Distance to Center')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_title('Alignment Over Time')
        axes[0, 1].set_ylabel('Average Velocity Alignment')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].set_title('Minimum Separation Over Time')
        axes[1, 0].set_ylabel('Minimum Distance')
        axes[1, 0].set_xlabel('Time Step')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Power consumption plot
        axes[1, 1].set_title('Power Consumption (SNN Only)')
        axes[1, 1].set_xlabel('Time Step')
        axes[1, 1].set_ylabel('Power (mW)')
        if snn_metrics and 'power' in snn_metrics[0]:
            power_data = [m['power'] for m in snn_metrics]
            axes[1, 1].plot(snn_steps, power_data, color='red')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
