import numpy as np
from typing import List, Tuple, Type
from ..boids.classical_boid import ClassicalBoid


try:
    from ..boids.snn_boid import SNNBoid
    SNN_AVAILABLE = True
except ImportError:
    SNN_AVAILABLE = False


class FlockingEnvironment:
    """Environment for boid flocking simulation"""
    
    def __init__(self, width: int = 800, height: int = 600, 
                 wrap_boundaries: bool = True):
        self.width = width
        self.height = height
        self.wrap_boundaries = wrap_boundaries
        self.boids: List[ClassicalBoid] = []
        self.time_step = 0
        
    def add_boid(self, boid: ClassicalBoid):
        """Add a boid to the environment"""
        self.boids.append(boid)
    
    def add_random_boids(self, num_boids: int, boid_class: Type = None):
        """Add random boids of a specific type to the environment"""
        if boid_class is None:
            boid_class = ClassicalBoid
            
        for i in range(num_boids):
            x = np.random.uniform(50, self.width - 50)
            y = np.random.uniform(50, self.height - 50)
            vx = np.random.uniform(-2, 2)
            vy = np.random.uniform(-2, 2)
            
            boid = boid_class(x, y, vx, vy, boid_id=len(self.boids))
            self.add_boid(boid)
        
        print(f"Added {num_boids} {boid_class.__name__} boids to environment")
    
    def update(self, dt: float = 1.0):
        """Update all boids in the environment"""
        # Update all boids
        for boid in self.boids:
            boid.update(self.boids, dt)
        
        # Handle boundary conditions
        if self.wrap_boundaries:
            self._wrap_boundaries()
        else:
            self._bounce_boundaries()
        
        self.time_step += 1
    
    def _wrap_boundaries(self):
        """Wrap boids around boundaries"""
        for boid in self.boids:
            if boid.position[0] < 0:
                boid.position[0] = self.width
            elif boid.position[0] > self.width:
                boid.position[0] = 0
                
            if boid.position[1] < 0:
                boid.position[1] = self.height
            elif boid.position[1] > self.height:
                boid.position[1] = 0
    


    def _bounce_boundaries(self):
        """Bounce boids off boundaries with speed preservation"""
        for boid in self.boids:
            # Add repulsion force near boundaries instead of hard bounce
            boundary_margin = 30
            repulsion_strength = 0.5
            
            if boid.position[0] <= boundary_margin:
                boid.velocity[0] += repulsion_strength
            elif boid.position[0] >= self.width - boundary_margin:
                boid.velocity[0] -= repulsion_strength
                
            if boid.position[1] <= boundary_margin:
                boid.velocity[1] += repulsion_strength
            elif boid.position[1] >= self.height - boundary_margin:
                boid.velocity[1] -= repulsion_strength
            
            # Still clamp to boundaries
            boid.position[0] = np.clip(boid.position[0], 0, self.width)
            boid.position[1] = np.clip(boid.position[1], 0, self.height)
    
    def get_flock_metrics(self) -> dict:
        """Enhanced metrics calculation with more detailed analysis"""
        if len(self.boids) < 2:
            return {
                'cohesion': 0.0,
                'alignment': 0.0, 
                'separation': 0.0,
                'num_boids': len(self.boids),
                'time_step': self.time_step,
                'boid_type': 'none'
            }
        
        positions = np.array([boid.position for boid in self.boids])
        velocities = np.array([boid.velocity for boid in self.boids])
        
        # Cohesion: average distance to center of mass
        center_of_mass = np.mean(positions, axis=0)
        distances_to_com = np.linalg.norm(positions - center_of_mass, axis=1)
        cohesion = np.mean(distances_to_com)
        
        # Alignment: average velocity alignment
        speeds = np.linalg.norm(velocities, axis=1)
        # Avoid division by zero
        speeds = np.where(speeds == 0, 1, speeds)
        normalized_velocities = velocities / speeds[:, np.newaxis]
        
        alignment = 0
        count = 0
        for i in range(len(self.boids)):
            for j in range(i + 1, len(self.boids)):
                alignment += np.dot(normalized_velocities[i], normalized_velocities[j])
                count += 1
        
        if count > 0:
            alignment = alignment / count
        
        # Separation: minimum distance between any two boids
        min_distance = float('inf')
        for i in range(len(self.boids)):
            for j in range(i + 1, len(self.boids)):
                dist = np.linalg.norm(positions[i] - positions[j])
                min_distance = min(min_distance, dist)
        
        # Determine boid type
        boid_type = 'unknown'
        if self.boids:
            if hasattr(self.boids[0], 'boid_type'):
                boid_type = self.boids[0].boid_type
            else:
                boid_type = 'classical'  # Default assumption
        
        return {
            'cohesion': cohesion,
            'alignment': alignment,
            'separation': min_distance if min_distance != float('inf') else 0.0,
            'num_boids': len(self.boids),
            'time_step': self.time_step,
            'boid_type': boid_type,
            'avg_speed': np.mean(speeds),
            'speed_variance': np.var(speeds)
        }
    
    def reset_environment(self):
        """Reset environment for new experiment"""
        self.boids.clear()
        self.time_step = 0
        print("Environment reset for new experiment")

    def get_experiment_summary(self) -> dict:
        """Get summary statistics for the current experiment"""
        if not self.boids:
            return {}
            
        positions = np.array([boid.position for boid in self.boids])
        velocities = np.array([boid.velocity for boid in self.boids])
        
        # Calculate bounding box of flock
        min_pos = np.min(positions, axis=0)
        max_pos = np.max(positions, axis=0)
        flock_size = np.linalg.norm(max_pos - min_pos)
        
        # Calculate average inter-boid distance
        total_distance = 0
        count = 0
        for i in range(len(self.boids)):
            for j in range(i + 1, len(self.boids)):
                total_distance += np.linalg.norm(positions[i] - positions[j])
                count += 1
        
        avg_distance = total_distance / count if count > 0 else 0
        
        return {
            'flock_size': flock_size,
            'avg_inter_boid_distance': avg_distance,
            'total_simulation_steps': self.time_step,
            'boid_count': len(self.boids),
            'boid_type': self.boids[0].boid_type if self.boids and hasattr(self.boids[0], 'boid_type') else 'unknown'
        }

        
    