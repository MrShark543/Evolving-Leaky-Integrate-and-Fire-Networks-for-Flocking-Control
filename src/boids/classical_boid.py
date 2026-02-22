import numpy as np
from typing import List, Tuple, Optional

class ClassicalBoid:
    """Traditional boid implementation using Craig Reynolds' rules with smooth movement"""
    
    def __init__(self, x: float, y: float, vx: float = 0, vy: float = 0, 
                 boid_id: int = 0):
        # Position and velocity
        self.position = np.array([x, y], dtype=np.float32)
        self.velocity = np.array([vx, vy], dtype=np.float32)
        self.id = boid_id
        
        # UPDATED PARAMETERS for smooth movement 
        self.max_speed = 2.0              #
        self.max_force = 0.05             # 
        self.perception_radius = 80.0     # 
        
        # UPDATED RULE WEIGHTS 
        self.separation_weight = 1.0      # 
        self.alignment_weight = 0.8       #  
        self.cohesion_weight = 0.6        # 
        
        # SMOOTHING PARAMETERS
        self.momentum_factor = 0.1        # Add momentum for smoother movement
        
        # Separation distance
        self.separation_distance = 30.0   
        # Boid type for identification
        self.boid_type = 'classical'
        
    def update(self, neighbors: List['ClassicalBoid'], dt: float = 1.0):
        """Boid position and velocity based on flocking rules with smoothing"""
        # Store previous velocity for momentum
        prev_velocity = self.velocity.copy()
        
        # Get local neighbors within perception radius
        local_neighbors = self._get_local_neighbors(neighbors)
        
        if len(local_neighbors) > 0:
            # Apply Reynolds' three rules
            sep_force = self._separation(local_neighbors) * self.separation_weight
            ali_force = self._alignment(local_neighbors) * self.alignment_weight
            coh_force = self._cohesion(local_neighbors) * self.cohesion_weight
            
            # Combine forces
            total_force = sep_force + ali_force + coh_force
            
            # Limit steering force 
            total_force = self._limit_force(total_force, self.max_force)
            
            # Update velocity
            self.velocity += total_force * dt
        
        # Apply momentum for smoother movement
        self.velocity = (1 - self.momentum_factor) * self.velocity + self.momentum_factor * prev_velocity
        
        # Limit speed
        self.velocity = self._limit_speed(self.velocity, self.max_speed)
        
        # Update position
        self.position += self.velocity * dt
        
    def _get_local_neighbors(self, neighbors: List['ClassicalBoid']) -> List['ClassicalBoid']:
        """Get neighbors within perception radius"""
        local = []
        for neighbor in neighbors:
            if neighbor.id != self.id:
                distance = np.linalg.norm(self.position - neighbor.position)
                if distance < self.perception_radius:
                    local.append(neighbor)
        return local
    
    def _separation(self, neighbors: List['ClassicalBoid']) -> np.ndarray:
        """Separation: steer to avoid crowding local flockmates"""
        steer = np.zeros(2, dtype=np.float32)
        count = 0
        
        for neighbor in neighbors:
            distance = np.linalg.norm(self.position - neighbor.position)
            if distance < self.separation_distance and distance > 0:
                # Calculate vector pointing away from neighbor
                diff = self.position - neighbor.position
                diff = diff / distance  # Normalize
                diff = diff / distance  # Weight by distance (closer = stronger)
                steer += diff
                count += 1
        
        if count > 0:
            steer = steer / count
            # Steering = Desired - Current
            if np.linalg.norm(steer) > 0:
                steer = self._normalize(steer) * self.max_speed
                steer = steer - self.velocity
        
        return steer
    
    def _alignment(self, neighbors: List['ClassicalBoid']) -> np.ndarray:
        """Alignment: steer towards average heading of neighbors"""
        if len(neighbors) == 0:
            return np.zeros(2, dtype=np.float32)
        
        # Calculate average velocity
        avg_velocity = np.zeros(2, dtype=np.float32)
        for neighbor in neighbors:
            avg_velocity += neighbor.velocity
        avg_velocity = avg_velocity / len(neighbors)
        
        # Steering = Desired - Current
        steer = avg_velocity - self.velocity
        return steer
    
    def _cohesion(self, neighbors: List['ClassicalBoid']) -> np.ndarray:
        """Cohesion: steer to move toward average position of neighbors"""
        if len(neighbors) == 0:
            return np.zeros(2, dtype=np.float32)
        
        # Calculate center of mass
        center = np.zeros(2, dtype=np.float32)
        for neighbor in neighbors:
            center += neighbor.position
        center = center / len(neighbors)
        
        # Seek center
        desired = center - self.position
        if np.linalg.norm(desired) > 0:
            desired = self._normalize(desired) * self.max_speed
        
        # Steering = Desired - Current
        steer = desired - self.velocity
        return steer
    
    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        """Normalize vector to unit length"""
        norm = np.linalg.norm(vector)
        if norm > 0:
            return vector / norm
        return vector
    
    def _limit_force(self, force: np.ndarray, max_force: float) -> np.ndarray:
        """Limit force magnitude"""
        force_mag = np.linalg.norm(force)
        if force_mag > max_force:
            return force * (max_force / force_mag)
        return force
    
    def _limit_speed(self, velocity: np.ndarray, max_speed: float) -> np.ndarray:
        """Limit velocity magnitude"""
        speed = np.linalg.norm(velocity)
        if speed > max_speed:
            return velocity * (max_speed / speed)
        return velocity