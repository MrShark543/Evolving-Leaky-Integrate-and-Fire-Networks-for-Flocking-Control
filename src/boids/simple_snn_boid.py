import numpy as np
import torch
from typing import List
from .classical_boid import ClassicalBoid
from ..neural.network import SimpleSNN
import torch.nn as nn

class SimpleSNNBoid(ClassicalBoid):
    """Enhanced SNN-controlled boid with separation focus"""
    
    def __init__(self, x: float, y: float, vx: float = 0, vy: float = 0, 
                 boid_id: int = 0, use_trained_weights: bool = False):
        super().__init__(x, y, vx, vy, boid_id)

        # Not using cuda
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # SNN: Creation
        self.network = SimpleSNN(input_size=8, hidden_size=12, output_size=3)
        
        
        self.network = self.network.to(self.device)
        
        # Override boid type
        self.boid_type = 'enhanced_snn'
        
      
        self.max_speed = 2.0             
        self.max_force = 0.05            
        self.perception_radius = 90.0     
  
        self.separation_distance = 30.0   
        self.separation_radius = 30.0     
        self.alignment_radius = 100.0      
        self.cohesion_radius = 80.0       
        
        # Initialize with separation-focused weights
        if not use_trained_weights:
            self._initialize_separation_focused_weights()
        
        # Performance tracking
        self.step_count = 0
        self.total_force_applied = 0.0
        
    def _initialize_separation_focused_weights(self):
        """Initialize with weights """
        with torch.no_grad():
            # Small random weights
            nn.init.uniform_(self.network.fc1.weight, -0.3, 0.3)
            nn.init.uniform_(self.network.fc1.bias, 0.3, 0.5)
            nn.init.uniform_(self.network.fc2.weight, -0.2, 0.2)
            
            # SEPARATION-FOCUSED output biases
            self.network.fc2.bias.data[0] = 0.6   # Separation (highest)
            self.network.fc2.bias.data[1] = 0.35  # Alignment  
            self.network.fc2.bias.data[2] = 0.35  # Cohesion
            
            # Strengthen separation connections
            separation_connections = torch.randperm(self.network.fc2.weight.shape[1])[:4]
            self.network.fc2.weight.data[0, separation_connections] += 0.3
    
    def update(self, neighbors: List['ClassicalBoid'], dt: float = 1.0):
        """Update boid position and velocity based on SNN and flocking rules """
        self.step_count += 1
        
        # Get local neighbors
        local_neighbors = self._get_local_neighbors(neighbors)
        
        # Get enhanced inputs with separation focus
        sensor_inputs = self._get_enhanced_inputs(local_neighbors)
        
        # Process through SNN 
        input_tensor = torch.tensor(sensor_inputs, dtype=torch.float32).to(self.device)
        output_spikes = self.network(input_tensor)
        
        output_rates = self.network.get_output_rates()
        if isinstance(output_rates, torch.Tensor):
            output_rates = output_rates.cpu().squeeze().detach().numpy()
        else:
            output_rates = np.array(output_rates)
        
        # Enhanced force calculation
        if len(local_neighbors) > 0:
            # Calculate classical forces
            classical_separation = self._separation_force(local_neighbors)
            classical_alignment = self._alignment_force(local_neighbors)
            classical_cohesion = self._cohesion_force(local_neighbors)
            
            # Get SNN weights 
            if len(output_rates) >= 3:

                sep_weight = np.clip(output_rates[0] * 2.0, 0.5, 2.5)  # Boost separation
                align_weight = np.clip(output_rates[1] * 1.5, 0.3, 1.5)  # Boost alignment
                cohesion_weight = np.clip(output_rates[2] * 1.5, 0.3, 1.5)  # Boost cohesion
                
                # Dynamic separation boosting based on danger
                separation_urgency = sensor_inputs[0]
                collision_warning = sensor_inputs[1]
                
                if collision_warning > 0.5:
                    sep_weight *= 3.0    # Emergency separation
                    align_weight *= 0.2
                    cohesion_weight *= 0.2
                elif separation_urgency > 0.4:
                    sep_weight *= 2.0    # High separation priority
                    align_weight *= 0.5
                    cohesion_weight *= 0.5
                    
            else:
                # Fallback: separation-focused weights
                sep_weight = 1.5
                align_weight = 0.5
                cohesion_weight = 0.5
            
            # Apply forces with enhanced separation scaling
            total_force = (
                sep_weight * classical_separation +
                align_weight * classical_alignment +
                cohesion_weight * classical_cohesion
            )
        else:
            total_force = np.array([0.0, 0.0])
        
        # Apply force and update
        self._apply_force(total_force, dt)
        self.total_force_applied += np.linalg.norm(total_force)
    
    def _get_enhanced_inputs(self, local_neighbors: List['ClassicalBoid']) -> List[float]:
        """Get 8 enhanced inputs"""
        if len(local_neighbors) == 0:
            return [0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5]
        
        # Calculate distances
        distances = [np.linalg.norm(self.position - n.position) for n in local_neighbors]
        min_distance = min(distances)
        
        # INPUT 0: Separation urgency (most critical)
        separation_urgency = max(0.0, (self.separation_distance - min_distance) / self.separation_distance)
        
        # INPUT 1: Collision warning (binary)
        collision_warning = 1.0 if min_distance < 20.0 else 0.0
        
        # INPUT 2: Dangerous neighbors count
        dangerous_count = sum(1 for d in distances if d < self.separation_distance)
        dangerous_neighbors = min(dangerous_count / 6.0, 1.0)
        
        # INPUT 3: Separation direction strength
        separation_vector = np.zeros(2)
        for i, neighbor in enumerate(local_neighbors):
            if distances[i] < self.separation_distance:
                diff = self.position - neighbor.position
                if distances[i] > 0:
                    weight = (self.separation_distance - distances[i]) / self.separation_distance
                    separation_vector += (diff / distances[i]) * weight
        
        separation_direction_strength = min(np.linalg.norm(separation_vector) / 2.0, 1.0)
        
        # INPUT 4: Current speed
        current_speed = min(np.linalg.norm(self.velocity) / self.max_speed, 1.0)
        
        # INPUT 5: Alignment signal
        neighbor_velocities = [n.velocity for n in local_neighbors]
        avg_neighbor_vel = np.mean(neighbor_velocities, axis=0)
        if np.linalg.norm(avg_neighbor_vel) > 0 and np.linalg.norm(self.velocity) > 0:
            my_vel_norm = self.velocity / np.linalg.norm(self.velocity)
            neighbor_vel_norm = avg_neighbor_vel / np.linalg.norm(avg_neighbor_vel)
            alignment = (np.dot(my_vel_norm, neighbor_vel_norm) + 1) / 2
        else:
            alignment = 0.5
        
        # INPUT 6: Cohesion signal
        center_of_mass = np.mean([n.position for n in local_neighbors], axis=0)
        center_distance = np.linalg.norm(center_of_mass - self.position)
        cohesion_signal = min(center_distance / 100.0, 1.0)
        
        # INPUT 7: Neighbor density
        neighbor_density = min(len(local_neighbors) / 8.0, 1.0)
        
        return [
            separation_urgency,
            collision_warning, 
            dangerous_neighbors,
            separation_direction_strength,
            current_speed,
            alignment,
            cohesion_signal,
            neighbor_density
        ]
    
    def _separation_force(self, neighbors: List['ClassicalBoid']) -> np.ndarray:
        """Classical separation force with correct distance"""
        steer = np.array([0.0, 0.0])
        count = 0
        
        for neighbor in neighbors:
            if neighbor == self:
                continue
                
            distance = np.linalg.norm(self.position - neighbor.position)
            if 0 < distance < self.separation_distance:  
                diff = self.position - neighbor.position
                diff = diff / distance  # Normalize
                diff = diff / distance  # Weight by distance
                steer += diff
                count += 1
        
        if count > 0:
            steer = steer / count
            if np.linalg.norm(steer) > 0:
                steer = steer / np.linalg.norm(steer) * self.max_speed
                steer = steer - self.velocity
                if np.linalg.norm(steer) > self.max_force:
                    steer = steer / np.linalg.norm(steer) * self.max_force
        
        return steer
    
    def _alignment_force(self, neighbors: List['ClassicalBoid']) -> np.ndarray:
        """Classical alignment force - steer towards average heading of neighbors"""
        if len(neighbors) == 0:
            return np.array([0.0, 0.0])
        
        # Calculate average velocity
        avg_velocity = np.zeros(2, dtype=np.float32)
        for neighbor in neighbors:
            avg_velocity += neighbor.velocity
        avg_velocity = avg_velocity / len(neighbors)
        
        # Steering = Desired - Current
        steer = avg_velocity - self.velocity
        
        # Limit steering force
        if np.linalg.norm(steer) > self.max_force:
            steer = steer / np.linalg.norm(steer) * self.max_force
        
        return steer
    
    def _cohesion_force(self, neighbors: List['ClassicalBoid']) -> np.ndarray:
        """Classical cohesion force - steer towards center of mass of neighbors"""
        if len(neighbors) == 0:
            return np.array([0.0, 0.0])
        
        # Calculate center of mass
        center = np.zeros(2, dtype=np.float32)
        for neighbor in neighbors:
            center += neighbor.position
        center = center / len(neighbors)
        
        # Calculate desired velocity towards center
        desired = center - self.position
        if np.linalg.norm(desired) > 0:
            desired = desired / np.linalg.norm(desired) * self.max_speed
        
        # Steering = Desired - Current
        steer = desired - self.velocity
        
        # Limit steering force
        if np.linalg.norm(steer) > self.max_force:
            steer = steer / np.linalg.norm(steer) * self.max_force
        
        return steer
    
   


    def _apply_force(self, force: np.ndarray, dt: float = 1.0):
        """Apply force and update boid position and velocity with smooth movement"""
        # Store previous velocity for momentum
        prev_velocity = self.velocity.copy()
        
        # Limit force magnitude to max_force
        if np.linalg.norm(force) > self.max_force:
            force = force / np.linalg.norm(force) * self.max_force
        
        # Update velocity with applied force
        self.velocity += force * dt
        
        # Apply momentum for smoother movement
        momentum_factor = 0.1
        self.velocity = (1 - momentum_factor) * self.velocity + momentum_factor * prev_velocity
        
        current_speed = np.linalg.norm(self.velocity)
        min_speed = 0.5  # Minimum speed to prevent stopping
        
        if current_speed < min_speed:
            # If speed is too low, set to minimum speed in random direction
            if current_speed > 0.01:
                self.velocity = (self.velocity / current_speed) * min_speed
            else:
                # Random direction if stopped
                random_angle = np.random.uniform(0, 2 * np.pi)
                self.velocity = np.array([np.cos(random_angle), np.sin(random_angle)]) * min_speed
        
        # Limit to max speed as before
        if np.linalg.norm(self.velocity) > self.max_speed:
            self.velocity = self.velocity / np.linalg.norm(self.velocity) * self.max_speed
        
        # Update position
        self.position += self.velocity * dt
        
        # Track total force applied
        self.total_force_applied += np.linalg.norm(force)


# Factory function
def create_simple_snn_boid(x: float, y: float, vx: float = 0, vy: float = 0, 
                          boid_id: int = 0, weights_file: str = None):
    """Create a simple SNN boid"""
    boid = SimpleSNNBoid(x, y, vx, vy, boid_id, use_trained_weights=True)
    print(f"Created simple SNN boid {boid_id}")
    return boid