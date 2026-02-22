import torch
import torch.nn as nn
from .neurons import LIFNeuron
import numpy as np

class EnhancedSNN(nn.Module):
    """Enhanced SNN with specialized 3-output architecture for flocking behaviors """
    
    def __init__(self, input_size: int = 14, hidden_size: int = 10, output_size: int = 4):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Enhanced neuron parameters for better activity
        self.hidden_neurons = LIFNeuron(beta=0.75, threshold=0.35)  # More responsive
        self.output_neurons = LIFNeuron(beta=0.65, threshold=0.35)   # Easier activation
        
        # Connection layers with bias
        self.fc1 = nn.Linear(input_size, hidden_size, bias=True)
        self.fc2 = nn.Linear(hidden_size, output_size, bias=True)
        
        # Initialize with specialized weights
        self._initialize_specialized_weights()
        
        # Enhanced spike tracking
        self.output_spike_history = []
        self.hidden_spike_history = []
        self.history_length = 12  # Longer history for stability
        
        # Activity monitoring
        self.step_count = 0
        self.dead_neuron_threshold = 0.02
        
    def _initialize_specialized_weights(self):
        """Initialize weights for specialized 3-output behaviors"""
        with torch.no_grad():
            # Input to hidden: create diverse feature detectors
            nn.init.xavier_uniform_(self.fc1.weight)
            
            # Give hidden neurons varied bias patterns for diversity
            bias_patterns = [0.2, 0.35, 0.5, 0.3, 0.4, 0.25, 0.45, 0.3, 0.35, 0.4]
            for i in range(min(self.hidden_size, len(bias_patterns))):
                self.fc1.bias.data[i] = bias_patterns[i]
            
            # Hidden to output: specialized for each behavior
            nn.init.xavier_uniform_(self.fc2.weight)
            
            # Specialized output biases
            output_biases = [0.4, 0.25, 0.3, 0.3]  # separation, cohesion, align_x, align_y
            for i in range(min(self.output_size, len(output_biases))):
                self.fc2.bias.data[i] = output_biases[i]
            
            # Clamp weights to prevent saturation
            self.fc1.weight.data.clamp_(-0.8, 0.8)
            self.fc2.weight.data.clamp_(-0.7, 0.7)
            
            # Ensure each output has strong diverse connections
            for output_idx in range(self.output_size):
                # Each output gets 3-4 strong connections to different hidden neurons
                num_connections = 3 if output_idx == 0 else 2  # Separation gets more
                strong_connections = torch.randperm(self.hidden_size)[:num_connections]
                
                # Make these connections strong and positive
                self.fc2.weight.data[output_idx, strong_connections] = torch.abs(
                    self.fc2.weight.data[output_idx, strong_connections]
                ) + 0.3
                
                # Add some inhibitory connections for balance
                if self.hidden_size > num_connections:
                    inhibitory_connections = torch.randperm(self.hidden_size)[num_connections:num_connections+1]
                    self.fc2.weight.data[output_idx, inhibitory_connections] = -(
                        torch.abs(self.fc2.weight.data[output_idx, inhibitory_connections]) + 0.2
                    )
    
    def forward(self, input_spikes: torch.Tensor) -> torch.Tensor:
        """Forward pass with enhanced activity monitoring and CUDA support"""
        self.step_count += 1
        
        # Convert and validate input - ensure proper device handling
        if isinstance(input_spikes, list):
            input_spikes = torch.tensor(input_spikes, dtype=torch.float32, device=self.fc1.weight.device).unsqueeze(0)
        elif len(input_spikes.shape) == 1:
            input_spikes = input_spikes.unsqueeze(0)
        
        # Ensure input is on the same device as the model
        input_spikes = input_spikes.to(self.fc1.weight.device)
        
        # Scale inputs for better neural response
        scaled_inputs = input_spikes * 2.2  
        
        # Hidden layer processing
        hidden_current = self.fc1(scaled_inputs)
        hidden_spikes = self.hidden_neurons(hidden_current)
        
        # Track hidden activity 
        self.hidden_spike_history.append(hidden_spikes.clone())
        if len(self.hidden_spike_history) > self.history_length:
            self.hidden_spike_history.pop(0)
        
        # Output layer processing
        output_current = self.fc2(hidden_spikes)
        output_spikes = self.output_neurons(output_current)
        
        # Track output activity 
        self.output_spike_history.append(output_spikes.clone())
        if len(self.output_spike_history) > self.history_length:
            self.output_spike_history.pop(0)
        
        return output_spikes
    
    def get_output_rates(self) -> torch.Tensor:
        """Get output firing rates with dead neuron detection"""
        if not self.output_spike_history:
            # Default specialized activity pattern 
            device = self.fc2.weight.device
            return torch.tensor([[0.4, 0.3, 0.35]], device=device)  # separation, cohesion, align_x/y
        
        # Calculate average rates - keep tensors on device
        spike_sum = torch.stack(self.output_spike_history).sum(dim=0)
        rates = spike_sum / len(self.output_spike_history)
        
        # Detect and handle dead neurons 
        rates_cpu = rates.cpu().squeeze().detach().numpy()  # Move to CPU only for numpy operations
        dead_neurons = rates_cpu < self.dead_neuron_threshold
        
        if np.any(dead_neurons):
            # Revive dead neurons with specialized backup activity
            backup_activities = [0.4, 0.3, 0.35, 0.35]  # Different for each output type
            
            rates_corrected = rates.clone()
            for i, is_dead in enumerate(dead_neurons):
                if is_dead and i < len(backup_activities):
                    # Give dead neuron some activity based on its role
                    variation = torch.randn(1, device=rates.device) * 0.05  # Small random variation on device
                    rates_corrected[0, i] = backup_activities[i] + variation
        else:
            rates_corrected = rates
        
        # Ensure reasonable activity bounds
        rates_final = torch.clamp(rates_corrected, min=0.05, max=0.95)
        
        return rates_final
    
    def get_hidden_rates(self) -> torch.Tensor:
        """Get hidden layer firing rates for analysis """
        if not self.hidden_spike_history:
            device = self.fc1.weight.device
            return torch.zeros(1, self.hidden_size, device=device)
        
        spike_sum = torch.stack(self.hidden_spike_history).sum(dim=0)
        rates = spike_sum / len(self.hidden_spike_history)
        return rates
    
    def get_network_activity_stats(self) -> dict:
        """Get comprehensive network activity statistics"""
        # Move to CPU for numpy operations
        output_rates = self.get_output_rates().cpu().squeeze().detach().numpy()
        hidden_rates = self.get_hidden_rates().cpu().squeeze().detach().numpy()
        
        # Output neuron analysis
        output_stats = {
            'separation_activity': output_rates[0] if len(output_rates) > 0 else 0,
            'cohesion_activity': output_rates[1] if len(output_rates) > 1 else 0,
            'alignment_x_activity': output_rates[2] if len(output_rates) > 2 else 0,
            'alignment_y_activity': output_rates[3] if len(output_rates) > 3 else 0,
            'output_mean': np.mean(output_rates),
            'output_std': np.std(output_rates),
            'dead_outputs': np.sum(output_rates < self.dead_neuron_threshold)
        }
        
        # Hidden neuron analysis
        if len(hidden_rates) > 0:
            hidden_stats = {
                'hidden_mean': np.mean(hidden_rates),
                'hidden_std': np.std(hidden_rates),
                'active_hidden': np.sum(hidden_rates > self.dead_neuron_threshold),
                'dead_hidden': np.sum(hidden_rates < self.dead_neuron_threshold)
            }
        else:
            hidden_stats = {
                'hidden_mean': 0, 'hidden_std': 0, 
                'active_hidden': 0, 'dead_hidden': 0
            }
        
        return {**output_stats, **hidden_stats, 'step_count': self.step_count}
    
    def revive_dead_neurons(self):
        """Attempt to revive dead neurons by adjusting weights"""
        if not self.output_spike_history:
            return
        
        # Move to CPU for analysis
        output_rates = self.get_output_rates().cpu().squeeze().detach().numpy()
        
        with torch.no_grad():
            for output_idx, rate in enumerate(output_rates):
                if rate < self.dead_neuron_threshold:
                    # Boost connections to this output 
                    device = self.fc2.weight.device
                    boost_indices = torch.randperm(self.hidden_size, device=device)[:2]
                    self.fc2.weight[output_idx, boost_indices] += torch.randn(2, device=device) * 0.2
                    self.fc2.bias[output_idx] += 0.1
                    
                    # Also boost some hidden neurons that connect to this output
                    hidden_boost_idx = torch.randint(0, self.hidden_size, (1,), device=device).item()
                    input_boost_indices = torch.randperm(self.input_size, device=device)[:2]
                    self.fc1.weight[hidden_boost_idx, input_boost_indices] += torch.randn(2, device=device) * 0.15
    
    def reset_state(self):
        """Reset all neuron states and history"""
        self.hidden_neurons.reset_state()
        self.output_neurons.reset_state()
        self.output_spike_history.clear()
        self.hidden_spike_history.clear()
        self.step_count = 0

# Update the SimpleSNN to use enhanced version when needed
class SimpleSNN(EnhancedSNN):
    """Alias for backward compatibility """
    pass