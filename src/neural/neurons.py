import torch
import torch.nn as nn
import numpy as np

class LIFNeuron(nn.Module):
    """Leaky Integrate-and-Fire Neuron with proper state management and CUDA support"""
    
    def __init__(self, beta: float = 0.9, threshold: float = 1.0, 
                 reset_mechanism: str = 'zero', refractory_period: int = 1):
        super().__init__()
        self.beta = beta  # Membrane decay constant
        self.threshold = threshold  # Spiking threshold
        self.reset_mechanism = reset_mechanism
        self.refractory_period = refractory_period
        
        # State variables 
        self.membrane_potential = None
        self.refractory_counter = None
        
    def forward(self, input_current: torch.Tensor) -> torch.Tensor:
        """Forward pass of LIF neuron with CUDA support"""
        batch_size, num_neurons = input_current.shape
        device = input_current.device  # Use the device of input tensor
        
        # Initialize state 
        if (self.membrane_potential is None or 
            self.membrane_potential.device != device or
            self.membrane_potential.shape != (batch_size, num_neurons)):
            
            self.membrane_potential = torch.zeros(batch_size, num_neurons, 
                                                device=device, dtype=input_current.dtype)
            self.refractory_counter = torch.zeros(batch_size, num_neurons, 
                                                device=device, dtype=input_current.dtype)
        
        # Update membrane potential 
        non_refractory = (self.refractory_counter <= 0)
        
        # Decay membrane potential
        self.membrane_potential = self.beta * self.membrane_potential
        
        # Add input current
        self.membrane_potential += input_current * non_refractory.float()
        
        # Generate spikes
        spikes = (self.membrane_potential >= self.threshold).float() * non_refractory.float()
        
        # Reset membrane potential after spiking
        if self.reset_mechanism == 'zero':
            self.membrane_potential = self.membrane_potential * (1 - spikes)
        elif self.reset_mechanism == 'subtract':
            self.membrane_potential = self.membrane_potential - spikes * self.threshold
        
        # Set refractory period for spiking neurons
        self.refractory_counter = torch.where(
            spikes > 0, 
            torch.full_like(spikes, self.refractory_period), 
            torch.maximum(self.refractory_counter - 1, torch.zeros_like(self.refractory_counter))
        )
        
        return spikes
    
    def reset_state(self):
        """Reset neuron state"""
        self.membrane_potential = None
        self.refractory_counter = None
    
    def get_membrane_potential(self) -> torch.Tensor:
        """Get current membrane potential for visualization"""
        return self.membrane_potential if self.membrane_potential is not None else torch.tensor([])