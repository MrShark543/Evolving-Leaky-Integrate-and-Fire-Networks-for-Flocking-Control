"""
Boid implementations: Classical and SNN-controlled
"""

from .classical_boid import ClassicalBoid

# Try to import SNN boid (may not exist yet)
try:
    from .snn_boid import SNNBoid
    __all__ = ['ClassicalBoid', 'SNNBoid']
except ImportError:
    __all__ = ['ClassicalBoid']