# import numpy as np
from typing import Callable, List, Tuple, Any
import matplotlib.pyplot as plt

class QuantumInspiredSearch:
    def __init__(self, 
                 objective_function: Callable[[np.ndarray], float],
                 dimensions: int,
                 population_size: int = 50,
                 max_iterations: int = 100,
                 bounds: List[Tuple[float, float]] = None):
        """
        Initialize a quantum-inspired probabilistic search algorithm.
        
        Args:
            objective_function: Function to minimize (takes numpy array)
            dimensions: Number of dimensions in the search space
            population_size: Number of candidate solutions
            max_iterations: Maximum number of iterations
            bounds: List of (min, max) bounds for each dimension
        """
        self.objective_function = objective_function
        self.dimensions = dimensions
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.bounds = bounds or [(-10, 10)] * dimensions
        
        # Initialize quantum bits (qubits) in superposition
        self.qubits = np.random.uniform(-1, 1, size=(population_size, dimensions, 2))
        self.qubits /= np.linalg.norm(self.qubits, axis=2, keepdims=True)  # Normalize
        
        # Initialize best solution
        self.best_solution = None
        self.best_fitness = float('inf')
        self.convergence_history = []
    
    def _measure_qubits(self) -> np.ndarray:
        """Measure the quantum states to get classical solutions."""
        solutions = np.zeros((self.population_size, self.dimensions))
        for i in range(self.population_size):
            for j in range(self.dimensions):
                # Probability amplitudes
                alpha, beta = self.qubits[i, j]
                prob = alpha**2  # Probability of being in state 0
                
                # Sample from probability distribution
                value = np.random.choice([0, 1], p=[prob, 1 - prob])
                
                # Map binary value to solution space
                min_val, max_val = self.bounds[j]
                solutions[i, j] = min_val + value * (max_val - min_val)
        return solutions
    
    def _update_qubits(self, solutions: np.ndarray, fitness_values: np.ndarray):
        """Update quantum states based on measurement outcomes and fitness."""
        # Find the best solution in current population
        current_best_idx = np.argmin(fitness_values)
        current_best_solution = solutions[current_best_idx]
        current_best_fitness = fitness_values[current_best_idx]
        
        # Update global best
        if current_best_fitness < self.best_fitness:
            self.best_fitness = current_best_fitness
            self.best_solution = current_best_solution.copy()
        
        self.convergence_history.append(self.best_fitness)
        
        # Quantum-inspired update rule
        learning_rate = 0.01 * (1 - len(self.convergence_history) / self.max_iterations)
        
        for i in range(self.population_size):
            for j in range(self.dimensions):
                # Calculate direction towards best solution
                direction = 1 if current_best_solution[j] > solutions[i, j] else -1
                
                # Update qubit amplitudes
                alpha, beta = self.qubits[i, j]
                
                # Rotate qubit state towards direction
                theta = learning_rate * direction
                rotation_matrix = np.array([
                    [np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]
                ])
                
                new_state = rotation_matrix @ np.array([alpha, beta])
                self.qubits[i, j] = new_state / np.linalg.norm(new_state)  # Renormalize
    
    def optimize(self) -> Tuple[np.ndarray, float]:
        """Run the quantum-inspired optimization algorithm."""
        for iteration in range(self.max_iterations):
            # Measure qubits to get classical solutions
            solutions = self._measure_qubits()
            
            # Evaluate fitness
            fitness_values = np.array([self.objective_function(sol) for sol in solutions])
            
            # Update qubits based on fitness
            self._update_qubits(solutions, fitness_values)
            
            # Optional: Print progress
            if iteration % 10 == 0:
                print(f"Iteration {iteration}, Best Fitness: {self.best_fitness:.4f}")
        
        return self.best_solution, self.best_fitness
    
    def plot_convergence(self):
        """Plot the convergence history."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.convergence_history, 'b-', linewidth=2)
        plt.xlabel('Iteration')
        plt.ylabel('Fitness (Objective Value)')
        plt.title('Convergence of Quantum-Inspired Search')
        plt.grid(True)
        plt.show()

# Example usage with Rastrigin function
def rastrigin_function(x: np.ndarray) -> float:
    """Rastrigin test function (multimodal, challenging for optimization)."""
    A = 10
    return A * len(x) + sum([(xi**2 - A * np.cos(2 * np.pi * xi)) for xi in x])

# Run optimization
if __name__ == "__main__":
    # Define optimization problem
    dimensions = 2
    bounds = [(-5.12, 5.12)] * dimensions
    
    # Initialize optimizer
    optimizer = QuantumInspiredSearch(
        objective_function=rastrigin_function,
        dimensions=dimensions,
        population_size=30,
        max_iterations=100,
        bounds=bounds
    )
    
    # Run optimization
    best_solution, best_fitness = optimizer.optimize()
    
    # Print results
    print("\nOptimization Results:")
    print(f"Best Solution: {np.round(best_solution, 4)}")
    print(f"Best Fitness: {best_fitness:.4f}")
    
    # Plot convergence
    optimizer.plot_convergence()
