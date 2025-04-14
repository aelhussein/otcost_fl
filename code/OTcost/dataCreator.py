import pandas as pd
import numpy as np
import random
from sklearn.utils import shuffle

global ROOT_DIR
ROOT_DIR = '/gpfs/commons/groups/gursoy_lab/aelhussein/classes/otcost_fl'
DATA_DIR = f'{ROOT_DIR}/data/Synthetic'

class SyntheticDataGenerator:
    """
    Generates synthetic tabular datasets with controlled distribution properties
    and specifically designed to have orthogonal representations in feature space.
    """
    def __init__(self, n_features=10, random_state=42):
        self.n_features = n_features
        self.random_state = random_state
        np.random.seed(random_state)
        
    def generate_orthogonal_matrices(self, dim):
        """Generate orthogonal matrices using QR decomposition"""
        # Generate a random matrix
        H = np.random.randn(dim, dim)
        # QR decomposition
        Q, R = np.linalg.qr(H)
        return Q
        
    def generate_base_distributions(self, n_samples=1000):
        """
        Generate two base distributions with orthogonal principal components
        """
        # Create orthogonal matrices (basis)
        orthogonal_matrix = self.generate_orthogonal_matrices(self.n_features)
        
        # Split the orthogonal matrix to create two separate basis matrices
        basis_1 = orthogonal_matrix[:, :self.n_features//2]
        basis_2 = orthogonal_matrix[:, self.n_features//2:]
        
        # Generate data from each basis
        data_1 = np.random.normal(0, 1, (n_samples, self.n_features//2))
        data_2 = np.random.normal(0, 1, (n_samples, self.n_features//2))
        
        # Project the data onto the full feature space
        dataset_1 = data_1 @ basis_1.T
        dataset_2 = data_2 @ basis_2.T
        
        # Add some random noise to make it more realistic
        dataset_1 += np.random.normal(0, 0.1, dataset_1.shape)
        dataset_2 += np.random.normal(0, 0.1, dataset_2.shape)
        
        return dataset_1, dataset_2
    
    def apply_nonlinear_transformations(self, dataset, distribution_type='mixed'):
        """
        Apply non-linear transformations to create more complex distributions
        
        Parameters:
        -----------
        dataset: np.ndarray
            The base dataset to transform
        distribution_type: str
            The type of distribution to create, options are:
            - 'normal': primarily normal distributions
            - 'skewed': primarily skewed distributions
            - 'mixed': mixture of different distributions
            - 'multi_modal': multi-modal distributions
        """
        transformed = np.copy(dataset)
        n_samples, n_features = dataset.shape
        
        if distribution_type == 'normal':
            # Create mostly normal distributions with different means and variances
            for i in range(n_features):
                mean = np.random.uniform(-3, 3)
                std = np.random.uniform(0.5, 2)
                transformed[:, i] = dataset[:, i] * std + mean
                
        elif distribution_type == 'skewed':
            # Create mostly skewed distributions
            for i in range(n_features):
                if i % 3 == 0:
                    # Exponential-like
                    transformed[:, i] = np.exp(dataset[:, i] / 2)
                elif i % 3 == 1:
                    # Power-law-like
                    transformed[:, i] = np.sign(dataset[:, i]) * np.abs(dataset[:, i])**np.random.uniform(1.5, 3)
                else:
                    # Log-normal-like
                    transformed[:, i] = np.exp(dataset[:, i] / 3)
                    
        elif distribution_type == 'mixed':
            # Create a mix of different distributions
            for i in range(n_features):
                if i % 5 == 0:
                    # Normal
                    transformed[:, i] = dataset[:, i] * np.random.uniform(0.5, 2)
                elif i % 5 == 1:
                    # Exponential
                    transformed[:, i] = np.exp(dataset[:, i] / 2) - 1
                elif i % 5 == 2:
                    # Sine wave modulation
                    transformed[:, i] = np.sin(dataset[:, i] * 2) + dataset[:, i] / 2
                elif i % 5 == 3:
                    # Polynomial
                    transformed[:, i] = dataset[:, i]**3 / 5 + dataset[:, i]
                else:
                    # Mixture
                    transformed[:, i] = np.abs(dataset[:, i]) + np.random.uniform(-1, 1, n_samples) * 0.5
        
        elif distribution_type == 'multi_modal':
            # Create multi-modal distributions
            for i in range(n_features):
                modes = np.random.randint(2, 5)
                centers = np.random.uniform(-5, 5, modes)
                widths = np.random.uniform(0.5, 1.5, modes)
                
                # Initialize with zeros
                feature_values = np.zeros(n_samples)
                
                # Assign each sample to a random mode
                mode_assignments = np.random.randint(0, modes, n_samples)
                
                for j in range(n_samples):
                    mode = mode_assignments[j]
                    feature_values[j] = dataset[j, i] * widths[mode] + centers[mode]
                
                transformed[:, i] = feature_values
        
        return transformed
        
    
    def generate_datasets(self, n_samples=1000, dist_type1='normal', dist_type2='skewed', add_labels=True, label_noise=0.1):
        """
        Generate two complete datasets with labels
        
        Parameters:
        -----------
        n_samples: int
            Number of samples per dataset
        dist_type1, dist_type2: str
            Distribution types for datasets 1 and 2
        add_labels: bool
            Whether to add classification labels
        label_noise: float
            Proportion of noisy labels
            
        Returns:
        --------
        dict containing two datasets with their features and labels
        """
        # Generate orthogonal base distributions
        base1, base2 = self.generate_base_distributions(n_samples)
        
        # Apply non-linear transformations
        dataset1 = self.apply_nonlinear_transformations(base1, dist_type1)
        dataset2 = self.apply_nonlinear_transformations(base2, dist_type2)
        
        # Generate labels if requested
        if add_labels:
            sums1 = np.sum(dataset1, axis=1)
            sums2 = np.sum(dataset2, axis=1)

            median1 = np.median(sums1)
            median2 = np.median(sums2)
            # Simple classification rule based on sum of features
            labels1 = (sums1 > median1).astype(int)
            labels2 = (sums2 > median2).astype(int)
            
            # Add label noise
            noise_mask1 = np.random.random(n_samples) < label_noise
            noise_mask2 = np.random.random(n_samples) < label_noise
            
            labels1[noise_mask1] = 1 - labels1[noise_mask1]
            labels2[noise_mask2] = 1 - labels2[noise_mask2]
        else:
            labels1 = None
            labels2 = None
        
        return {
            'dataset1': {
                'X': dataset1,
                'y': labels1,
                'distribution': dist_type1
            },
            'dataset2': {
                'X': dataset2,
                'y': labels2,
                'distribution': dist_type2
            },
        }
    
def non_iid_creator(frac, total_cases=800, n_features=10, dist_type1='normal', dist_type2='skewed', label_noise=0.05, random_state=42):    
    # Create generator
    generator = SyntheticDataGenerator(n_features=n_features, random_state=random_state)
    
    # Generate first dataset
    result1 = generator.generate_datasets(
        n_samples=total_cases*2,
        dist_type1=dist_type1,
        dist_type2=dist_type2,  # Same type for consistency
        add_labels=True,
        label_noise=label_noise
    )

    
    # Split first distribution data into two parts
    first_dist_data = result1['dataset1']['X']
    first_dist_labels = result1['dataset1']['y']
    
    # First half for dataset 1
    X1 = first_dist_data[:total_cases]
    y1 = first_dist_labels[:total_cases]
    
    # Second half for mixing
    X1b = first_dist_data[total_cases:]
    y1b = first_dist_labels[total_cases:]
    
    # Get second distribution data
    X2 = result1['dataset2']['X'][:total_cases]
    y2 = result1['dataset2']['y'][:total_cases]

    # Shuffle first dataset
    X1, y1 = shuffle(X1, y1, random_state=random_state)
    
    # Create mixed second dataset
    a = int(np.floor(total_cases * frac))
    X2_mixed = np.concatenate((X2[:a], X1b[a:]))
    y2_mixed = np.concatenate((y2[:a], y1b[a:]))
    
    # Shuffle mixed dataset
    X2_mixed, y2_mixed = shuffle(X2_mixed, y2_mixed, random_state=random_state)
    
    # Create return dictionaries
    data, label = {}, {}
    data['1'], label['1'] = X1, y1
    data['2'], label['2'] = X2_mixed, y2_mixed
    
    return data, label


def saveDataset(X,y, name):
    d1= np.concatenate((X, y.reshape(-1,1)), axis=1)
    np.savetxt(f'{DATA_DIR}/{name}.csv',d1)
    return
