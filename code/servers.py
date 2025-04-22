"""
Defines the Server base class and specific federated learning server implementations
like FedAvg, FedProx, pFedMe, Ditto. Handles client management, aggregation,
and model distribution.
"""
import copy
import numpy as np # Keep for potential future use in aggregation or metrics
import torch
from typing import Dict, Optional, List, Tuple, Any, Iterator

# Import necessary components from other modules
from clients import Client, FedProxClient, PFedMeClient, DittoClient # Import specific client types
from helper import ModelDiversity, MetricsCalculator, cleanup_gpu # Import utilities
# Note: TrainerConfig, SiteData, ModelState are now defined WITHIN servers.py or clients.py
# Let's assume they are defined here for now, or adjust imports if they move to clients.py

# --- Data Structures (Potentially move to a separate 'common.py' or keep here) ---
from dataclasses import dataclass, field

@dataclass
class TrainerConfig:
    """Configuration for training parameters passed to clients/servers."""
    dataset_name: str
    device: str
    learning_rate: float
    batch_size: int
    epochs: int = 5 # Default local epochs
    rounds: int = 20 # Default communication rounds
    requires_personal_model: bool = False # Flag for personalized FL algorithms
    algorithm_params: Optional[Dict[str, Any]] = None # Algo-specific params (e.g., mu for FedProx)

    def __post_init__(self):
        # Ensure algorithm_params is a dict if not None
        if self.algorithm_params is None:
            self.algorithm_params = {}

@dataclass
class SiteData:
    """Holds DataLoader instances and metadata for a client/site."""
    site_id: str
    train_loader: torch.utils.data.DataLoader
    val_loader: torch.utils.data.DataLoader
    test_loader: torch.utils.data.DataLoader
    weight: float = 1.0 # Weight for aggregation (e.g., based on num_samples)
    num_samples: int = 0

    def __post_init__(self):
        # Calculate num_samples from train_loader dataset if not provided
        if self.num_samples == 0 and self.train_loader is not None and self.train_loader.dataset is not None:
            try:
                 self.num_samples = len(self.train_loader.dataset)
            except TypeError: # Handle datasets without __len__ (less common)
                 print(f"Warning: Could not determine num_samples for site {self.site_id}")
                 self.num_samples = 0 # Or estimate differently if possible

@dataclass
class ModelState:
    """Holds the state (model, optimizer, criterion, metrics) for a model."""
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    criterion: torch.nn.Module # Loss function
    best_loss: float = float('inf') # Track best validation loss
    best_model: Optional[torch.nn.Module] = None # State dict of the best model
    # Lists to store metrics per round/epoch
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[List[float]] = field(default_factory=list) # [[loss1], [loss2], ...] format
    val_scores: List[List[float]] = field(default_factory=list) # [[score1], [score2], ...] format
    test_losses: List[List[float]] = field(default_factory=list) # [[loss1], [loss2], ...] format
    test_scores: List[List[float]] = field(default_factory=list) # [[score1], [score2], ...] format

    def __post_init__(self):
        # Initialize best_model with a deep copy of the initial model state
        if self.best_model is None and self.model is not None:
            try:
                 # Ensure model is on a device to get parameters' device
                 device = next(self.model.parameters()).device
                 self.best_model = copy.deepcopy(self.model).to(device)
            except StopIteration: # Handle model with no parameters
                 self.best_model = copy.deepcopy(self.model) # Copy structure anyway

    def copy(self):
        """
        Creates a deep copy of the ModelState, including model architecture,
        weights, and optimizer state. Useful for initializing client states
        from the global state.
        """
        # Ensure the model has parameters before proceeding
        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = 'cpu' # Default device if model has no parameters

        # Create a new instance of the model and copy weights
        new_model = copy.deepcopy(self.model).to(device)

        # Create a new optimizer instance for the new model
        # Re-initialize optimizer state to avoid issues with parameter IDs
        new_optimizer = type(self.optimizer)(new_model.parameters(), **self.optimizer.defaults)
        # Deep copy optimizer state - this can be complex if state contains tensors
        # A common approach is to save and load state_dict, but requires temporary storage
        # Simpler approach for now: re-initialize state (might lose momentum etc.)
        # For more robust state copy, consider state_dict saving/loading or specific deepcopy logic
        # new_optimizer.load_state_dict(copy.deepcopy(self.optimizer.state_dict()))

        # Create a new ModelState instance, keeping metrics empty
        return ModelState(
            model=new_model,
            optimizer=new_optimizer, # Note: Optimizer state might be reset
            criterion=self.criterion # Criterion is usually stateless or shared
            # best_loss, best_model, and metric lists are initialized by default
        )

# --- Base Server Class ---
class Server:
    """Base server class providing core functionalities for federated learning."""

    def __init__(self, config: TrainerConfig, globalmodelstate: ModelState):
        self.config = config
        self.device = config.device
        # Flag indicating if clients maintain personalized models
        self.personal = config.requires_personal_model
        # Dictionary to store client instances {client_id: Client}
        self.clients: Dict[str, Client] = {}
        # Holds the global model state (model, optimizer, criterion, metrics)
        self.serverstate = globalmodelstate
        # Ensure the global model and best model are on the correct device
        self.serverstate.model = self.serverstate.model.to(self.device)
        if self.serverstate.best_model:
             self.serverstate.best_model = self.serverstate.best_model.to(self.device)
        # Placeholders for server type and tuning status (set by set_server_type)
        self.server_type: str = "BaseServer"
        self.tuning: bool = False # Default to evaluation run


    def set_server_type(self, name: str, tuning: bool):
        """Sets the server type name and tuning status."""
        self.server_type = name
        self.tuning = tuning
        print(f"Server type set to: {self.server_type}, Tuning: {self.tuning}")

    def _create_client(self, clientdata: SiteData, modelstate: ModelState,
                       personal_model: bool) -> Client:
        """
        Factory method to create a client instance.
        Subclasses can override this to create specific client types (FedProxClient, etc.).
        """
        # Default implementation creates a base Client instance
        print(f"Creating base Client for site {clientdata.site_id}")
        return Client(
            config=self.config,
            data=clientdata,
            modelstate=modelstate.copy(), # Give client a deep copy of the initial state
            metrics_calculator=MetricsCalculator(self.config.dataset_name),
            personal_model=personal_model # Pass flag for personalized models
        )

    def add_client(self, clientdata: SiteData):
        """Creates and adds a new client to the server's federation."""
        if not isinstance(clientdata, SiteData):
             print(f"Warning: Invalid clientdata provided to add_client (type: {type(clientdata)}). Skipping.")
             return

        print(f"Adding client: {clientdata.site_id} with {clientdata.num_samples} samples.")
        # Use the factory method to create the appropriate client type
        client = self._create_client(
            clientdata=clientdata,
            modelstate=self.serverstate, # Pass the current global state for initialization
            personal_model=self.personal
        )

        # Add the created client to the server's dictionary
        self.clients[clientdata.site_id] = client
        # Update aggregation weights whenever a client is added/removed
        self._update_client_weights()

    def _update_client_weights(self):
        """Calculates and updates client weights based on their number of samples."""
        if not self.clients:
            return # No clients to weight

        total_samples = sum(client.data.num_samples for client in self.clients.values() if client.data.num_samples > 0)

        if total_samples == 0:
            # Handle case where all clients have 0 samples (assign equal weight or error)
            print("Warning: Total samples across all clients is 0. Assigning equal weights.")
            num_clients = len(self.clients)
            weight = 1.0 / num_clients if num_clients > 0 else 0
            for client in self.clients.values():
                client.data.weight = weight
            return

        # Calculate weight = client_samples / total_samples
        for client in self.clients.values():
            client.data.weight = client.data.num_samples / total_samples
            # print(f"  Client {client.data.site_id}: weight = {client.data.weight:.4f}") # Verbose

    def train_round(self) -> Tuple[float, float, float]:
        """
        Runs one complete round of federated training.
        1. Triggers training on all clients.
        2. Aggregates results (losses, scores).
        3. Aggregates models (implemented by subclasses).
        4. Distributes the updated global model (implemented by subclasses).
        5. Updates the best performing global model based on validation loss.

        Returns:
            Tuple[float, float, float]: Aggregated training loss, validation loss,
                                        and validation score for the round.
        """
        if not self.clients:
             print("Warning: train_round called with no clients.")
             return 0.0, float('inf'), 0.0 # Return defaults indicating no training occurred

        round_train_loss = 0.0
        round_val_loss = 0.0
        round_val_score = 0.0

        # --- Client Training & Validation Phase ---
        for client_id, client in self.clients.items():
            # print(f"  Training client {client_id}...") # Verbose
            # Client trains for its configured number of local epochs
            # The 'personal' flag determines which model (global or personal) is trained
            client_train_loss = client.train(personal=self.personal)
            # Client validates using its current model state
            client_val_loss, client_val_score = client.validate(personal=self.personal)

            # Aggregate weighted metrics (use client's weight)
            round_train_loss += client_train_loss * client.data.weight
            round_val_loss += client_val_loss * client.data.weight
            round_val_score += client_val_score * client.data.weight

        # --- Server Aggregation & Distribution Phase ---
        # Aggregate model updates from clients (specific logic in subclasses)
        self.aggregate_models()
        # Distribute the newly aggregated global model back to clients
        self.distribute_global_model(test=False) # Distribute the current training model

        # --- Track Server-Side Metrics ---
        # Store aggregated metrics for the round (use list-of-lists format)
        self.serverstate.train_losses.append([round_train_loss])
        self.serverstate.val_losses.append([round_val_loss])
        self.serverstate.val_scores.append([round_val_score])

        # --- Update Best Model ---
        # Check if the current round's validation loss is the best seen so far
        if round_val_loss < self.serverstate.best_loss:
            print(f"  New best validation loss: {round_val_loss:.4f} (prev: {self.serverstate.best_loss:.4f})")
            self.serverstate.best_loss = round_val_loss
            # Save a deep copy of the current best model state
            self.serverstate.best_model = copy.deepcopy(self.serverstate.model)

        return round_train_loss, round_val_loss, round_val_score

    def test_global(self) -> Tuple[float, float]:
        """
        Tests the best performing global model on the test sets of all clients.

        Returns:
            Tuple[float, float]: Aggregated test loss and test score.
        """
        if not self.clients:
             print("Warning: test_global called with no clients.")
             return float('inf'), 0.0 # Return defaults indicating no testing

        round_test_loss = 0.0
        round_test_score = 0.0

        # Ensure clients have the latest *best* global model for testing
        self.distribute_global_model(test=True) # Signal to distribute the best model

        # --- Client Testing Phase ---
        for client_id, client in self.clients.items():
            # print(f"  Testing client {client_id}...") # Verbose
            # Client tests using the best global model state received
            # The 'personal' flag here should determine if the personal model's test score
            # is calculated (if applicable) or if the global model is tested.
            # For standard evaluation, we test the final global model (personal=False).
            client_test_loss, client_test_score = client.test(personal=False) # Test the global model

            # Aggregate weighted metrics
            round_test_loss += client_test_loss * client.data.weight
            round_test_score += client_test_score * client.data.weight

        # --- Track Server-Side Metrics ---
        self.serverstate.test_losses.append([round_test_loss])
        self.serverstate.test_scores.append([round_test_score])

        return round_test_loss, round_test_score

    def aggregate_models(self):
        """
        Placeholder for model aggregation logic.
        Subclasses (like FedAvgServer) must implement this.
        """
        # print("Warning: Base Server aggregate_models called - no aggregation performed.")
        pass # Base implementation does nothing

    def distribute_global_model(self, test: bool = False):
        """
        Placeholder for distributing the global model state to clients.
        Subclasses (like FLServer) must implement this.

        Args:
            test (bool): If True, distribute the best model state; otherwise,
                         distribute the current training model state.
        """
        # print("Warning: Base Server distribute_global_model called - no distribution performed.")
        pass # Base implementation does nothing


# --- Federated Learning Server Base Class ---
class FLServer(Server):
    """
    Base class for standard federated learning servers (like FedAvg, FedProx)
    that implement aggregation and distribution.
    """
    def aggregate_models(self):
        """
        Performs standard Federated Averaging (FedAvg) aggregation.
        Averages client model parameters weighted by client data size.
        """
        if not self.clients:
            return # Cannot aggregate without clients

        # Zero out the global model parameters before aggregation
        with torch.no_grad():
            for param in self.serverstate.model.parameters():
                param.data.zero_()

            # Accumulate weighted parameters from clients
            for client_id, client in self.clients.items():
                # Determine which model state to aggregate (global or personal)
                # For standard FL (FedAvg, FedProx), aggregate the state updated during client.train(personal=False)
                state_to_agg = client.global_state # FedAvg/FedProx update global state
                # Add weighted client parameters to the global model parameters
                for server_param, client_param in zip(self.serverstate.model.parameters(), state_to_agg.model.parameters()):
                    server_param.data.add_(client_param.data.detach(), alpha=client.data.weight)

    def distribute_global_model(self, test: bool = False):
        """
        Distributes the appropriate global model state dictionary to all clients.

        Args:
            test (bool): If True, distributes the best performing model state.
                         If False, distributes the current aggregated global model state.
        """
        if not self.clients:
            return

        # Select the state dictionary to distribute
        if test:
             # Use the best model found during validation
             if self.serverstate.best_model is not None:
                  state_dict_to_dist = self.serverstate.best_model.state_dict()
                  # print("Distributing best model state for testing.") # Verbose
             else:
                  # Fallback if best model wasn't recorded (shouldn't happen if validation ran)
                  print("Warning: Best model not available for testing, distributing current model state.")
                  state_dict_to_dist = self.serverstate.model.state_dict()
        else:
             # Use the current (just aggregated) global model state
             state_dict_to_dist = self.serverstate.model.state_dict()
             # print("Distributing current global model state for training.") # Verbose


        # Send the state dictionary to each client
        for client in self.clients.values():
            # Client's method updates its internal model(s)
            client.set_model_state(state_dict_to_dist, test)


# --- Specific Server Implementations ---

class FedAvgServer(FLServer):
    """
    Standard Federated Averaging server. Inherits aggregation and distribution
    from FLServer. Adds diversity calculation.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.diversity_calculator: Optional[ModelDiversity] = None
        self.diversity_metrics: Dict[str, List[float]] = {'weight_div': [], 'weight_orient': []}

    def _set_diversity_calculation(self):
        """Initializes the diversity calculator using the first two clients."""
        # Requires at least 2 clients to calculate pairwise diversity
        if len(self.clients) >= 2:
            client_ids = list(self.clients.keys())
            client_1 = self.clients[client_ids[0]]
            client_2 = self.clients[client_ids[1]]
            self.diversity_calculator = ModelDiversity(client_1, client_2)
            print("Diversity calculator initialized between clients:", client_ids[0], client_ids[1])
        else:
            print("Warning: Not enough clients (< 2) to initialize diversity calculator.")
            self.diversity_calculator = None


    def train_round(self) -> Tuple[float, float, float]:
        """
        Overrides train_round to calculate diversity metrics BEFORE aggregation.
        """
        if not self.clients:
             print("Warning: train_round called with no clients.")
             return 0.0, float('inf'), 0.0

        round_train_loss = 0.0
        round_val_loss = 0.0
        round_val_score = 0.0

        # --- Client Training & Validation Phase ---
        for client in self.clients.values():
            client_train_loss = client.train(personal=self.personal) # Should be False for FedAvg
            client_val_loss, client_val_score = client.validate(personal=self.personal)

            round_train_loss += client_train_loss * client.data.weight
            round_val_loss += client_val_loss * client.data.weight
            round_val_score += client_val_score * client.data.weight

        # --- Calculate Diversity (BEFORE aggregation) ---
        if self.diversity_calculator is None and len(self.clients) >= 2:
            self._set_diversity_calculation() # Initialize on first round with enough clients

        if self.diversity_calculator is not None:
            try:
                weight_div, weight_orient = self.diversity_calculator.calculate_weight_divergence()
                self.diversity_metrics['weight_div'].append(weight_div)
                self.diversity_metrics['weight_orient'].append(weight_orient)
            except Exception as e:
                 print(f"Error calculating diversity metrics: {e}")
                 # Append placeholder values?
                 self.diversity_metrics['weight_div'].append(np.nan)
                 self.diversity_metrics['weight_orient'].append(np.nan)


        # --- Server Aggregation & Distribution Phase ---
        self.aggregate_models()
        self.distribute_global_model(test=False)

        # --- Track Server-Side Metrics ---
        self.serverstate.train_losses.append([round_train_loss])
        self.serverstate.val_losses.append([round_val_loss])
        self.serverstate.val_scores.append([round_val_score])

        # --- Update Best Model ---
        if round_val_loss < self.serverstate.best_loss:
            print(f"  New best validation loss: {round_val_loss:.4f} (prev: {self.serverstate.best_loss:.4f})")
            self.serverstate.best_loss = round_val_loss
            self.serverstate.best_model = copy.deepcopy(self.serverstate.model)

        return round_train_loss, round_val_loss, round_val_score

class FedProxServer(FLServer):
    """
    Server implementation for FedProx.
    Relies on FLServer for aggregation/distribution. Creates FedProxClient instances.
    """
    def _create_client(self, clientdata: SiteData, modelstate: ModelState,
                       personal_model: bool = False) -> FedProxClient:
        """Overrides factory to create FedProxClient instances."""
        # FedProx modifies the global model state, so personal_model should be False
        if personal_model:
            print("Warning: FedProxServer received personal_model=True, but FedProx typically uses personal_model=False. Forcing False.")
            personal_model = False # Enforce correct behavior

        print(f"Creating FedProxClient for site {clientdata.site_id}")
        return FedProxClient(
            config=self.config,
            data=clientdata,
            modelstate=modelstate.copy(), # Pass a copy of the global state
            metrics_calculator=MetricsCalculator(self.config.dataset_name),
            personal_model=personal_model # Should be False
        )

class PFedMeServer(FLServer):
    """
    Server implementation for pFedMe.
    Uses standard FedAvg aggregation/distribution but creates pFedMe clients.
    """
    def _create_client(self, clientdata: SiteData, modelstate: ModelState,
                       personal_model: bool = True) -> PFedMeClient: # Default personal=True
        """Overrides factory to create PFedMeClient instances."""
        # pFedMe requires personalized models
        if not personal_model:
             print("Warning: PFedMeServer received personal_model=False. pFedMe requires personalized models. Forcing True.")
             personal_model = True # Enforce correct behavior

        print(f"Creating PFedMeClient for site {clientdata.site_id}")
        return PFedMeClient(
            config=self.config,
            data=clientdata,
            modelstate=modelstate.copy(), # Global state copy
            metrics_calculator=MetricsCalculator(self.config.dataset_name),
            personal_model=personal_model # Should be True
        )

class DittoServer(FLServer):
    """
    Server implementation for Ditto.
    Uses standard FedAvg aggregation/distribution for the global model.
    Relies on DittoClient for managing both global and personal models.
    """
    def _create_client(self, clientdata: SiteData, modelstate: ModelState,
                       personal_model: bool = True) -> DittoClient: # Default personal=True
        """Overrides factory to create DittoClient instances."""
        if not personal_model:
             print("Warning: DittoServer received personal_model=False. Ditto requires personalized models. Forcing True.")
             personal_model = True # Enforce correct behavior

        print(f"Creating DittoClient for site {clientdata.site_id}")
        return DittoClient(
            config=self.config,
            data=clientdata,
            modelstate=modelstate.copy(), # Global state copy
            metrics_calculator=MetricsCalculator(self.config.dataset_name),
            personal_model=personal_model # Should be True
        )

    def train_round(self) -> Tuple[float, float, float]:
        """
        Runs one round of Ditto training. Includes separate global and personal steps.
        Reports metrics based on the *personal* model validation.
        Updates best *global* model based on *global* model validation.

        Returns:
            Tuple[float, float, float]: Aggregated personal training loss,
                                        personal validation loss,
                                        and personal validation score for the round.
        """
        if not self.clients:
             print("Warning: train_round called with no clients.")
             return 0.0, float('inf'), 0.0

        # --- 1. Global Model Update Step ---
        global_train_loss = 0.0
        global_val_loss = 0.0
        global_val_score = 0.0
        print("  Ditto: Starting Global Model Update Phase...")
        for client in self.clients.values():
            # Train global model copy on client
            client_global_train_loss = client.train(personal=False)
            # Validate global model copy on client
            client_global_val_loss, client_global_val_score = client.validate(personal=False)

            global_train_loss += client_global_train_loss * client.data.weight
            global_val_loss += client_global_val_loss * client.data.weight
            global_val_score += client_global_val_score * client.data.weight

        # Aggregate global models and distribute the new global average
        self.aggregate_models() # Averages client.global_state.model updates
        self.distribute_global_model(test=False) # Sends aggregated model to client.global_state

        # --- Track Global Model Validation (for best model selection) ---
        # Note: Ditto doesn't explicitly track global metrics on the server state in the paper,
        # but we need it to select the best *global* model for saving.
        # Let's store it temporarily or decide if it should be part of serverstate.
        print(f"  Ditto: Global Phase Metrics - ValLoss: {global_val_loss:.4f}, ValScore: {global_val_score:.4f}")


        # --- 2. Personal Model Update Step ---
        personal_train_loss = 0.0
        personal_val_loss = 0.0
        personal_val_score = 0.0
        print("  Ditto: Starting Personal Model Update Phase...")
        for client in self.clients.values():
            # Train personal model on client (uses regularization towards current global)
            client_personal_train_loss = client.train(personal=True)
            # Validate personal model on client
            client_personal_val_loss, client_personal_val_score = client.validate(personal=True)

            personal_train_loss += client_personal_train_loss * client.data.weight
            personal_val_loss += client_personal_val_loss * client.data.weight
            personal_val_score += client_personal_val_score * client.data.weight

        # --- Track Server-Side Personal Metrics ---
        # These are the primary metrics reported for Ditto's performance during training
        self.serverstate.train_losses.append([personal_train_loss])
        self.serverstate.val_losses.append([personal_val_loss])
        self.serverstate.val_scores.append([personal_val_score])

        # --- Update Best Global Model ---
        # The 'best model' saved by the server is the best *global* model found
        # based on the validation performance *of the global model updates*.
        if global_val_loss < self.serverstate.best_loss:
            print(f"  New best *global* validation loss: {global_val_loss:.4f} (prev: {self.serverstate.best_loss:.4f})")
            self.serverstate.best_loss = global_val_loss
            self.serverstate.best_model = copy.deepcopy(self.serverstate.model) # Save the global model

        # Return the aggregated *personal* model metrics for the round
        return personal_train_loss, personal_val_loss, personal_val_score