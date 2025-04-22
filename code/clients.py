"""
Defines the base Client class and specific implementations for federated
learning algorithms like FedProx, pFedMe, and Ditto.
Handles local training, evaluation, and model state management.
"""
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Optional, Tuple, List, Iterator, Any
from dataclasses import dataclass, field

# Import necessary components from other modules
# Avoid importing Server classes here to prevent circular dependency if Server imports Client
# Instead, necessary info like config, model state is passed during __init__
from helper import move_to_device, cleanup_gpu, get_dice_score # General utilities
# Import data structures if they live in servers.py (or a common file)
from servers import TrainerConfig, SiteData, ModelState, MetricsCalculator


# --- Base Client Class ---
class Client:
    """
    Base class for a federated learning client. Handles local data, model state(s),
    training, validation, and testing.
    """
    def __init__(self,
                 config: TrainerConfig,
                 data: SiteData,
                 modelstate: ModelState, # Initial global model state from server
                 metrics_calculator: MetricsCalculator,
                 personal_model: bool = False): # Flag to create personal state
        self.config = config
        self.data = data
        self.device = config.device
        self.metrics_calculator = metrics_calculator
        self.site_id = data.site_id # Convenience accessor

        # Initialize model states
        # Everyone gets a global state (initially from server)
        self.global_state: ModelState = modelstate
        # Personal state is a separate copy if required by the algorithm
        self.personal_state: Optional[ModelState] = None
        if personal_model:
             self.personal_state = self.global_state.copy()
             print(f"Client {self.site_id}: Initialized personal model state.")

    def get_client_state(self, personal: bool) -> ModelState:
        """
        Returns the appropriate ModelState (global or personal) based on the flag.

        Args:
            personal (bool): If True, return personal_state; otherwise, global_state.

        Returns:
            ModelState: The requested model state object.

        Raises:
            ValueError: If personal_state is requested but not initialized.
        """
        if personal:
            if self.personal_state is None:
                raise ValueError(f"Personal model state requested for client {self.site_id}, but it was not initialized.")
            return self.personal_state
        else:
            return self.global_state

    def set_model_state(self, state_dict: Dict, test: bool = False):
        """
        Updates the client's model state(s) from a received state dictionary.
        Usually updates the global state copy. If testing, updates the best model.

        Args:
            state_dict (Dict): The model state dictionary to load.
            test (bool): If True, load into the `best_model` attribute for testing.
                         Otherwise, load into the main `model` attribute.
        """
        # Primarily updates the client's copy of the global model state
        state_to_update = self.global_state

        target_model = state_to_update.best_model if test else state_to_update.model

        if target_model is None and test:
             print(f"Warning: Client {self.site_id} has no best_model to load state into for testing.")
             # Optionally load into the main model as a fallback?
             # target_model = state_to_update.model
             return # Skip loading if target model is None
        elif target_model is None:
             print(f"Warning: Client {self.site_id} model is None. Cannot load state_dict.")
             return


        try:
            target_model.load_state_dict(state_dict)
            # print(f"Client {self.site_id}: Updated {'best' if test else 'main'} model state.") # Verbose
        except RuntimeError as e:
            print(f"ERROR loading state_dict for client {self.site_id}: {e}")
            # This might happen if architectures mismatch, log details
            print("  State dict keys:", state_dict.keys())
            print("  Model keys:", target_model.state_dict().keys())
        except Exception as e:
             print(f"Unexpected error loading state_dict for client {self.site_id}: {e}")


    def update_best_model(self, loss: float, personal: bool):
        """
        Updates the best model state if the current validation loss is lower.

        Args:
            loss (float): The current validation loss.
            personal (bool): Whether to update the personal or global best model state.
        """
        state = self.get_client_state(personal) # Get the correct state (personal or global)

        if loss < state.best_loss:
            state.best_loss = loss
            # Store a deep copy of the *current* model weights as the new best
            state.best_model = copy.deepcopy(state.model).to(self.device)
            # print(f"Client {self.site_id}: Updated best {'personal' if personal else 'global'} model (loss: {loss:.4f}).") # Verbose
            return True # Indicate that the best model was updated
        return False # Best model not updated

    def train_epoch(self, personal: bool) -> float:
        """
        Performs one epoch of local training on the client's training data.

        Args:
            personal (bool): If True, trains the personal model; otherwise,
                             trains the global model copy.

        Returns:
            float: The average training loss for the epoch.
        """
        try:
            # Get the relevant model state (personal or global)
            state = self.get_client_state(personal)
            model = state.model.to(self.device)
            model.train() # Set model to training mode

            total_loss = 0.0
            num_batches = 0

            if self.data.train_loader is None or len(self.data.train_loader) == 0:
                print(f"Warning: Client {self.site_id} has no training data/loader.")
                return 0.0 # Or handle as appropriate

            for batch_idx, batch in enumerate(self.data.train_loader):
                # Ensure batch is a tuple/list (handle different loader types)
                if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                    print(f"Warning: Unexpected batch format from DataLoader for client {self.site_id}. Skipping batch.")
                    continue
                batch_x, batch_y = batch[0], batch[1]

                # Move data to the correct device
                batch_x = move_to_device(batch_x, self.device)
                batch_y = move_to_device(batch_y, self.device)

                # Perform training step
                state.optimizer.zero_grad()
                outputs = model(batch_x)

                # Reshape labels if necessary (e.g., for CrossEntropyLoss)
                # TODO: Consider making label processing more robust or configurable
                if state.criterion.__class__.__name__ == 'CrossEntropyLoss':
                    if batch_y.ndim == 2 and batch_y.shape[1] == 1:
                        batch_y = batch_y.squeeze(1)
                    # Ensure labels are long type for CrossEntropy
                    batch_y = batch_y.long()
                # Add handling for other loss types if needed (e.g., BCEWithLogitsLoss needs float)

                loss = state.criterion(outputs, batch_y)
                loss.backward()
                state.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            # Calculate average loss for the epoch
            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
            # Store training loss (potentially replace with per-epoch tracking if needed)
            # Current structure assumes one loss value per train() call
            # state.train_losses.append(avg_loss) # Moved to train() method
            return avg_loss

        except Exception as e:
             print(f"Error during training epoch for client {self.site_id}: {e}")
             traceback.print_exc()
             return float('inf') # Return high loss on error
        finally:
            # Attempt to clean up tensors to free memory
            del batch_x, batch_y, outputs, loss
            # Optionally move model back to CPU if memory is very constrained
            # model.to('cpu')
            # cleanup_gpu() # Call GPU cleanup if needed

    def train(self, personal: bool) -> float:
        """
        Runs local training for the configured number of epochs.

        Args:
            personal (bool): If True, trains the personal model; otherwise,
                             trains the global model copy.

        Returns:
            float: The average training loss over the final epoch.
        """
        final_epoch_loss = 0.0
        model_type = "personal" if personal else "global"
        # print(f"Client {self.site_id}: Starting training for {self.config.epochs} epochs ({model_type} model).") # Verbose
        for epoch in range(self.config.epochs):
            final_epoch_loss = self.train_epoch(personal)
            # print(f"  Epoch {epoch+1}/{self.config.epochs}, Loss: {final_epoch_loss:.4f}") # Verbose progress

        # Store the final epoch's average loss
        state = self.get_client_state(personal)
        # Append loss as a list containing a single value to match server format
        state.train_losses.append([final_epoch_loss])

        return final_epoch_loss

    def evaluate(self,
                 loader: DataLoader,
                 personal: bool,
                 use_best_model: bool # Changed from 'validate' for clarity
                ) -> Tuple[float, float]:
        """
        Evaluates the specified model (current or best) on the provided DataLoader.

        Args:
            loader: The DataLoader to evaluate on (e.g., validation or test).
            personal: If True, evaluate the personal model; otherwise, the global model.
            use_best_model: If True, use the `best_model` state; otherwise, use the current `model` state.

        Returns:
            Tuple[float, float]: Average loss and calculated score.
        """
        if loader is None or len(loader) == 0:
            # print(f"Warning: Empty or None DataLoader provided for evaluation on client {self.site_id}.") # Verbose
            return float('inf'), 0.0 # Return default values indicating no evaluation

        try:
            # Get the relevant model state
            state = self.get_client_state(personal)

            # Select the model to evaluate (current training state or best validated state)
            model_to_eval = state.best_model if use_best_model else state.model

            if model_to_eval is None:
                # This might happen if use_best_model is True but no validation has occurred
                print(f"Warning: Model state for evaluation ('{'best' if use_best_model else 'current'}') is None "
                      f"for client {self.site_id} ({'personal' if personal else 'global'}). Returning default metrics.")
                return float('inf'), 0.0

            model = model_to_eval.to(self.device)
            model.eval() # Set model to evaluation mode

            total_loss = 0.0
            all_predictions = []
            all_labels = []
            num_batches = 0

            with torch.no_grad(): # Disable gradient calculations
                for batch_idx, batch in enumerate(loader):
                    if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                        print(f"Warning: Unexpected batch format during evaluation for client {self.site_id}. Skipping batch.")
                        continue
                    batch_x, batch_y = batch[0], batch[1]

                    batch_x = move_to_device(batch_x, self.device)
                    batch_y = move_to_device(batch_y, self.device)

                    outputs = model(batch_x)

                    # Process labels same way as in training
                    if state.criterion.__class__.__name__ == 'CrossEntropyLoss':
                         if batch_y.ndim == 2 and batch_y.shape[1] == 1: batch_y = batch_y.squeeze(1)
                         batch_y = batch_y.long()
                    # Add handling for other loss types (e.g., Dice expects floats)
                    elif self.metrics_calculator.dataset_name == 'IXITiny':
                         batch_y = batch_y.float() # Ensure target is float for Dice loss/score

                    loss = state.criterion(outputs, batch_y)
                    total_loss += loss.item()

                    # Process outputs for metric calculation
                    # Note: For Dice, outputs might already be probabilities/logits needed by the score function
                    if self.metrics_calculator.dataset_name == 'IXITiny':
                         # Keep predictions as tensors for get_dice_score
                         processed_preds = outputs
                         processed_labels = batch_y
                    else:
                         # Process for standard metrics (argmax, numpy conversion)
                         processed_preds, processed_labels = self.metrics_calculator.process_predictions(
                             outputs, batch_y
                         )

                    all_predictions.append(processed_preds) # Store processed preds/labels
                    all_labels.append(processed_labels)
                    num_batches += 1

            # Calculate average loss
            avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')

            # Concatenate results and calculate final score
            if not all_labels or not all_predictions:
                print(f"Warning: No labels or predictions collected during evaluation for client {self.site_id}.")
                score = 0.0
            else:
                 # Handle different data types (numpy arrays vs tensors for Dice)
                 if self.metrics_calculator.dataset_name == 'IXITiny':
                      # Dice expects tensors
                      all_predictions_tensor = torch.cat(all_predictions, dim=0)
                      all_labels_tensor = torch.cat(all_labels, dim=0)
                      score = self.metrics_calculator.calculate_metrics(
                          all_labels_tensor, all_predictions_tensor
                      )
                 else:
                      # Other metrics expect numpy arrays
                      all_predictions_np = np.concatenate(all_predictions, axis=0)
                      all_labels_np = np.concatenate(all_labels, axis=0)
                      score = self.metrics_calculator.calculate_metrics(
                          all_labels_np, all_predictions_np
                      )

            return avg_loss, score

        except Exception as e:
             print(f"Error during evaluation for client {self.site_id}: {e}")
             traceback.print_exc()
             return float('inf'), 0.0 # Return default values on error
        finally:
            # Clean up tensors
            del batch_x, batch_y, outputs, loss
            if 'model' in locals() and model is not None:
                 model.to('cpu') # Move model back to CPU after eval if desired
            # cleanup_gpu()

    def validate(self, personal: bool) -> Tuple[float, float]:
        """
        Validates the current model state (personal or global) using the
        validation loader. Updates the best model if performance improves.

        Args:
            personal: If True, validate the personal model; otherwise, the global model.

        Returns:
            Tuple[float, float]: Validation loss and score.
        """
        state = self.get_client_state(personal)
        # Evaluate the *current* model state (use_best_model=False)
        val_loss, val_score = self.evaluate(
            loader=self.data.val_loader,
            personal=personal,
            use_best_model=False
        )

        # Append metrics (append list to match server format)
        state.val_losses.append([val_loss])
        state.val_scores.append([val_score])

        # Update the best model based on this validation loss
        self.update_best_model(val_loss, personal)

        return val_loss, val_score

    def test(self, personal: bool) -> Tuple[float, float]:
        """
        Tests the best performing model state (personal or global) using the
        test loader.

        Args:
            personal: If True, test the best personal model; otherwise, the best global model.

        Returns:
            Tuple[float, float]: Test loss and score.
        """
        state = self.get_client_state(personal)
        # Evaluate the *best* model state found during training (use_best_model=True)
        test_loss, test_score = self.evaluate(
            loader=self.data.test_loader,
            personal=personal,
            use_best_model=True
        )

        # Append metrics (append list to match server format)
        state.test_losses.append([test_loss])
        state.test_scores.append([test_score])

        return test_loss, test_score

# --- Algorithm-Specific Client Implementations ---

class FedProxClient(Client):
    """Client implementation for FedProx."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Retrieve regularization parameter mu from config
        if 'reg_param' not in self.config.algorithm_params:
             # Provide a default or raise error if mu is critical
             default_mu = 0.01
             print(f"Warning: 'reg_param' (mu) not found in algorithm_params for FedProx. Using default: {default_mu}")
             self.reg_param = default_mu
             # OR: raise ValueError("FedProx requires 'reg_param' (mu) in algorithm_params.")
        else:
            self.reg_param = self.config.algorithm_params['reg_param']
        print(f"FedProxClient {self.site_id} initialized with mu = {self.reg_param}")

    def train_epoch(self, personal: bool = False) -> float:
        """Performs one epoch of FedProx training, adding the proximal term."""
        if personal:
             # FedProx algorithm description typically updates the client's copy
             # of the global model state, not a separate personal model.
             print("Warning: FedProxClient training called with personal=True. "
                   "FedProx standardly updates the global model state copy.")
             # Proceeding with personal=False logic as intended by FedProx.

        # FedProx modifies the client's copy of the global model state
        state = self.get_client_state(personal=False)
        model = state.model.to(self.device)
        model.train() # Set to training mode

        # --- Crucial: Get reference to the *initial* global model for this round ---
        # This state is held in self.global_state and assumed to be set by the
        # server via distribute_global_model *before* client training starts.
        # We need a detached copy of these initial parameters.
        try:
             global_model_params_initial = [
                 p.detach().clone() for p in self.global_state.model.parameters()
             ]
        except Exception as e:
             print(f"Error cloning initial global model parameters for FedProx: {e}")
             return float('inf')


        total_original_loss = 0.0
        num_batches = 0

        if self.data.train_loader is None or len(self.data.train_loader) == 0:
            print(f"Warning: Client {self.site_id} has no training data/loader.")
            return 0.0

        try:
            for batch_idx, batch in enumerate(self.data.train_loader):
                if not isinstance(batch, (list, tuple)) or len(batch) < 2: continue # Skip malformed batch
                batch_x, batch_y = batch[0], batch[1]
                batch_x = move_to_device(batch_x, self.device)
                batch_y = move_to_device(batch_y, self.device)

                state.optimizer.zero_grad()
                outputs = model(batch_x)

                # 1. Calculate standard task loss
                if state.criterion.__class__.__name__ == 'CrossEntropyLoss':
                    if batch_y.ndim == 2 and batch_y.shape[1] == 1: batch_y = batch_y.squeeze(1)
                    batch_y = batch_y.long()
                # Add handling for other loss types if necessary
                loss = state.criterion(outputs, batch_y)

                # 2. Calculate FedProx proximal term
                proximal_term = self.compute_proximal_term(
                    model.parameters(),        # Current local model params
                    global_model_params_initial # Initial global model params from round start
                )

                # 3. Combine losses
                total_loss_batch = loss + proximal_term
                total_loss_batch.backward() # Calculate gradients based on combined loss

                state.optimizer.step() # Update local model weights

                total_original_loss += loss.item() # Track original task loss for reporting
                num_batches += 1

            # Calculate average original loss for the epoch
            avg_loss = total_original_loss / num_batches if num_batches > 0 else 0.0
            # Note: FedProx paper doesn't explicitly track train_losses this way,
            # but it's useful for monitoring. The state updated is the client's global_state copy.
            # state.train_losses.append(avg_loss) # Moved to train() method

            return avg_loss

        except Exception as e:
             print(f"Error during FedProx training epoch for client {self.site_id}: {e}")
             traceback.print_exc()
             return float('inf')
        finally:
             del batch_x, batch_y, outputs, loss, proximal_term, total_loss_batch, global_model_params_initial
             # cleanup_gpu() # Optional cleanup

    def compute_proximal_term(self,
                              model_params: Iterator[torch.nn.Parameter],
                              reference_params: List[torch.Tensor]
                             ) -> torch.Tensor:
        """Calculates the FedProx proximal term: (mu / 2) * ||w - w_global||^2."""
        proximal_term = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        try:
            for param, ref_param in zip(model_params, reference_params):
                 if param.requires_grad: # Only include trainable parameters
                     # Ensure reference parameter is on the correct device
                     ref_param_device = ref_param.to(self.device)
                     # Calculate squared Euclidean distance (L2 norm squared)
                     param_diff_norm_sq = torch.sum(torch.pow(param - ref_param_device, 2))
                     proximal_term += param_diff_norm_sq
        except Exception as e:
            print(f"Error computing proximal term: {e}")
            # Return zero tensor or handle error as appropriate
            return torch.tensor(0.0, device=self.device, dtype=torch.float32)

        return (self.reg_param / 2.0) * proximal_term

class PFedMeClient(Client):
    """Client implementation for pFedMe."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # pFedMe requires personal model state
        if self.personal_state is None:
            raise ValueError("PFedMeClient requires personal_model=True during initialization.")
        # Get regularization parameter lambda from config
        if 'reg_param' not in self.config.algorithm_params:
             # Provide a default or raise error if lambda is critical
             default_lambda = 15.0 # Example default from paper, might need tuning
             print(f"Warning: 'reg_param' (lambda) not found in algorithm_params for pFedMe. Using default: {default_lambda}")
             self.reg_param = default_lambda
             # OR: raise ValueError("pFedMe requires 'reg_param' (lambda) in algorithm_params.")
        else:
            self.reg_param = self.config.algorithm_params['reg_param']
        print(f"PFedMeClient {self.site_id} initialized with lambda = {self.reg_param}")
        # pFedMe often involves multiple local steps (k) and learning rate eta
        # These could also be passed via algorithm_params if needed, e.g., self.k_steps, self.eta_lr

    def train_epoch(self, personal: bool = True) -> float:
        """
        Performs one epoch of pFedMe training.
        This typically involves multiple local steps updating the personal model
        while keeping the global model fixed as a reference for regularization.

        Args:
            personal (bool): Should always be True for pFedMe's main operation.

        Returns:
            float: The average task loss (excluding regularization) for the epoch.
        """
        if not personal:
             print("Warning: pFedMeClient train_epoch called with personal=False. pFedMe operates on the personal model.")
             # Decide behavior: train global model copy? Return 0? For now, return 0 loss.
             return 0.0

        # Get personal model state
        state = self.get_client_state(personal=True)
        model = state.model.to(self.device)
        model.train() # Set personal model to training mode

        # Get reference to the fixed global model received from the server
        # (Assumed to be stored correctly in self.global_state.model)
        global_model = self.global_state.model.to(self.device)
        # It's crucial that global_model parameters are not updated during this epoch
        # Detaching or setting requires_grad=False might be safer depending on implementation details
        # global_model_params = [p.detach() for p in global_model.parameters()]

        total_task_loss = 0.0
        num_batches = 0

        if self.data.train_loader is None or len(self.data.train_loader) == 0:
            print(f"Warning: Client {self.site_id} has no training data/loader.")
            return 0.0

        # --- pFedMe Local Update Loop ---
        # The original pFedMe involves multiple local steps (approximating Moreau envelope)
        # Let's simulate this within the epoch concept
        k_steps = self.config.algorithm_params.get('k_steps', 10) # Example: 10 local steps per "epoch"

        try:
            for k in range(k_steps):
                epoch_step_loss = 0.0
                num_step_batches = 0
                for batch_idx, batch in enumerate(self.data.train_loader):
                     if not isinstance(batch, (list, tuple)) or len(batch) < 2: continue
                     batch_x, batch_y = batch[0], batch[1]
                     batch_x = move_to_device(batch_x, self.device)
                     batch_y = move_to_device(batch_y, self.device)

                     state.optimizer.zero_grad()
                     outputs = model(batch_x)

                     # Calculate task loss
                     if state.criterion.__class__.__name__ == 'CrossEntropyLoss':
                         if batch_y.ndim == 2 and batch_y.shape[1] == 1: batch_y = batch_y.squeeze(1)
                         batch_y = batch_y.long()
                     loss = state.criterion(outputs, batch_y)

                     # Calculate pFedMe regularization term (|| theta - w ||^2)
                     proximal_term = self.compute_proximal_term(
                         model.parameters(),
                         global_model.parameters(), # Regularize towards the fixed global model
                     )

                     # Combine task loss and regularization
                     total_batch_loss = loss + proximal_term
                     total_batch_loss.backward()

                     # Update personal model parameters (theta)
                     state.optimizer.step()

                     epoch_step_loss += loss.item() # Track task loss
                     num_step_batches += 1

                # Optional: Track loss per k-step if needed
                # print(f"  Client {self.site_id} pFedMe Step {k+1}/{k_steps} Avg Loss: {epoch_step_loss / num_step_batches:.4f}")

            # Average task loss over the last k-step (or the whole epoch)
            avg_loss = epoch_step_loss / num_step_batches if num_step_batches > 0 else 0.0
            # state.train_losses.append(avg_loss) # Moved to train() method
            return avg_loss

        except Exception as e:
             print(f"Error during pFedMe training epoch for client {self.site_id}: {e}")
             traceback.print_exc()
             return float('inf')
        finally:
             del batch_x, batch_y, outputs, loss, proximal_term, total_batch_loss
             # cleanup_gpu()

    def train(self, personal: bool = True) -> float:
        """Main training loop, calls train_epoch multiple times."""
        # pFedMe's primary operation is on the personal model
        if not personal:
            print("Warning: pFedMeClient train called with personal=False. Executing personal training instead.")
            personal = True

        final_loss = 0.0
        # pFedMe paper uses local epochs concept differently, often implying multiple
        # batches or steps (k). Here, self.config.epochs maps to outer loops.
        for epoch in range(self.config.epochs):
            final_loss = self.train_epoch(personal=True)

        # Store the final epoch's average loss in the personal state
        state = self.get_client_state(personal=True)
        state.train_losses.append([final_loss]) # Match list-of-lists format

        return final_loss

    def compute_proximal_term(self,
                              model_params: Iterator[torch.nn.Parameter],
                              reference_params: Iterator[torch.nn.Parameter]
                             ) -> torch.Tensor:
        """Calculates the pFedMe regularization term: lambda/2 * ||theta - w||^2"""
        proximal_term = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        try:
            for param, ref_param in zip(model_params, reference_params):
                 # pFedMe regularizes all parameters (typically)
                 # Ensure reference parameter is detached and on the correct device
                 ref_param_device = ref_param.detach().to(self.device)
                 param_diff_norm_sq = torch.sum(torch.pow(param - ref_param_device, 2))
                 proximal_term += param_diff_norm_sq
        except Exception as e:
            print(f"Error computing pFedMe proximal term: {e}")
            return torch.tensor(0.0, device=self.device, dtype=torch.float32)

        return (self.reg_param / 2.0) * proximal_term

class DittoClient(Client):
    """Client implementation for Ditto."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Ditto requires personal model state
        if self.personal_state is None:
            raise ValueError("DittoClient requires personal_model=True during initialization.")
        # Get regularization parameter lambda from config
        if 'reg_param' not in self.config.algorithm_params:
             default_lambda = 0.1 # Example default, needs tuning
             print(f"Warning: 'reg_param' (lambda) not found for Ditto. Using default: {default_lambda}")
             self.reg_param = default_lambda
             # OR: raise ValueError("Ditto requires 'reg_param' (lambda) in algorithm_params.")
        else:
            self.reg_param = self.config.algorithm_params['reg_param']
        print(f"DittoClient {self.site_id} initialized with lambda = {self.reg_param}")

    def train_epoch(self, personal: bool) -> float:
        """
        Performs one epoch of local training for either the global or personal model.
        If training the personal model, applies gradient regularization.

        Args:
            personal: If True, train personal model; else, train global model copy.

        Returns:
            Average task loss for the epoch.
        """
        # Use standard training epoch if updating the global model copy
        if not personal:
            return super().train_epoch(personal=False)

        # --- Training the Personalized Model (personal=True) ---
        else:
            try:
                # Get personal model state
                state = self.get_client_state(personal=True)
                model = state.model.to(self.device)
                model.train() # Set personal model to training mode

                # Get reference to the fixed global model (w) received from the server
                global_model = self.global_state.model.to(self.device)
                # Ensure global model parameters are not updated
                global_model.eval() # Set to eval mode might be safer than detach

                total_loss = 0.0
                num_batches = 0

                if self.data.train_loader is None or len(self.data.train_loader) == 0:
                    print(f"Warning: Client {self.site_id} has no training data/loader.")
                    return 0.0

                for batch_idx, batch in enumerate(self.data.train_loader):
                    if not isinstance(batch, (list, tuple)) or len(batch) < 2: continue
                    batch_x, batch_y = batch[0], batch[1]
                    batch_x = move_to_device(batch_x, self.device)
                    batch_y = move_to_device(batch_y, self.device)

                    state.optimizer.zero_grad()
                    outputs = model(batch_x)

                    # Calculate task loss (F_k(v_k))
                    if state.criterion.__class__.__name__ == 'CrossEntropyLoss':
                        if batch_y.ndim == 2 and batch_y.shape[1] == 1: batch_y = batch_y.squeeze(1)
                        batch_y = batch_y.long()
                    loss = state.criterion(outputs, batch_y)

                    # Calculate gradients of the task loss w.r.t. personal model (v_k)
                    loss.backward()

                    # Add Ditto's gradient regularization term: lambda * (v_k - w)
                    # This modifies the gradients before the optimizer step
                    self.add_gradient_regularization(
                        model.parameters(),
                        global_model.parameters() # Use fixed global model w
                    )

                    # Update personal model parameters (v_k) using the modified gradient
                    state.optimizer.step()
                    total_loss += loss.item() # Track only the task loss
                    num_batches += 1

                # Calculate average task loss for the epoch
                avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
                # state.train_losses.append(avg_loss) # Moved to train() method
                return avg_loss

            except Exception as e:
                 print(f"Error during Ditto personal training epoch for client {self.site_id}: {e}")
                 traceback.print_exc()
                 return float('inf')
            finally:
                 del batch_x, batch_y, outputs, loss
                 # cleanup_gpu()

    def add_gradient_regularization(self,
                                    model_params: Iterator[torch.nn.Parameter],
                                    reference_params: Iterator[torch.nn.Parameter]):
        """Adds Ditto regularization term lambda * (param - ref_param) directly to gradients."""
        try:
            for param, ref_param in zip(model_params, reference_params):
                # Apply only if the parameter has a gradient computed
                if param.grad is not None:
                    # Ensure reference param is on the same device and detached
                    ref_param_device = ref_param.detach().to(param.device)
                    # Calculate regularization term: lambda * (v_k - w)
                    reg_term = self.reg_param * (param.detach() - ref_param_device) # Use detached param to avoid modifying buffer in-place
                    # Add regularization term to the existing gradient
                    param.grad.add_(reg_term)
        except Exception as e:
             print(f"Error adding Ditto gradient regularization: {e}")
             # Decide how to handle: raise error, skip regularization?