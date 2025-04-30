# clients.py
"""
Client implementations for Federated Learning. Streamlined version.
Uses ModelState for state, TrainingManager for utils, direct overrides for algorithms.
"""
import copy
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Optional, Tuple, List, Iterator, Any, Callable, Union
from helper import gpu_scope, TrainerConfig, SiteData, ModelState, TrainingManager, MetricsCalculator

# =============================================================================
# == Base Client Class ==
# =============================================================================
class Client:
    """Base FL client manages model states and training execution."""
    def __init__(self,
                 config: TrainerConfig,
                 data: SiteData,
                 initial_global_state: ModelState, # Has model/criterion from server
                 metric_fn: Callable,
                 personal_model: bool = False):
        self.config = config
        self.data = data
        self.metric_fn = metric_fn
        self.site_id = data.site_id
        self.requires_personal_model = personal_model
        self.training_manager = TrainingManager(config.device)
        self.class_weights = self._calculate_class_weights()

        # Initialize ModelStates
        self.global_state = ModelState(
            model=copy.deepcopy(initial_global_state.model).cpu(),
            criterion=initial_global_state.criterion
        )
        self.global_state.optimizer = self._create_optimizer(self.global_state.model)

        self.personal_state: Optional[ModelState] = None
        if self.requires_personal_model:
            self.personal_state = ModelState(
                model=copy.deepcopy(initial_global_state.model).cpu(),
                criterion=initial_global_state.criterion
            )
            self.personal_state.optimizer = self._create_optimizer(self.personal_state.model)

    def _create_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """Creates optimizer for a given model instance (CPU)."""
        wd = self.config.algorithm_params.get('weight_decay', 1e-4)
        lr = self.config.learning_rate
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, eps = 1e-8)

    def _get_state(self, personal: bool) -> ModelState:
        """Helper to get the correct state object."""
        state = self.personal_state if personal and self.requires_personal_model else self.global_state
        if state is None: raise RuntimeError(f"State {'personal' if personal else 'global'} not available.")
        return state
    
    def _calculate_class_weights(self):
        """
        Calculate class weights based on training data distribution.
        """
        all_labels = []
        for batch in self.data.train_loader:
            y = batch[1]
            all_labels.append(y.cpu())
        all_labels_tensor = torch.cat(all_labels)
        label_counts = torch.bincount(all_labels_tensor.flatten())
        label_counts = torch.clamp(label_counts, min=1)
        weights = 1.0 / label_counts.float()
        total_weight = weights.sum()
        weights = weights * len(weights) / total_weight
        return weights if self.config.use_weighted_loss else None
    
    def set_model_state(self, state_dict: Dict, test: bool = False):
        """Loads state dict into the client's global model state (CPU)."""
        self.global_state.load_current_model_state_dict(state_dict)

    def _train_batch(self, model: nn.Module, optimizer: optim.Optimizer, criterion: Union[nn.Module, Callable], batch_x: Any, batch_y: Any) -> float:
        """
        Performs a single training step. Assumes model, batch_x, batch_y on compute_device.
        Base implementation for FedAvg. Subclasses override this for algorithm logic.
        """
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y, self.class_weights)
        loss.backward()
        optimizer.step()
        return loss.item()

    def _process_epoch(self, 
                      loader: DataLoader, 
                      model: nn.Module, 
                      criterion: Union[nn.Module, Callable],
                      is_training: bool, 
                      optimizer: Optional[optim.Optimizer] = None) -> Tuple[float, List, List]:
        """
        Unified function to process one epoch for either training or evaluation.
        
        Args:
            loader: The data loader to use
            model: The model to train/evaluate 
            criterion: Loss function
            is_training: Whether this is a training epoch
            optimizer: Optimizer (required for training only)
            
        Returns:
            Tuple of (avg_loss, predictions_list, labels_list)
        """
        # Set model to correct mode and device
        model = model.to(self.training_manager.compute_device)
        if is_training:
            model.train()
        else:
            model.eval()

        epoch_loss = 0.0
        num_batches = 0
        epoch_predictions_cpu, epoch_labels_cpu = [], []
        context = torch.enable_grad() if is_training else torch.no_grad()

        with gpu_scope():
            with context:
                for batch in loader:
                    prepared_batch = self.training_manager.prepare_batch(batch, criterion)
                    batch_x_dev, batch_y_dev, batch_y_orig_cpu = prepared_batch
                    if is_training:
                        batch_loss = self._train_batch(model, optimizer, criterion, batch_x_dev, batch_y_dev)
                        epoch_loss += batch_loss
                    else:  # Evaluation
                        outputs = model(batch_x_dev)
                        loss = criterion(outputs, batch_y_dev, self.class_weights)
                        epoch_loss += loss.item()
                        epoch_predictions_cpu.append(outputs.detach().cpu())
                        epoch_labels_cpu.append(batch_y_orig_cpu)

                    num_batches += 1
                    # Cleanup temporary tensors
                    del batch_x_dev, batch_y_dev

            # Ensure model is back on CPU after processing
            model.to(self.training_manager.cpu_device)

        avg_loss = epoch_loss / num_batches if num_batches > 0 else (0.0 if is_training else float('inf'))
        return avg_loss, epoch_predictions_cpu, epoch_labels_cpu

    # --- Public API ---

    def train(self, personal: bool) -> Tuple[float, Optional[Dict]]:
        """Runs local training for multiple epochs."""
        state = self._get_state(personal)
        final_epoch_loss = float('inf')
        for _ in range(self.config.epochs):
            avg_loss, _, _ = self._process_epoch(
                loader=self.data.train_loader,
                model=state.model,
                criterion=state.criterion,
                is_training=True,
                optimizer=state.optimizer
            )
            final_epoch_loss = avg_loss
            if avg_loss == float('inf'):
                break  # Stop if epoch failed
                
        return final_epoch_loss, state.get_current_model_state_dict()

    def evaluate(self, loader: Optional[DataLoader], personal: bool, use_best: bool) -> Tuple[float, float]:
        """Evaluates a specified model state (current or best)."""
        state = self._get_state(personal)            
        state_dict_to_eval = state.get_best_model_state_dict() if use_best else state.get_current_model_state_dict()
        eval_model = copy.deepcopy(state.model).cpu()
        eval_model.load_state_dict(state_dict_to_eval)
        avg_loss, predictions_cpu, labels_cpu = self._process_epoch(
            loader=loader,
            model=eval_model,
            criterion=state.criterion,
            is_training=False
        )
        # Calculate score on CPU results
        score = 0.0
        all_preds = torch.cat(predictions_cpu, dim=0)
        all_labels = torch.cat(labels_cpu, dim=0)
        score = self.metric_fn(all_labels, all_preds)

        del eval_model  # Cleanup temporary model
        return avg_loss, score

    def validate(self, personal: bool) -> Tuple[float, float]:
        """Validates the *current* model state and updates the *best* state."""
        state = self._get_state(personal)
        val_loss, val_score = self.evaluate(self.data.val_loader, personal, use_best=False)
        if val_loss != float('inf'):
            state.update_best_state(val_loss)  # Update best state within the ModelState
        return val_loss, val_score

    def test(self, personal: bool) -> Tuple[float, float]:
        """Tests the *best* recorded model state."""
        return self.evaluate(self.data.test_loader, personal, use_best=True)

    def train_and_validate(self, personal: bool) -> Dict:
        """Called by Server's run_clients for a standard training round."""
        train_loss, final_state_dict = self.train(personal=personal)
        val_loss, val_score = self.validate(personal=personal)
        return {
            'train_loss': train_loss, 
            'val_loss': val_loss, 
            'val_score': val_score,
            'state_dict': final_state_dict  # CPU state dict of trained model
        }

# =============================================================================
# == Algorithm-Specific Client Implementations ==
# =============================================================================

class FedProxClient(Client):
    """FedProx Client: Overrides _train_batch to add proximal term."""
    def __init__(self, *args, **kwargs):
        config = args[0]; config.requires_personal_model = False
        super().__init__(*args, **kwargs, personal_model=False)
        self.reg_param = self.config.algorithm_params.get('reg_param', 0.01) # Mu
        self._initial_global_state_dict_cpu = self.global_state.get_current_model_state_dict()

    def set_model_state(self, state_dict: Dict, test: bool = False):
        """Loads state and stores it as the reference for the prox term."""
        super().set_model_state(state_dict, test)
        self._initial_global_state_dict_cpu = copy.deepcopy(state_dict)

    def _train_batch(self, model: nn.Module, optimizer: optim.Optimizer, criterion: Union[nn.Module, Callable], batch_x: Any, batch_y: Any) -> float:
        """Adds FedProx proximal term to the loss before backward."""
        optimizer.zero_grad()
        outputs = model(batch_x)
        task_loss = criterion(outputs, batch_y, self.class_weights)
        proximal_term = torch.tensor(0.0, device=self.training_manager.compute_device)
        if self._initial_global_state_dict_cpu:
            initial_params_iter = iter(self._initial_global_state_dict_cpu.values())
            for param_current in model.parameters():
                if param_current.requires_grad:
                    param_initial_cpu = next(initial_params_iter) # Assume order matches
                    param_initial_gpu = param_initial_cpu.to(self.training_manager.compute_device)
                    proximal_term += torch.sum(torch.pow(param_current - param_initial_gpu, 2))
                    del param_initial_gpu
        total_loss = task_loss + (self.reg_param / 2.0) * proximal_term
        total_loss.backward()
        optimizer.step()
        return task_loss.item() # Report original task loss


class PFedMeClient(Client):
    """pFedMe Client: Overrides train method for k-step inner loop."""
    def __init__(self, *args, **kwargs):
        config = args[0]; config.requires_personal_model = True
        super().__init__(*args, **kwargs, personal_model=True)
        self.reg_param = self.config.algorithm_params.get('reg_param', 15.0) # Lambda
        self.k_steps = self.config.algorithm_params.get('k_steps', 5)

    def train(self, personal: bool = True) -> Tuple[float, Optional[Dict]]:
        """Custom pFedMe training loop for the personal model."""
        if not personal: return super().train(personal=False) # Delegate global training if needed

        personal_state = self._get_state(True)
        global_state = self._get_state(False)
        if not self.data.train_loader: return float('inf'), personal_state.get_current_model_state_dict()

        model = personal_state.model.to(self.training_manager.compute_device)
        global_model_cpu = global_state.model
        model.train()
        criterion = personal_state.criterion
        optimizer = personal_state.optimizer
        epoch_task_loss = 0.0; num_batches_processed = 0

        with gpu_scope():
            # Removed outer try/finally, rely on gpu_scope
            for _ in range(self.config.epochs):
                for batch in self.data.train_loader:
                    prepared_batch = self.training_manager.prepare_batch(batch, criterion)
                    if prepared_batch is None: continue
                    batch_x_dev, batch_y_dev, _ = prepared_batch

                    # K-step optimization
                    temp_model_state = copy.deepcopy(model.state_dict())
                    for _ in range(self.k_steps):
                         outputs = model(batch_x_dev)
                         loss = criterion(outputs, batch_y_dev, self.class_weights)
                         optimizer.zero_grad(); loss.backward(); optimizer.step()

                    # Proximal update step
                    with torch.no_grad():
                         for param_personal, param_global_cpu in zip(model.parameters(), global_model_cpu.parameters()):
                             param_global_gpu = param_global_cpu.detach().to(self.training_manager.compute_device)
                             update_step = self.config.learning_rate * self.reg_param * (param_personal - param_global_gpu)
                             param_personal.sub_(update_step)
                             del param_global_gpu

                    # Track task loss after update
                    with torch.no_grad():
                        outputs = model(batch_x_dev)
                        task_loss_post_update = criterion(outputs, batch_y_dev, self.class_weights)
                        epoch_task_loss += task_loss_post_update.item()
                    num_batches_processed += 1
                    del batch_x_dev, batch_y_dev, outputs, task_loss_post_update
            model.to(self.training_manager.cpu_device) # Move back to CPU after all epochs/batches

        avg_loss = epoch_task_loss / num_batches_processed if num_batches_processed > 0 else float('inf')
        return avg_loss, personal_state.get_current_model_state_dict()


class DittoClient(Client):
    """Ditto Client: Overrides _train_batch to modify gradients."""
    def __init__(self, *args, **kwargs):
        config = args[0]; config.requires_personal_model = True
        super().__init__(*args, **kwargs, personal_model=True)
        self.reg_param = self.config.algorithm_params.get('reg_param', 0.75) # Lambda

    def _train_batch(self, model: nn.Module, optimizer: optim.Optimizer, criterion: Union[nn.Module, Callable], batch_x: Any, batch_y: Any) -> float:
        """Ditto training step: Modifies gradients only for the personal model."""
        # Check if this is the personal model being trained
        is_personal_model = (model is self.personal_state.model) if self.personal_state else False

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y, self.class_weights)
        loss.backward() # Calculate standard gradients

        # --- Ditto Logic ---
        if is_personal_model:
            global_state_cpu = self.global_state.model
            with torch.no_grad():
                ref_param_gpu = None
                for param_personal, param_global_cpu in zip(model.parameters(), global_state_cpu.parameters()):
                    if param_personal.grad is not None:
                         ref_param_gpu = param_global_cpu.detach().to(self.training_manager.compute_device)
                         reg_term = self.reg_param * (param_personal.detach() - ref_param_gpu)
                         param_personal.grad.add_(reg_term)
                         del ref_param_gpu; ref_param_gpu = None
        # --- End Ditto Logic ---

        optimizer.step()
        return loss.item()