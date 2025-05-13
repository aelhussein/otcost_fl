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
from helper import (gpu_scope, TrainerConfig, SiteData, ModelState, TrainingManager, 
                    MetricsCalculator, calculate_class_weights, get_parameters_for_dataset, get_model_instance)

from losses import ISICLoss, WeightedCELoss, get_dice_loss
from torch.cuda.amp import autocast, GradScaler
# =============================================================================
# == Base Client Class ==
# =============================================================================
class Client:
    """Base FL client manages model states and training execution."""
    def __init__(self,
                config: TrainerConfig,
                data: SiteData,
                initial_global_state: ModelState,
                metric_fn: Callable,
                personal_model: bool = False):
        from helper import get_model_instance  # Import the helper function
        
        self.config = config
        self.data = data
        self.metric_fn = metric_fn
        self.site_id = data.site_id
        self.requires_personal_model = personal_model
        self.training_manager = TrainingManager(config.device)

        # Initialize client-specific criterion before creating model states
        self._initialize_criterion()

        # Initialize ModelStates with fresh model instances
        global_model = get_model_instance(self.config.dataset_name)
        # Load the state dict from the initial model
        global_model.load_state_dict(initial_global_state.model.state_dict())
        
        # Compile the model if on a supported device
        if hasattr(torch, 'compile') and torch.cuda.is_available():
            try:
                global_model = torch.compile(global_model)
            except Exception as e:
                print(f"Warning: Could not compile model for {self.site_id}: {e}")
        
        self.global_state = ModelState(
            model=global_model.cpu(),
        )
        self.global_state.optimizer = self._create_optimizer(self.global_state.model)
        self.scaler = GradScaler()

        self.personal_state: Optional[ModelState] = None
        if self.requires_personal_model:
            # Create another fresh model instance for personal model
            personal_model = get_model_instance(self.config.dataset_name)
            # Load the same initial state dict
            personal_model.load_state_dict(initial_global_state.model.state_dict())
            
            # Compile the personal model if on a supported device
            if hasattr(torch, 'compile') and torch.cuda.is_available():
                try:
                    personal_model = torch.compile(personal_model)
                except Exception as e:
                    print(f"Warning: Could not compile personal model for {self.site_id}: {e}")
            
            self.personal_state = ModelState(
                model=personal_model.cpu(),
            )
            self.personal_state.optimizer = self._create_optimizer(self.personal_state.model)

    def _create_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """Creates optimizer for a given model instance (CPU)."""
        wd = self.config.algorithm_params.get('weight_decay', 1e-4)
        lr = self.config.learning_rate
        
        # Check if we can use fused implementation (requires CUDA and PyTorch 2.0+)
        use_fused = (
            hasattr(optim.AdamW, 'fused') and 
            torch.cuda.is_available() and 
            'cuda' in self.training_manager.compute_device.type
        )
        
        if use_fused:
            try:
                return optim.AdamW(
                    model.parameters(), 
                    lr=lr, 
                    weight_decay=wd, 
                    eps=1e-8,
                    fused=True  # Use fused implementation
                )
            except Exception as e:
                print(f"Warning: Could not use fused optimizer, falling back to standard implementation: {e}")
    
        # Standard implementation as fallback
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, eps=1e-8)
    
    def _initialize_criterion(self):
        """Initialize criterion based on configuration."""

        dataset_params = get_parameters_for_dataset(self.config.dataset_name)
        
        num_classes = dataset_params.get('fixed_classes')
        criterion_type = dataset_params.get('criterion_type', 'CrossEntropyLoss')
        use_weighted_loss_flag = dataset_params.get('use_weighted_loss', False)
        
        if criterion_type == 'CrossEntropyLoss':
            if use_weighted_loss_flag and hasattr(self.data.train_loader, 'dataset'):
                if num_classes is None:
                    raise ValueError(f"'fixed_classes' must be defined in config for {self.config.dataset_name} when using weighted loss")
                # Calculate class weights
                class_weights = calculate_class_weights(self.data.train_loader.dataset, num_classes)
                # Pass weights directly to constructor
                self.criterion = WeightedCELoss(weights=class_weights)
            else:
                self.criterion = nn.CrossEntropyLoss()
        
        elif criterion_type == 'ISICLoss':
            if num_classes is None:
                raise ValueError(f"'fixed_classes' must be defined in config for {self.config.dataset_name} when using ISICLoss")
            
            if hasattr(self.data.train_loader, 'dataset'):
                client_alpha = calculate_class_weights(self.data.train_loader.dataset, num_classes)
                self.criterion = ISICLoss(alpha=client_alpha)
            else:
                # Fallback if no dataset available (shouldn't happen)
                default_alpha = torch.tensor([1.0] * num_classes)
                self.criterion = ISICLoss(alpha=default_alpha)
    
        elif criterion_type == 'DiceLoss':
            self.criterion = get_dice_loss
        
        else:
            self.criterion = nn.CrossEntropyLoss()

    def _get_state(self, personal: bool) -> ModelState:
        """Helper to get the correct state object."""
        state = self.personal_state if personal and self.requires_personal_model else self.global_state
        if state is None: raise RuntimeError(f"State {'personal' if personal else 'global'} not available.")
        return state
    
    
    def set_model_state(self, state_dict: Dict, test: bool = False):
        """Loads state dict into the client's global model state (CPU)."""
        self.global_state.load_current_model_state_dict(state_dict)

    def _train_batch(self, model: nn.Module, optimizer: optim.Optimizer, criterion: Union[nn.Module, Callable], batch_x: Any, batch_y: Any) -> float:
        """
        Performs a single training step. Assumes model, batch_x, batch_y on compute_device.
        Base implementation for FedAvg. Subclasses override this for algorithm logic.
        """
        optimizer.zero_grad()
        
        # Use autocast for forward pass
        with autocast():
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
        
        # Scale gradients and update weights
        self.scaler.scale(loss).backward()
        self.scaler.step(optimizer)
        self.scaler.update()
        
        return loss.item()

    def _process_epoch(self, 
                    loader: DataLoader, 
                    model: nn.Module, 
                    is_training: bool, 
                    optimizer: Optional[optim.Optimizer] = None) -> Tuple[float, List, List]:
        """
        Unified function to process one epoch for either training or evaluation.
        Reduced CPU transfers during evaluation.
        """
        if is_training:
            model.train()
            if optimizer is None:
                raise ValueError("Optimizer required for training")
        else:
            model.eval()

        epoch_loss = 0.0
        num_batches = 0
        
        # For evaluation, keep tensors on GPU until the end
        gpu_outputs = []
        gpu_labels = []
        
        # For training, we don't collect outputs/labels, just track loss
        epoch_predictions_cpu, epoch_labels_cpu = [], []
        
        context = torch.enable_grad() if is_training else torch.no_grad()

        with gpu_scope():
            with context:
                for batch in loader:
                    prepared_batch = self.training_manager.prepare_batch(batch, self.criterion)
                    if prepared_batch is None: 
                        continue
                        
                    batch_x_dev, batch_y_dev, batch_y_orig_cpu = prepared_batch
                    
                    if is_training:
                        batch_loss = self._train_batch(model, optimizer, self.criterion, batch_x_dev, batch_y_dev)
                        epoch_loss += batch_loss
                    else:  # Evaluation - keep tensors on GPU
                        outputs = model(batch_x_dev)
                        loss = self.criterion(outputs, batch_y_dev)
                        epoch_loss += loss.item()
                        
                        # Store detached tensors on GPU
                        gpu_outputs.append(outputs.detach())
                        # Use the device tensor directly instead of the CPU one
                        gpu_labels.append(batch_y_dev.detach() if torch.is_tensor(batch_y_dev) else batch_y_orig_cpu)

                    num_batches += 1
                    # Cleanup temporary tensors
                    del batch_x_dev, batch_y_dev

                # Only transfer to CPU once after all batches, when in evaluation mode
                if not is_training and gpu_outputs:
                    # Concatenate on GPU first, then transfer to CPU once
                    if len(gpu_outputs) > 0:
                        try:
                            all_outputs = torch.cat(gpu_outputs, dim=0)
                            epoch_predictions_cpu = [all_outputs.cpu()]  # Single tensor in a list
                            
                            # Handle labels - might be on different devices
                            if all(tensor.device == gpu_outputs[0].device for tensor in gpu_labels if torch.is_tensor(tensor)):
                                all_labels = torch.cat([label for label in gpu_labels if torch.is_tensor(label)], dim=0)
                                epoch_labels_cpu = [all_labels.cpu()]  # Single tensor in a list
                            else:
                                # Fall back to per-tensor CPU transfer if needed
                                epoch_labels_cpu = [label.cpu() if torch.is_tensor(label) else label for label in gpu_labels]
                        except Exception as e:
                            # Fallback to per-tensor CPU transfer
                            print(f"Error during GPU tensor concatenation: {e}. Falling back to per-tensor CPU transfer.")
                            epoch_predictions_cpu = [output.cpu() for output in gpu_outputs]
                            epoch_labels_cpu = [label.cpu() if torch.is_tensor(label) else label for label in gpu_labels]
                    
                    # Clean up GPU tensors
                    del gpu_outputs, gpu_labels

        avg_loss = epoch_loss / num_batches if num_batches > 0 else (0.0 if is_training else float('inf'))
        return avg_loss, epoch_predictions_cpu, epoch_labels_cpu

    # --- Public API ---

    def train_and_validate(self, personal: bool) -> Dict:
        """Efficient implementation that minimizes CPU-GPU transfers."""
        # 1. Get the correct state to operate on
        state = self._get_state(personal)
        
        # 2. Move model to GPU once for the entire operation
        model_on_gpu = state.model.to(self.training_manager.compute_device)
        optimizer = state.optimizer
        criterion = self.criterion
        
        # Track metrics
        final_train_loss = float('inf')
        val_loss, val_score = float('inf'), 0.0
        
        try:
            # 3. Perform all training epochs on the GPU model
            for _ in range(self.config.epochs):
                avg_train_loss_epoch, _, _ = self._process_epoch(
                    loader=self.data.train_loader,
                    model=model_on_gpu,
                    is_training=True,
                    optimizer=optimizer
                )
                final_train_loss = avg_train_loss_epoch
                if final_train_loss == float('inf'):
                    break  # Stop if training fails
            
            # 4. Perform validation on the same, already updated GPU model
            val_loss, val_predictions_cpu, val_labels_cpu = self._process_epoch(
                loader=self.data.val_loader,
                model=model_on_gpu,
                is_training=False
            )
            
            # Calculate validation score
            if val_predictions_cpu and val_labels_cpu:
                try:
                    all_preds = torch.cat(val_predictions_cpu, dim=0)
                    all_labels = torch.cat(val_labels_cpu, dim=0)
                    val_score = self.metric_fn(all_labels, all_preds)
                except Exception as e:
                    print(f"Error calculating validation score: {e}")
            
            # 5. Synchronize state back to CPU model (only once)
            state.model.load_state_dict(model_on_gpu.state_dict())
            state.model.cpu()  # Ensure it's on CPU
            
            # 7. Update best state if validation improved
            if val_loss != float('inf'):
                state.update_best_state(val_loss)
        
        finally:
            # 6. Clean up GPU resources
            model_on_gpu = model_on_gpu.cpu() if hasattr(model_on_gpu, 'cpu') else model_on_gpu
            del model_on_gpu
            with torch.cuda.device(self.training_manager.compute_device):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # 8. Return results with CPU state dict
        return {
            'train_loss': final_train_loss,
            'val_loss': val_loss,
            'val_score': val_score,
            'state_dict': state.get_current_model_state_dict()  # Already on CPU
        }

    def test(self, personal: bool) -> Tuple[float, float]:
        """Optimized testing method that properly handles model transfers."""
        # 1. Get correct state for testing
        state = self._get_state(personal)
        
        # 2. Get best state dictionary (on CPU)
        best_cpu_state_dict = state.get_best_model_state_dict()
        if best_cpu_state_dict is None:
            print(f"Warning: No best state available for testing (client: {self.site_id}, personal: {personal}).")
            return float('inf'), 0.0
        
        test_loss, test_score = float('inf'), 0.0
        
        try:
            # 3. Create temporary model and load state
            temp_eval_model = get_model_instance(self.config.dataset_name)
            
            # Handle the case where best state might be from a compiled model
            try:
                if any(k.startswith('_orig_mod.') for k in best_cpu_state_dict.keys()):
                    # Convert compiled state dict format
                    converted_state_dict = {}
                    for k, v in best_cpu_state_dict.items():
                        if k.startswith('_orig_mod.'):
                            converted_state_dict[k[len('_orig_mod.'):]] = v
                        else:
                            converted_state_dict[k] = v
                    temp_eval_model.load_state_dict(converted_state_dict)
                else:
                    # Regular state dict
                    temp_eval_model.load_state_dict(best_cpu_state_dict)
            except Exception as e:
                print(f"Error loading state dict for testing (client: {self.site_id}): {e}")
                return float('inf'), 0.0
            
            # 4. Move temporary model to GPU once
            model_on_gpu = temp_eval_model.to(self.training_manager.compute_device)
            
            # 5. Perform evaluation
            test_loss, test_predictions_cpu, test_labels_cpu = self._process_epoch(
                loader=self.data.test_loader,
                model=model_on_gpu,
                is_training=False
            )
            
            # Calculate test score
            if test_predictions_cpu and test_labels_cpu:
                try:
                    all_preds = torch.cat(test_predictions_cpu, dim=0)
                    all_labels = torch.cat(test_labels_cpu, dim=0)
                    test_score = self.metric_fn(all_labels, all_preds)
                except Exception as e:
                    print(f"Error calculating test score: {e}")
        
        finally:
            # 6. Clean up GPU resources
            if 'model_on_gpu' in locals():
                model_on_gpu = model_on_gpu.cpu() if hasattr(model_on_gpu, 'cpu') else model_on_gpu
                del model_on_gpu
            if 'temp_eval_model' in locals():
                del temp_eval_model
            with torch.cuda.device(self.training_manager.compute_device):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # 7. Return test metrics
        return test_loss, test_score

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
        # Create GPU parameters list for faster training
        self._initial_global_params_gpu = None

    def set_model_state(self, state_dict: Dict, test: bool = False):
        """Loads state and stores it as the reference for the prox term."""
        super().set_model_state(state_dict, test)
        self._initial_global_state_dict_cpu = copy.deepcopy(state_dict)
        # Clear GPU params cache on state update
        self._initial_global_params_gpu = None
        
    def _ensure_global_params_on_gpu(self):
        """Transfers global model parameters to GPU once per training session."""
        if self._initial_global_params_gpu is None:
            # Transfer parameters to GPU in advance
            self._initial_global_params_gpu = []
            for param_tensor in self._initial_global_state_dict_cpu.values():
                if isinstance(param_tensor, torch.Tensor):
                    self._initial_global_params_gpu.append(
                        param_tensor.to(self.training_manager.compute_device)
                    )

    def _train_batch(self, model: nn.Module, optimizer: optim.Optimizer, criterion: Union[nn.Module, Callable], batch_x: Any, batch_y: Any) -> float:
        """Adds FedProx proximal term to the loss before backward."""
        # Ensure reference parameters are on GPU
        self._ensure_global_params_on_gpu()
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        task_loss = criterion(outputs, batch_y)
        proximal_term = torch.tensor(0.0, device=self.training_manager.compute_device)
        
        # Use pre-transferred GPU parameters instead of moving them each iteration
        if self._initial_global_params_gpu:
            for idx, param_current in enumerate(p for p in model.parameters() if p.requires_grad):
                if idx < len(self._initial_global_params_gpu):
                    param_initial_gpu = self._initial_global_params_gpu[idx]
                    proximal_term += torch.sum(torch.pow(param_current - param_initial_gpu, 2))
        
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
        
        # Transfer global model parameters to GPU once at the beginning
        global_model_params_on_gpu = [
            p.detach().clone().to(self.training_manager.compute_device) 
            for p in global_model_cpu.parameters()
        ]
        
        model.train()
        criterion = self.criterion
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
                         loss = criterion(outputs, batch_y_dev)
                         optimizer.zero_grad(); loss.backward(); optimizer.step()

                    # Proximal update step - use pre-transferred parameters
                    with torch.no_grad():
                         for param_personal, param_global_gpu in zip(model.parameters(), global_model_params_on_gpu):
                             update_step = self.config.learning_rate * self.reg_param * (param_personal - param_global_gpu)
                             param_personal.sub_(update_step)

                    # Track task loss after update
                    with torch.no_grad():
                        outputs = model(batch_x_dev)
                        task_loss_post_update = criterion(outputs, batch_y_dev)
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
        # Cache for global model parameters on GPU
        self._global_params_gpu = None

    def _prepare_global_params_gpu(self):
        """Transfers global model parameters to GPU once."""
        if self._global_params_gpu is None:
            self._global_params_gpu = [
                p.detach().clone().to(self.training_manager.compute_device)
                for p in self.global_state.model.parameters()
            ]
        return self._global_params_gpu

    def _train_batch(self, model: nn.Module, optimizer: optim.Optimizer, criterion: Union[nn.Module, Callable], batch_x: Any, batch_y: Any) -> float:
        """Ditto training step: Modifies gradients only for the personal model."""
        # Check if this is the personal model being trained
        is_personal_model = (model is self.personal_state.model) if self.personal_state else False

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward() # Calculate standard gradients

        # --- Ditto Logic ---
        if is_personal_model:
            # Get global parameters on GPU (prepared once per training session)
            global_params_gpu = self._prepare_global_params_gpu()
            
            with torch.no_grad():
                for idx, param_personal in enumerate(model.parameters()):
                    if param_personal.grad is not None and idx < len(global_params_gpu):
                        # Use pre-transferred global parameters
                        reg_term = self.reg_param * (param_personal.detach() - global_params_gpu[idx])
                        param_personal.grad.add_(reg_term)
        # --- End Ditto Logic ---

        optimizer.step()
        return loss.item()

    def train(self, personal: bool) -> Tuple[float, Optional[Dict]]:
        """Override to reset GPU parameter cache at the start of training."""
        if personal:
            # Clear GPU params cache at the start of training to ensure fresh copy
            self._global_params_gpu = None
            
        return super().train(personal)