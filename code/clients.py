from configs import *
from helper import *

@dataclass
class TrainerConfig:
    """Configuration for training parameters."""
    dataset_name: str
    device: str
    learning_rate: float
    batch_size: int
    epochs: int = 5
    rounds: int = 20
    requires_personal_model: bool = False
    algorithm_params: Optional[float] = None


@dataclass
class SiteData:
    """Holds DataLoader and metadata for a site."""
    site_id: str
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    weight: float = 1.0
    
    def __post_init__(self):
        if self.train_loader is not None:
            self.num_samples = len(self.train_loader.dataset)

@dataclass
class ModelState:
    """Holds state for a single model (global or personalized)."""
    model: nn.Module
    optimizer: torch.optim.Optimizer
    criterion: nn.Module
    best_loss: float = float('inf')
    best_model: Optional[nn.Module] = None
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    val_scores: List[float] = field(default_factory=list)
    test_losses: List[float] = field(default_factory=list)
    test_scores: List[float] = field(default_factory=list)

    def __post_init__(self):
        if self.best_model is None and self.model is not None:
            self.best_model = copy.deepcopy(self.model).to(next(self.model.parameters()).device)
    
    def copy(self):
        """Create a new ModelState with copied model and optimizer."""
        # Create new model instance
        new_model = copy.deepcopy(self.model).to(next(self.model.parameters()).device)
        
        # Setup optimizer
        optimizer_state = copy.deepcopy(self.optimizer.state_dict())
        new_optimizer = type(self.optimizer)(new_model.parameters(), **self.optimizer.defaults)
        new_optimizer.load_state_dict(optimizer_state)
        
        # Create new model state
        return ModelState(
            model=new_model,
            optimizer=new_optimizer,
            criterion= self.criterion 
        )
    
    
class MetricsCalculator:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.continuous_outcome = ['Weather']
        self.squeeze_required = ['Synthetic', 'Credit']
        self.long_required = ['CIFAR', 'EMNIST', 'ISIC']
        self.tensor_metrics = ['IXITiny']
        
    def get_metric_function(self):
        """Returns appropriate metric function based on dataset."""
        metric_mapping = {
            'Synthetic': partial(metrics.f1_score, average='macro'),
            'Credit': partial(metrics.f1_score, average='macro'),
            'Weather': metrics.r2_score,
            'EMNIST': metrics.accuracy_score,
            'CIFAR': metrics.accuracy_score,
            'IXITiny': get_dice_score,
            'ISIC': metrics.balanced_accuracy_score
        }
        return metric_mapping[self.dataset_name]

    def process_predictions(self, predictions, labels):
        """Process model predictions based on dataset requirements."""
        predictions = predictions.cpu().numpy()
        labels = labels.cpu().numpy()
        
        if self.dataset_name in self.continuous_outcome:
            predictions = np.clip(predictions, -2, 2)
        elif self.dataset_name in self.squeeze_required:
            predictions = (predictions >= 0.5).astype(int)
        elif self.dataset_name in self.long_required:
            predictions = predictions.argmax(axis=1)
            
        return predictions, labels

    def calculate_metrics(self, true_labels, predictions):
        """Calculate appropriate metric score."""
        metric_func = self.get_metric_function()
        
        if self.dataset_name in self.tensor_metrics:
            return metric_func(
                torch.tensor(true_labels, dtype=torch.float32),
                torch.tensor(predictions, dtype=torch.float32)
            )
        else:
            return metric_func(
                np.array(true_labels),
                np.array(predictions)
            )        

class Client:
    """Client class that handles both model management and training."""
    def __init__(self, 
                 config: TrainerConfig, 
                 data: SiteData, 
                 modelstate: ModelState,
                 metrics_calculator: MetricsCalculator,
                 personal_model: bool = False):
        self.config = config
        self.data = data
        self.device = config.device
        self.metrics_calculator = metrics_calculator

        # Initialize model states
        self.global_state = modelstate

        # Create personal state if needed
        self.personal_state = self.global_state.copy() if personal_model else None

    def get_client_state(self, personal):
        """Get model state dictionary."""
        state = self.personal_state if personal else self.global_state
        return state

    def set_model_state(self, state_dict):
        """Set model state from dictionary."""
        state = self.get_client_state(personal = False)
        state.model.load_state_dict(state_dict)

    def update_best_model(self, loss, personal):
        """Update best model if loss improves."""
        state = self.get_client_state(personal)
        
        if loss < state.best_loss:
            state.best_loss = loss
            state.best_model = copy.deepcopy(state.model).to(self.device)
            return True
        return False

    def train_epoch(self, personal):
        """Train for one epoch."""
        try:
            state = self.get_client_state(personal)
            model = state.model.train()#.to(self.device)
            total_loss = 0.0
            for batch_x, batch_y in self.data.train_loader:
                batch_x = move_to_device(batch_x, self.device)
                batch_y = move_to_device(batch_y, self.device)
                
                state.optimizer.zero_grad()
                outputs = model(batch_x)
                loss = state.criterion(outputs, batch_y)
                loss.backward()
                
                state.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(self.data.train_loader)
            state.train_losses.append(avg_loss)
            return avg_loss
            
        finally:
            del batch_x, batch_y, outputs, loss
            #model.to('cpu')
            #cleanup_gpu()

    def train(self, personal):
        """Train for multiple epochs."""
        for epoch in range(self.config.epochs):
            final_loss = self.train_epoch(personal)
        return final_loss

    def evaluate(self, loader, personal, validate):
        """Evaluate model performance."""
        try:
            state = self.get_client_state(personal)
            model = (state.model if validate else state.best_model)#.to(self.device)
            model.eval()
            
            total_loss = 0.0
            all_predictions = []
            all_labels = []
            with torch.no_grad():
                for batch_x, batch_y in loader:
                    batch_x = move_to_device(batch_x, self.device)
                    batch_y = move_to_device(batch_y, self.device)
                    outputs = model(batch_x)
                    loss = state.criterion(outputs, batch_y)
                    total_loss += loss.item()
                    
                    predictions, labels = self.metrics_calculator.process_predictions(
                        outputs, batch_y
                    )
                    all_predictions.extend(predictions)
                    all_labels.extend(labels)

            avg_loss = total_loss / len(loader)
            metrics = self.metrics_calculator.calculate_metrics(
                np.array(all_labels),
                np.array(all_predictions)
            )
            return avg_loss, metrics
            
        finally:
            del batch_x, batch_y, outputs, loss
            #model.to('cpu')
            #cleanup_gpu()

    def validate(self, personal):
        """Validate current model."""
        state = self.get_client_state(personal)
        val_loss, val_metrics = self.evaluate(
            self.data.val_loader, 
            personal, 
            validate=True
        )
        
        state.val_losses.append(val_loss)
        state.val_scores.append(val_metrics)
        
        self.update_best_model(val_loss, personal)
        return val_loss, val_metrics

    def test(self, personal):
        """Test using best model."""
        state = self.get_client_state(personal)
        test_loss, test_metrics = self.evaluate(
            self.data.test_loader,
            personal,
            validate = False
        )
        
        state.test_losses.append(test_loss)
        state.test_scores.append(test_metrics)
        
        return test_loss, test_metrics
    


class PFedMeClient(Client):
    """PFedMe client implementation with proximal regularization."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reg_param = self.config.algorithm_params

    def train_epoch(self, personal=True):
        """Train for one epoch with proximal term regularization."""
        try:
            state = self.get_client_state(personal)
            model = state.model.train()#.to(self.device)
            global_model = self.global_state.model.train()#.to(self.device)
            total_loss = 0.0
            for batch_x, batch_y in self.data.train_loader:
                batch_x = move_to_device(batch_x, self.device)
                batch_y = move_to_device(batch_y, self.device)
                
                state.optimizer.zero_grad()
                outputs = model(batch_x)
                loss = state.criterion(outputs, batch_y)
                
                proximal_term = self.compute_proximal_term(
                    model.parameters(),
                    global_model.parameters(),
                )
                
                total_batch_loss = loss + proximal_term
                total_batch_loss.backward()
                
                state.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(self.data.train_loader)
            state.train_losses.append(avg_loss)
            return avg_loss
            
        finally:
            del batch_x, batch_y, outputs, loss
            #model.to('cpu')
            #global_model.to('cpu')
            #cleanup_gpu()

    def train(self, personal=True):
        """Main training loop defaulting to personal model."""
        final_loss = super().train(personal)
        return final_loss
    
    def compute_proximal_term(self, model_params, reference_params):
        """Calculate proximal term between two sets of model parameters."""
        proximal_term = 0.0
        for param, ref_param in zip(model_params, reference_params):
            proximal_term += (self.reg_param / 2) * torch.norm(param - ref_param) ** 2
        return proximal_term
    

class DittoClient(Client):
    """Ditto client implementation."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reg_param = self.config.algorithm_params

    def train_epoch(self, personal):
        if not personal:
            return super().train_epoch(personal=False)
        else:
            try:
                state = self.get_client_state(personal)
                model = state.model.train()#.to(self.device)
                global_model = self.global_state.model.train()#.to(self.device)
                total_loss = 0.0
                for batch_x, batch_y in self.data.train_loader:
                    batch_x = move_to_device(batch_x, self.device)
                    batch_y = move_to_device(batch_y, self.device)
                    
                    state.optimizer.zero_grad()
                    outputs = model(batch_x)
                    loss = state.criterion(outputs, batch_y)
                    loss.backward()
                    
                    self.add_gradient_regularization(
                        model.parameters(),
                        global_model.parameters()
                    )
                        
                    state.optimizer.step()
                    total_loss += loss.item()

                avg_loss = total_loss / len(self.data.train_loader)
                state.train_losses.append(avg_loss)
                return avg_loss
                
            finally:
                del batch_x, batch_y, outputs, loss
                #model.to('cpu')
                #global_model.to('cpu')
                #cleanup_gpu()
        
    def add_gradient_regularization(self, model_params, reference_params):
        """Add regularization directly to gradients."""
        for param, ref_param in zip(model_params, reference_params):
            if param.grad is not None:
                reg_term = self.reg_param * (param - ref_param)
                param.grad.add_(reg_term)
