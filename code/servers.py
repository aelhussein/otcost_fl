from configs import *
from clients import *
from helper import *

class Server:
    """Base server class for federated learning."""
    def __init__(self, config: TrainerConfig, globalmodelstate: ModelState):
        self.config = config
        self.device = config.device
        self.personal =config.requires_personal_model
        self.clients = {}
        self.serverstate = globalmodelstate
        self.serverstate.model = self.serverstate.model.to(self.device)
        self.serverstate.best_model = self.serverstate.best_model.to(self.device)
        
 
    def set_server_type(self, name, tuning):
        self.server_type = name
        self.tuning = tuning
    
    def _create_client(self, clientdata, modelstate, personal_model):
        """Create a client instance."""
        return Client(
            config=self.config,  
            data=clientdata, 
            modelstate=modelstate.copy(),
            metrics_calculator=MetricsCalculator(self.config.dataset_name),
            personal_model=personal_model
        )

    def add_client(self, clientdata: SiteData):
        """Add a client to the federation."""        
        client = self._create_client(
            clientdata=clientdata,
            modelstate=self.serverstate,
            personal_model=self.personal
        )
        
        # Add client to federation
        self.clients[clientdata.site_id] = client
        self._update_client_weights()

    def _update_client_weights(self):
        """Update client weights based on dataset sizes."""
        total_samples = sum(client.data.num_samples for client in self.clients.values())
        for client in self.clients.values():
            client.data.weight = client.data.num_samples / total_samples

    def train_round(self):
        """Run one round of training."""
        # Train all clients
        train_loss = 0
        val_loss = 0
        val_score = 0
        for client in self.clients.values():
            # Train and validate
            client_train_loss = client.train(self.personal)
            client_val_loss, client_val_score = client.validate(self.personal)
            
            # Weight metrics by client dataset size
            train_loss += client_train_loss * client.data.weight
            val_loss += client_val_loss * client.data.weight
            val_score +=  client_val_score * client.data.weight
        # Track metrics
        self.serverstate.train_losses.append(train_loss)
        self.serverstate.val_losses.append(val_loss)
        self.serverstate.val_scores.append(val_score)
        # Aggregate and distribute
        self.aggregate_models()
        self.distribute_global_model()

        # Update best model if improved
        if val_loss < self.serverstate.best_loss:
            self.serverstate.best_loss = val_loss
            self.serverstate.best_model = copy.deepcopy(self.serverstate.model)
        return train_loss, val_loss, val_score

    def test_global(self):
        """Test the model across all clients."""
        test_loss = 0
        test_score = 0
        
        for client in self.clients.values():
            client_loss, client_score = client.test(self.personal)
            test_loss += client_loss * client.data.weight
            test_score +=  client_score * client.data.weight

        self.serverstate.test_losses.append(test_loss)
        self.serverstate.test_scores.append(test_score)

        return test_loss, test_score
    
    def aggregate_models(self):
        """Base aggregation method - to be implemented by subclasses."""
        return
    
    def distribute_global_model(self):
        """Base distribution method - to be implemented by subclasses"""
        return
    
class FLServer(Server):
    """Base federated learning server with FedAvg implementation."""
    def aggregate_models(self):
        """Standard FedAvg aggregation."""
        # Reset global model parameters
        for param in self.serverstate.model.parameters():
            param.data.zero_()
            
        # Aggregate parameters
        for client in self.clients.values():
            client_model = client.personal_state.model if self.personal else client.global_state.model
            for g_param, c_param in zip(self.serverstate.model.parameters(), client_model.parameters()):
                g_param.data.add_(c_param.data * client.data.weight)

    def distribute_global_model(self):
        """Distribute global model to all clients."""
        global_state = self.serverstate.model.state_dict()
        for client in self.clients.values():
            client.set_model_state(global_state)

class FedAvgServer(FLServer):
    """FedAvg server implementation."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.diversity_calculator = None

    def _set_diversity_calculation(self):
        client_ids = list(self.clients.keys())
        self.diversity_calculator =  ModelDiversity(self.clients[client_ids[0]], self.clients[client_ids[1]])
        self.diversity_metrics = { 'weight_div': [], 'weight_orient': []}

    def train_round(self):
        """Run one round of training with diversity tracking."""
        # Train all clients
        train_loss = 0
        val_loss = 0
        val_score = 0
        
        for client in self.clients.values():
            # Train and validate
            client_train_loss = client.train(self.personal)
            client_val_loss, client_val_score = client.validate(self.personal)
            
            # Weight metrics by client dataset size
            train_loss += client_train_loss * client.data.weight
            val_loss += client_val_loss * client.data.weight
            val_score += client_val_score * client.data.weight
            
        # Track metrics
        self.serverstate.train_losses.append(train_loss)
        self.serverstate.val_losses.append(val_loss)
        self.serverstate.val_scores.append(val_score)
        
        # Calculate diversity BEFORE aggregation
        if self.diversity_calculator is None:
            self._set_diversity_calculation()
        weight_div, weight_orient = self.diversity_calculator.calculate_weight_divergence()
        self.diversity_metrics['weight_div'].append(weight_div)
        self.diversity_metrics['weight_orient'].append(weight_orient)
        
        # Then aggregate and distribute
        self.aggregate_models()
        self.distribute_global_model()

        # Update best model if improved
        if val_loss < self.serverstate.best_loss:
            self.serverstate.best_loss = val_loss
            self.serverstate.best_model = copy.deepcopy(self.serverstate.model)
            
        return train_loss, val_loss, val_score


class PFedMeServer(FLServer):
    """PFedMe server implementation."""
    def _create_client(self, clientdata, modelstate, personal_model = True):
        """Create a client instance."""
        return PFedMeClient(
            config=self.config,  
            data=clientdata, 
            modelstate=modelstate.copy(),  
            metrics_calculator=MetricsCalculator(self.config.dataset_name),
            personal_model=personal_model
        )

class DittoServer(FLServer):
    """Ditto server implementation."""
    def _create_client(self, clientdata, modelstate, personal_model = True):
        """Create a client instance."""
        return DittoClient(
            config=self.config,  
            data=clientdata, 
            modelstate=modelstate.copy(),  
            metrics_calculator=MetricsCalculator(self.config.dataset_name),
            personal_model=personal_model
        )
    
    def train_round(self):
        """Run one round of training."""
        # First do global model updates (FedAvg style)
        global_train_loss = 0
        global_val_loss = 0
        global_val_score = 0

        # 1. Global model update step (like FedAvg)
        for client in self.clients.values():
            # Train and validate global model
            client_train_loss = client.train(personal=False)  # Force global update
            client_val_loss, client_val_score = client.validate(personal=False)
            
            # Weight metrics by client dataset size
            global_train_loss += client_train_loss * client.data.weight
            global_val_loss += client_val_loss * client.data.weight
            global_val_score += client_val_score*  client.data.weight

        # Aggregate and distribute global model
        self.aggregate_models()
        self.distribute_global_model()

        # 2. Personal model update step (Ditto)
        personal_train_loss = 0
        personal_val_loss = 0
        personal_val_score = 0

        for client in self.clients.values():
            # Train and validate personal model
            client_train_loss = client.train(personal=True)  # Personal update
            client_val_loss, client_val_score = client.validate(personal=True)
            
            # Weight metrics by client dataset size
            personal_train_loss += client_train_loss * client.data.weight
            personal_val_loss += client_val_loss * client.data.weight
            personal_val_score += client_val_score * client.data.weight

        self.serverstate.train_losses.append(personal_train_loss)
        self.serverstate.val_losses.append(personal_val_loss)
        self.serverstate.val_scores.append(personal_val_score)

        # Update best model if improved (using global model performance)
        if global_val_loss < self.serverstate.best_loss:
            self.serverstate.best_loss = global_val_loss
            self.serverstate.best_model = copy.deepcopy(self.serverstate.model)

        return personal_train_loss, personal_val_loss, personal_val_score

class ModelDiversity:
    """Calculates diversity metrics between two clients' models."""
    def __init__(self, client_1: Client, client_2: Client):
        self.client_1 = client_1
        self.client_2 = client_2

    def calculate_weight_divergence(self):
        """Calculate weight divergence metrics between two clients."""
        weights_1 = self._get_weights(self.client_1)
        weights_2 = self._get_weights(self.client_2)
        
        # Normalize weights
        norm_1 = torch.norm(weights_1)
        norm_2 = torch.norm(weights_2)
        w1_normalized = weights_1 / norm_1 if norm_1 != 0 else weights_1
        w2_normalized = weights_2 / norm_2 if norm_2 != 0 else weights_2
        
        # Calculate divergence metrics
        weight_div = torch.norm(w1_normalized - w2_normalized)
        weight_orient = torch.dot(w1_normalized, w2_normalized)
        
        return weight_div.item(), weight_orient.item()

    def _get_weights(self, client: Client):
        """Extract weights from a client's model."""
        weights = []
        state = client.global_state
        for param in state.model.parameters():
            weights.append(param.data.view(-1))  # Flatten weights
        return torch.cat(weights)


