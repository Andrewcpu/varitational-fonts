import os
from abc import ABC, abstractmethod


# import torch

class AbstractLossLayer(ABC):
    def __init__(self, model, optimizer_setup=None):
        self.model = model
        self.last_loss = None
        self.has_optimizer = True if optimizer_setup else False
        # Allow subclasses to define their own optimizer setup.
        # The `optimizer_setup` is expected to be a function that returns the optimizer.
        self._own_optimizer = optimizer_setup(model.parameters()) if optimizer_setup else None

    @property
    def performs_own_passthrough(self):
        """
        Indicates whether this layer performs its own passthrough.
        Override in subclasses if they perform their own passthrough.
        """
        return False

    @abstractmethod
    def calculate_loss(self, batch_data, model_output, prepared_input_passthrough_data, shared_output=None):
        """
        This method must be implemented by subclasses.
        It should calculate and return the loss based on the data and target.
        """
        pass

    def passthrough(self, batch_data, prepared_input_passthrough_data, shared_output=None):
        """
        Returns the shared output by default; can be overridden for custom passthrough.
        """
        return shared_output

    def step(self, batch_data, prepared_input_passthrough_data, shared_output=None, global_optimizer=None):
        """
        Orchestrates the passthrough, loss calculation, backpropagation,
        and optimization step, recording the last loss value.
        """
        output = self.passthrough(batch_data, prepared_input_passthrough_data, shared_output=shared_output)
        loss = self.calculate_loss(batch_data, output, prepared_input_passthrough_data, shared_output=shared_output)

        # Determine which optimizer to use
        optimizer = self._own_optimizer if self._own_optimizer else global_optimizer

        # Backpropagation and optimization
        if optimizer and (self.performs_own_passthrough and self._own_optimizer):
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()

        self.last_loss = loss.item()
        return loss

    def implementing_parameters(self):
        """
        Subclasses can implement this method to save additional parameters.
        """
        return {}

    def load_implementing_parameters(self, parameters):
        """
        Subclasses can implement this method to load additional parameters.
        """
        pass

    def save_state(self, folder, name):
        """
        Saves the state of the layer, including the optimizer state and last loss.
        """
        state = {
            'own_optimizer_state_dict': self._own_optimizer.state_dict() if self._own_optimizer else None,
            'last_loss': self.last_loss,
            'implementing_params': self.implementing_parameters(),
        }
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
        torch.save(state, f"{folder}/{name}_state.pth")

    def load_state(self, folder, name):
        filepath = f"{folder}/{name}_state.pth"
        if os.path.exists(filepath):
            state = torch.load(filepath)

            # Load the optimizer state if present and the layer has its own optimizer
            if self._own_optimizer and 'own_optimizer_state_dict' in state:
                self._own_optimizer.load_state_dict(state['own_optimizer_state_dict'])

            self.last_loss = state.get('last_loss', None)
            self.load_implementing_parameters(state.get('implementing_params', {}))
            return self
        else:
            return None  # or raise an exception

    # @classmethod
    # def load_state(cls, folder, name, model, optimizer_setup=None):
    #     """
    #     Load and return an instance of the layer from saved state.
    #     """
    #     filepath = f"{folder}/{name}_state.pth"
    #     if os.path.exists(filepath):
    #         state = torch.load(filepath)
    #         # Instantiate the layer, potentially with its own optimizer setup
    #         instance = cls(model, optimizer_setup=optimizer_setup)
    #
    #         # Load the optimizer state if present
    #         if instance._own_optimizer and 'own_optimizer_state_dict' in state:
    #             instance._own_optimizer.load_state_dict(state['own_optimizer_state_dict'])
    #
    #         instance.last_loss = state.get('last_loss', None)
    #         instance.load_implementing_parameters(state.get('implementing_params', {}))
    #         return instance
    #     else:
    #         return None  # or raise an exception


import torch
from abc import ABC, abstractmethod


class AbstractAdvancedModule(ABC):
    def __init__(self, root_dir):
        self.root_dir = root_dir  # Base directory for saving/loading module components
        self.model = self.setup_model()  # Abstract method to setup the model
        self.loss_layers = {}  # Dictionary to hold all loss layers by name
        self.optimizer = self.setup_optimizer()  # Abstract method to setup the optimizer
        self.status = {}
        self.loss_history = {}

    def generate(self, *args, **kwargs):
        self.model.eval()
        args, kwargs = self.preprocess_input(*args, **kwargs)
        res = self.model(*args, **kwargs)
        self.model.train()
        return res

    def process_output(self, *args, **kwargs):
        return args, kwargs

    def preprocess_input(self, *args, **kwargs):
        return args, kwargs

    def process_model_training_output(self, model_output):
        return model_output

    @abstractmethod
    def setup_model(self):
        """
        Implemented by subclasses to define and return the model.
        """
        pass

    @abstractmethod
    def setup_optimizer(self):
        """
        Implemented by subclasses to create and return the optimizer.
        Typically should use self.model.parameters() for the parameters.
        """
        pass

    def add_loss_layer(self, name, loss_layer):
        """
        Adds a loss layer to the module with a specified name.
        """
        if name in self.loss_layers:
            raise ValueError(f"Loss layer '{name}' already exists.")
        self.loss_layers[name] = loss_layer

    def log_status(self):
        keys = list(self.status.keys())
        values = [self.status[key] for key in keys]
        total_loss = sum(values)
        print("")
        print(f"{self.__class__.__name__}: (Total loss: {total_loss:.4f})")
        max_key_length = max(len(key) for key in keys) + 5
        print(" | ".join(['_' * max_key_length for n in range(0, min(3, len(keys)))]))

        for i in range(0, len(keys), 3):
            loss_values = [f"{val:.4f}" for val in values[i:i + 3]]
            loss_names = keys[i:i + 3]

            print(" | ".join(f"{val:<{max_key_length}}" for val in loss_values))
            print(" | ".join(f"{name:<{max_key_length}}" for name in loss_names))

            if i + 3 < len(keys):
                print(" | ".join(['_' * max_key_length for n in range(0, min(3, len(keys)))]))

    def preprocess_training_input(self, batch_data):
        return batch_data, None

    def step(self, batch_data, expand=True):
        """
        Orchestrates a training step for all loss layers.
        """
        self.model.train()  # Ensure the model is in training mode
        total_loss = 0
        prepared_input, prepared_input_passthrough_data = self.preprocess_training_input(batch_data)
        if expand:
            shared_output = self.model(*prepared_input)  # Perform shared passthrough
        else:
            shared_output = self.model(prepared_input)  # Perform shared passthrough
        self.status = {}
        # Process layers using shared or individual passthroughs
        for name, loss_layer in self.loss_layers.items():
            if loss_layer.has_optimizer:
                continue
            loss = loss_layer.step(prepared_input, prepared_input_passthrough_data, shared_output=shared_output,
                                   global_optimizer=self.optimizer)
            loss_item = loss.item()
            self.status[name] = loss_item
            total_loss = total_loss + loss  # Process layers using shared or individual passthroughs
        self.optimizer.zero_grad()
        total_loss.backward(retain_graph=True)
        self.optimizer.step()
        self.optimizer.zero_grad()
        for name, loss_layer in self.loss_layers.items():
            if not loss_layer.has_optimizer:
                continue
            loss = loss_layer.step(prepared_input, prepared_input_passthrough_data, shared_output=shared_output,
                                   global_optimizer=self.optimizer)
            loss_item = loss.item()
            self.status[name] = loss_item

        for key in self.status.keys():
            if key not in self.loss_history:
                self.loss_history[key] = []
            self.loss_history[key].append(self.status[key])

        # print(status)
        return total_loss, shared_output

    def save_model_and_optimizer(self, folder, model_name, optimizer_name):
        """
        Saves the model and optimizer states.
        """
        if not os.path.exists(folder):
            os.mkdir(folder)
        torch.save(self.model.state_dict(), f"{folder}/{model_name}")
        torch.save(self.optimizer.state_dict(), f"{folder}/{optimizer_name}")

    def load_model_and_optimizer(self, folder, model_name, optimizer_name):
        """
        Loads the model and optimizer states.
        """
        model_state = torch.load(f"{folder}/{model_name}")
        optimizer_state = torch.load(f"{folder}/{optimizer_name}")
        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(optimizer_state)

    def save_state(self):
        """
        Saves the state of the entire module, including all loss layers.
        """
        self.save_model_and_optimizer(self.root_dir, "model_state.pth", "optimizer_state.pth")
        for name, layer in self.loss_layers.items():
            layer.save_state(self.root_dir, name)

    def load_state(self):
        """
        Loads the state of the entire module, including all loss layers.
        """
        self.load_model_and_optimizer(self.root_dir, "model_state.pth", "optimizer_state.pth")
        for name, layer in self.loss_layers.items():
            self.loss_layers[name] = layer.load_state(self.root_dir, name)
