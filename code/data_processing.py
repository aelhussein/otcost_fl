from configs import *
from helper import *

def sample_per_class(labels, class_size=500):
    df = pd.DataFrame({'labels': labels})
    df_stratified = df.groupby('labels').apply(lambda x: x.sample(class_size, replace=False))
    return df_stratified.index.get_level_values(1)


def get_common_name(full_path):
    return os.path.basename(full_path).split('_')[0]


def align_image_label_files(image_files, label_files):
    labels_dict = {get_common_name(path): path for path in label_files}
    images_dict = {get_common_name(path): path for path in image_files}
    
    common_keys = sorted(set(labels_dict.keys()) & set(images_dict.keys()))
    return [images_dict[key] for key in common_keys], [labels_dict[key] for key in common_keys]

def partition_dirichlet_indices(dataset: TorchDataset,
                                num_clients: int,
                                alpha: float,
                                seed: int = 42,
                                **kwargs) -> Dict[int, List[int]]:
    """
    Partition indices so each client has the same total number of samples,
    but class proportions differ according to Dirichlet(alpha).
    """
    np.random.seed(seed)

    # 1) get labels array
    try:
        labels = np.array(dataset.targets)
    except AttributeError:
        labels = np.array([dataset[i][1] for i in range(len(dataset))])

    n = len(dataset)
    classes = np.unique(labels)
    n_classes = len(classes)

    # 2) build index pools per class
    idx_by_class = {c: np.where(labels == c)[0].tolist() for c in classes}
    for c in classes:
        np.random.shuffle(idx_by_class[c])

    # 3) draw Dirichlet class‐weights for each client
    #    proportions[client, class]
    proportions = np.random.dirichlet([alpha]*n_classes, size=num_clients)

    # 4) decide how many total samples per client
    base_quota = n // num_clients
    quotas = [base_quota + (1 if i < (n % num_clients) else 0)
              for i in range(num_clients)]

    # 5) for each client, assign samples
    client_indices = {i: [] for i in range(num_clients)}
    for client_id in range(num_clients):
        p = proportions[client_id]
        # compute how many from each class
        counts = (p * quotas[client_id]).astype(int)
        # adjust rounding error on the last class
        counts[-1] = quotas[client_id] - counts[:-1].sum()

        for cls_idx, cls in enumerate(classes):
            take = counts[cls_idx]
            pool = idx_by_class[cls]
            if take > len(pool):
                # if you run out, just take what’s left
                take = len(pool)
            client_indices[client_id].extend(pool[:take])
            idx_by_class[cls] = pool[take:]

    # 6) shuffle each client’s list
    for i in client_indices:
        np.random.shuffle(client_indices[i])

    return client_indices

def validate_dataset_config(config: Dict, dataset_name: str):
    """Basic validation for required config keys."""
    required_keys = ['data_source', 'partitioning_strategy', 'cost_interpretation',
                     'dataset_class', 'default_num_clients']
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(f"Dataset config for '{dataset_name}' missing required keys: {missing_keys}")
    # Add specific checks, e.g., for site_mappings if needed by strategy
    if config['partitioning_strategy'] == 'pre_split' and dataset_name in ['ISIC', 'IXITiny']:
         if 'site_mappings' not in config.get('source_args', {}):
             print(f"Warning: 'site_mappings' potentially missing in source_args for pre-split {dataset_name}.")


# --- Data Source Loaders ---

def load_torchvision_dataset(dataset_name: str, data_dir: str, source_args: Dict, transform_config: Dict = None):
    """Loads train and test sets for specified torchvision dataset."""
    transform_config = transform_config or {}
    tv_dataset_name = source_args.get('dataset_name')
    tv_dataset_args = {k: v for k, v in source_args.items() if k not in ['dataset_name', 'data_dir']}
    root_dir = os.path.join(data_dir) # Construct full path

    print(f"Loading torchvision dataset: {tv_dataset_name} from {root_dir} with args {tv_dataset_args}")

    if tv_dataset_name == 'CIFAR10':
        train_transform = transform_config.get('train') # Allow override via config later
        test_transform = transform_config.get('test')
        if not train_transform:
             train_transform=transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
                ])
        if not test_transform:
             test_transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
                ])
        train_dataset = CIFAR10(root=root_dir, train=True, download=True, transform=train_transform)
        test_dataset = CIFAR10(root=root_dir, train=False, download=True, transform=test_transform)
        return train_dataset, test_dataset

    elif tv_dataset_name == 'EMNIST':
        split = tv_dataset_args.get('split')
        if split != 'digits':
            print(f"Warning: Requested EMNIST split '{split}', but only 'digits' is configured for 10 classes.")
            # Force digits for now based on config, could make this configurable
            tv_dataset_args['split'] = 'digits'

        train_transform = transform_config.get('train') or transforms.Compose([
                transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
            ])
        test_transform = transform_config.get('test') or transforms.Compose([
                 transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
             ])
        train_dataset = EMNIST(root=root_dir, train=True, download=True, transform=train_transform, **tv_dataset_args)
        test_dataset = EMNIST(root=root_dir, train=False, download=True, transform=test_transform, **tv_dataset_args)
        return train_dataset, test_dataset
    else:
         raise ValueError(f"Torchvision loader not configured for: {tv_dataset_name}")


def load_pre_split_csv_client(dataset_name: str, data_dir: str, client_num: int, cost_suffix, config: Dict):
    """Loads data for one client from a pre-split CSV."""
    source_args = config.get('source_args', {})
    root_dir = source_args.get('data_dir', os.path.join(data_dir, dataset_name)) # Use specific dir from args or default
    column_count = source_args.get('column_count')
    if column_count is None:
         raise ValueError(f"Config for {dataset_name} needs 'source_args' with 'column_count'.")

    # Format suffix correctly (assuming numeric cost becomes suffix like _1.00)
    file_suffix_str = f"_{float(cost_suffix):.2f}" if isinstance(cost_suffix, (int, float)) else f"_{cost_suffix}"
    file_path = os.path.join(root_dir, f'data_{client_num}{file_suffix_str}.csv')

    print(f"Loading pre-split CSV: {file_path}")
    if not os.path.exists(file_path):
        # Try without formatting if suffix was already string like 'all'
        file_path_alt = os.path.join(root_dir, f'data_{client_num}_{cost_suffix}.csv')
        if not os.path.exists(file_path_alt):
             raise FileNotFoundError(f"CSV file not found: {file_path} or {file_path_alt}")
        else:
             file_path = file_path_alt


    X_df = pd.read_csv(file_path, sep=' ', header=None, names=list(range(column_count)))

    # Apply sampling if configured
    sampling_config = config.get('sampling_config')
    if sampling_config and sampling_config.get('type') == 'fixed_total':
         sample_size = sampling_config.get('size')
         replace = sampling_config.get('replace', False)
         if sample_size and len(X_df) > sample_size:
              print(f"Sampling client {client_num} data down to {sample_size} (replace={replace})")
              X_df = X_df.sample(sample_size, replace=replace, random_state=42) # Use fixed seed for sampling consistency

    y = X_df.iloc[:, -1].values
    X = X_df.iloc[:, :-1].values
    return X, y

def load_ixi_client_paths(dataset_name: str, data_dir: str, client_num: int, cost_key_or_suffix, config: Dict):
    """Loads file paths for one IXI client based on cost key using config."""
    source_args = config.get('source_args', {})
    root_dir = source_args.get('data_dir', os.path.join(data_dir, dataset_name))
    sites_map = source_args.get('site_mappings')
    cost_key = cost_key_or_suffix # Assume the key is passed directly

    if not sites_map:
         raise ValueError(f"Missing 'site_mappings' in 'source_args' for {dataset_name} config.")
    if cost_key not in sites_map:
        raise ValueError(f"Invalid cost key '{cost_key}' for IXITiny site mapping in config. Available: {list(sites_map.keys())}")

    available_clients_for_cost = len(sites_map[cost_key])
    if client_num > available_clients_for_cost:
         raise ValueError(f"Client number {client_num} out of range ({available_clients_for_cost} available) for cost key '{cost_key}' in IXITiny mapping.")

    site_names_for_client = sites_map[cost_key][client_num - 1] # Get list of site names for this client
    print(f"Loading IXI paths for client {client_num}, sites: {site_names_for_client} (cost_key={cost_key})")

    image_files, label_files = [], []
    image_dir = os.path.join(root_dir, 'flamby/image') # Standard flamby structure assumed
    label_dir = os.path.join(root_dir, 'flamby/label')

    if not os.path.isdir(image_dir) or not os.path.isdir(label_dir):
         raise FileNotFoundError(f"IXI data directories not found at {image_dir} or {label_dir}")

    for name_part in site_names_for_client:
        image_files.extend(glob.glob(os.path.join(image_dir, f'*{name_part}*.nii.gz')))
        label_files.extend(glob.glob(os.path.join(label_dir, f'*{name_part}*.nii.gz')))

    def get_ixi_id(path):
         base = os.path.basename(path)
         parts = base.split('_') # Expecting format like IXI..._image.nii.gz or IXI..._label.nii.gz
         return parts[0]

    labels_dict = {get_ixi_id(path): path for path in label_files}
    images_dict = {get_ixi_id(path): path for path in image_files}
    common_keys = sorted(list(set(labels_dict.keys()) & set(images_dict.keys())))

    if not common_keys:
         print(f"Warning: No matching image/label pairs found for IXI client {client_num}, sites {site_names_for_client}")

    aligned_image_files = [images_dict[key] for key in common_keys]
    aligned_label_files = [labels_dict[key] for key in common_keys]

    return np.array(aligned_image_files), np.array(aligned_label_files)


# MODIFIED: Uses site_mappings from config['source_args']
def load_isic_client_paths(dataset_name: str, data_dir: str, client_num: int, cost_key_or_suffix, config: Dict):
    """Loads image paths and labels for one ISIC client based on cost key using config."""
    source_args = config.get('source_args', {})
    root_dir = source_args.get('data_dir', os.path.join(data_dir, dataset_name))
    site_map = source_args.get('site_mappings')
    cost_key = cost_key_or_suffix

    if not site_map:
         raise ValueError(f"Missing 'site_mappings' in 'source_args' for {dataset_name} config.")
    if cost_key not in site_map:
         raise ValueError(f"Invalid cost key '{cost_key}' for ISIC site mapping in config. Available: {list(site_map.keys())}")

    available_clients_for_cost = len(site_map[cost_key])
    if client_num > available_clients_for_cost:
         raise ValueError(f"Client number {client_num} out of range ({available_clients_for_cost} available) for cost key '{cost_key}' in ISIC mapping.")

    site_index = site_map[cost_key][client_num - 1] # Get the site index (0-3)
    print(f"Loading ISIC paths for client {client_num}, site index: {site_index} (cost_key={cost_key})")

    site_csv_path = os.path.join(root_dir, f'site_{site_index}_files_used.csv')
    if not os.path.exists(site_csv_path):
         raise FileNotFoundError(f"ISIC site file not found: {site_csv_path}")

    # Load the CSV, potentially sampling using main config's sampling_config
    sampling_config = config.get('sampling_config')
    nrows = None
    if sampling_config and sampling_config.get('type') == 'fixed_total':
         nrows = sampling_config.get('size')
         print(f"Sampling ISIC client {client_num} (site {site_index}) down to {nrows} rows")

    files_df = pd.read_csv(site_csv_path, nrows=nrows)

    # Construct full image paths
    image_dir = os.path.join(root_dir, 'ISIC_2019_Training_Input_preprocessed')
    if not os.path.isdir(image_dir):
          raise FileNotFoundError(f"ISIC preprocessed image directory not found: {image_dir}")

    image_files = [os.path.join(image_dir, f"{file_stem}.jpg") for file_stem in files_df['image']]
    labels = files_df['label'].values

    return np.array(image_files), labels


# --- Partitioning Strategies ---

# MODIFIED: Incorporates sampling_config for size balancing
def partition_dirichlet_indices(dataset: TorchDataset,
                                num_clients: int,
                                alpha: float,
                                sampling_config: Dict = None, # Accept sampling_config
                                seed: int = 42,
                                **kwargs) -> Dict[int, List[int]]:
    """
    Partitions dataset indices based on labels using Dirichlet distribution for
    label proportions, aiming for a fixed total number of samples per client
    as specified in sampling_config OR balancing based on total data size.
    """
    np.random.seed(seed)

    # 1) Get labels array
    try:
        labels = np.array(dataset.targets)
    except AttributeError:
        try:
             labels = np.array([dataset[i][1] for i in range(len(dataset))])
        except Exception as e:
              raise AttributeError(f"Dataset {type(dataset)} needs labels accessible via .targets or iteration.") from e

    n_original = len(dataset)
    if n_original == 0:
        print("Warning: Input dataset for partitioning is empty.")
        return {i: [] for i in range(num_clients)}

    classes = np.unique(labels)
    n_classes = len(classes)
    all_original_indices = np.arange(n_original)

    # --- Determine target size (quota) per client ---
    target_samples_per_client = -1
    n_to_distribute = n_original

    if sampling_config and \
       sampling_config.get('type') == 'fixed_total' and \
       isinstance(sampling_config.get('size'), int) and \
       sampling_config.get('size') > 0:

        target_samples_per_client = sampling_config['size']
        n_to_distribute = target_samples_per_client * num_clients
        print(f"Using sampling_config: Targeting fixed size = {target_samples_per_client} samples per client.")

        if n_to_distribute > n_original:
            print(f"Warning: Total target samples ({n_to_distribute}) exceeds dataset size ({n_original}). "
                  f"Partitioning will use all {n_original} samples, client sizes will be balanced based on n_original.")
            target_samples_per_client = -1
            n_to_distribute = n_original
        elif n_to_distribute < n_original:
             print(f"Warning: Total target samples ({n_to_distribute}) is less than dataset size ({n_original}). "
                   f"Only {n_to_distribute} samples will be partitioned.")
             # Select a subset of indices to partition
             indices_to_partition = np.random.choice(all_original_indices, size=n_to_distribute, replace=False)
             labels = labels[indices_to_partition]
             all_original_indices = indices_to_partition

    else:
        print("No valid fixed_total sampling config found or type != 'fixed_total'. Balancing based on total available samples.")
        target_samples_per_client = -1
        n_to_distribute = n_original

    # Calculate exact quotas per client based on n_to_distribute
    base_quota = n_to_distribute // num_clients
    quotas = [base_quota + (1 if i < (n_to_distribute % num_clients) else 0)
              for i in range(num_clients)]
    print(f"Final client quotas: {quotas} (Sum: {sum(quotas)})")

    # 2) Build index pools per class from the (potentially subsetted) indices
    idx_by_class = {c: all_original_indices[labels == c].tolist() for c in np.unique(labels)} # Use unique labels from subset
    classes = list(idx_by_class.keys())
    n_classes = len(classes)
    if n_classes == 0:
         print("Warning: No classes found in the data selected for partitioning.")
         return {i: [] for i in range(num_clients)}

    for c in classes:
        np.random.shuffle(idx_by_class[c])

    # 3) Draw Dirichlet class‐weights for each client (only over available classes)
    proportions = np.random.dirichlet([alpha]*n_classes, size=num_clients)

    # 5) Assign samples to each client to meet quotas
    client_indices = {i: [] for i in range(num_clients)}
    available_class_indices = {c: list(idx) for c, idx in idx_by_class.items()} # Copy pools
    globally_assigned = set()

    for client_id in range(num_clients):
        quota = quotas[client_id]
        if quota == 0: continue

        p = proportions[client_id] # Target label proportions for this client over available classes
        target_class_counts = (p * quota).astype(int)
        # Adjust rounding error over available classes
        if n_classes > 0:
             target_class_counts[-1] = quota - target_class_counts[:-1].sum()
        target_class_counts = np.maximum(0, target_class_counts)

        assigned_count_this_client = 0
        for cls_idx, cls in enumerate(classes): # Iterate through available classes
            num_wanted = target_class_counts[cls_idx]
            if num_wanted == 0: continue

            pool = available_class_indices[cls]
            actual_take = min(num_wanted, len(pool))

            if actual_take > 0:
                indices_to_take = []
                taken_count = 0
                while taken_count < actual_take and pool:
                    idx = pool.pop()
                    if idx not in globally_assigned:
                        indices_to_take.append(idx)
                        globally_assigned.add(idx)
                        taken_count += 1
                client_indices[client_id].extend(indices_to_take)
                assigned_count_this_client += len(indices_to_take)

        shortfall = quota - assigned_count_this_client
        if shortfall > 0:
            print(f"Warning: Client {client_id+1} shortfall of {shortfall} samples due to class pool exhaustion.")

    # 6) Shuffle final list for each client
    for i in client_indices:
        np.random.shuffle(client_indices[i])

    # Final Check
    print("Final client sizes after partitioning:")
    total_assigned = 0
    for k in range(num_clients):
        size = len(client_indices[k])
        print(f"  Client {k+1}: {size} samples (Quota: {quotas[k]})")
        total_assigned += size
    print(f"Total samples assigned: {total_assigned}/{n_to_distribute}")

    return client_indices


def partition_iid_indices(dataset: TorchDataset, num_clients: int, seed: int = 42, **kwargs):
    """Partitions dataset indices equally and randomly (IID)."""
    n_samples = len(dataset)
    indices = np.arange(n_samples)
    np.random.seed(seed)
    np.random.shuffle(indices)
    split_indices = np.array_split(indices, num_clients)
    client_indices = {i: split_indices[i].tolist() for i in range(num_clients)}
    return client_indices

def partition_pre_defined(client_num: int, **kwargs):
    """Placeholder for pre-split data. Returns the client num for lookup."""
    # The actual data loading happens per client in the dispatcher
    return client_num


# --- DataPreprocessor Class ---

class DataPreprocessor:
    def __init__(self, dataset_name, batch_size):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.dataset_config = get_parameters_for_dataset(dataset_name) # Get full config
        self.final_dataset_class = self._get_final_dataset_class()
        # Internal dispatch map
        self._processor_map = {
            'subset': self._process_subset,
            'xy_dict': self._process_xy_dict,
            'path_dict': self._process_path_dict,
        }

    def _get_final_dataset_class(self):
         """Gets the actual Dataset class (e.g., CIFARDataset) based on config."""
         class_name = self.dataset_config.get('dataset_class')
         if not class_name:
              raise ValueError(f"'dataset_class' not defined in config for {self.dataset_name}")
         # Try to get class from current module (data_processing.py)
         if hasattr(sys.modules[__name__], class_name):
              return getattr(sys.modules[__name__], class_name)
         else:
              # Add logic here if Dataset classes are defined elsewhere
              raise ImportError(f"Dataset class '{class_name}' not found in data_processing.py")

    def process_client_data(self, client_input_data: Dict, input_type: str):
        """Process data for multiple clients based on input type using internal dispatch."""
        processed_data = {} # Will store {client_id: (train_loader, val_loader, test_loader)}

        processor_func = self._processor_map.get(input_type)
        if processor_func is None:
            raise ValueError(f"Unknown input_type for DataPreprocessor: {input_type}")

        print(f"Preprocessing data using method for input type: '{input_type}'")
        for client_id, data in client_input_data.items():
             processed_data[client_id] = processor_func(data)

        return processed_data

    def _process_subset(self, client_subset: Subset):
        """Process data for a single client when input is a Subset."""
        client_indices = client_subset.indices
        if not client_indices:
             print(f"Warning: Client subset is empty. Returning empty DataLoaders.")
             empty_loader = DataLoader([])
             return (empty_loader, empty_loader, empty_loader)

        # Split indices into train/val/test
        # Using fixed seed here for consistent splits across runs for the same partition
        train_indices, val_indices, test_indices = self._split_indices(client_indices, seed=42)

        original_dataset = client_subset.dataset

        # Create NEW Subset objects pointing to the original dataset but with split indices
        train_subset = Subset(original_dataset, train_indices) if train_indices else None
        val_subset   = Subset(original_dataset, val_indices) if val_indices else None
        test_subset  = Subset(original_dataset, test_indices) if test_indices else None

        # Create DataLoaders
        # We assume the original_dataset (e.g. torchvision CIFAR10 object) applies transforms correctly
        train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True, generator=g, num_workers=N_WORKERS) if train_subset else DataLoader([])
        val_loader = DataLoader(val_subset, batch_size=self.batch_size, shuffle=False, num_workers=N_WORKERS) if val_subset else DataLoader([])
        test_loader = DataLoader(test_subset, batch_size=self.batch_size, shuffle=False, num_workers=N_WORKERS) if test_subset else DataLoader([])

        return train_loader, val_loader, test_loader

    def _process_xy_dict(self, xy_data: Dict):
        """Process data for a single client when input is {'X': array, 'y': array}."""
        X, y = xy_data['X'], xy_data['y']
        if len(X) == 0:
             print(f"Warning: Client xy_dict data is empty. Returning empty DataLoaders.")
             empty_loader = DataLoader([])
             return (empty_loader, empty_loader, empty_loader)

        # Split these arrays
        train_data, val_data, test_data = self._split_data(X, y, seed=42)

        # Create specific final Dataset instances (e.g., SyntheticDataset)
        # These must handle X, y array inputs in __init__
        # Handle potential scaling for tabular data
        scaler_params = {}
        if 'standard_scale' in self.dataset_config.get('needs_preprocessing', []):
             print("Applying StandardScaler")
             scaler = StandardScaler().fit(train_data[0])
             train_data = (scaler.transform(train_data[0]), train_data[1])
             val_data = (scaler.transform(val_data[0]) if len(val_data[0]) > 0 else val_data[0], val_data[1])
             test_data = (scaler.transform(test_data[0]) if len(test_data[0]) > 0 else test_data[0], test_data[1])
             # Pass scaler only if needed by Dataset class, usually not needed after transform
             # scaler_params = {'scaler': scaler} # Pass if Dataset needs it

        # Instantiate the final Dataset class using the split (and potentially scaled) arrays
        train_dataset = self.final_dataset_class(train_data[0], train_data[1], is_train=True, **scaler_params) if len(train_data[0]) > 0 else None
        val_dataset = self.final_dataset_class(val_data[0], val_data[1], is_train=False, **scaler_params) if len(val_data[0]) > 0 else None
        test_dataset = self.final_dataset_class(test_data[0], test_data[1], is_train=False, **scaler_params) if len(test_data[0]) > 0 else None

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, generator=g, num_workers=N_WORKERS) if train_dataset else DataLoader([])
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=N_WORKERS) if val_dataset else DataLoader([])
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=N_WORKERS) if test_dataset else DataLoader([])

        return train_loader, val_loader, test_loader

    def _process_path_dict(self, path_data: Dict):
        """Process data for a single client when input is {'X': paths, 'y': paths/labels}."""
        X_paths, y_data = path_data['X'], path_data['y']
        if len(X_paths) == 0:
             print(f"Warning: Client path_dict data is empty. Returning empty DataLoaders.")
             empty_loader = DataLoader([])
             return (empty_loader, empty_loader, empty_loader)

        # Split paths/labels
        train_data, val_data, test_data = self._split_data(X_paths, y_data, seed=42)

        # Create specific final Dataset instances (e.g., IXITinyDataset)
        # These must handle path inputs in __init__
        train_dataset = self.final_dataset_class(train_data[0], train_data[1], is_train=True) if len(train_data[0]) > 0 else None
        val_dataset = self.final_dataset_class(val_data[0], val_data[1], is_train=False) if len(val_data[0]) > 0 else None
        test_dataset = self.final_dataset_class(test_data[0], test_data[1], is_train=False) if len(test_data[0]) > 0 else None

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, generator=g, num_workers=N_WORKERS) if train_dataset else DataLoader([])
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=N_WORKERS) if val_dataset else DataLoader([])
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=N_WORKERS) if test_dataset else DataLoader([])

        return train_loader, val_loader, test_loader

    # --- Splitting Helper Functions ---
    def _split_data(self, X, y, test_size=0.2, val_size=0.2, seed=42):
        """Splits arrays/lists (like X, y or paths) into train, val, test."""
        num_samples = len(X)
        if num_samples < 3: # Handle very small datasets
             print(f"Warning: Cannot split data with only {num_samples} samples. Using all for train.")
             return (X, y), (X[:0], y[:0]), (X[:0], y[:0]) # Return empty val/test

        indices = np.arange(num_samples)
        try:
             # Stratify if possible (classification task)
             if len(np.unique(y)) > 1 and len(y) == len(X):
                  stratify_param = y
             else:
                  stratify_param = None

             # Split into train+val and test
             idx_temp, idx_test = train_test_split(
                 indices, test_size=test_size, random_state=np.random.RandomState(seed), stratify=stratify_param
             )

             # Adjust val_size relative to the temp set
             relative_val_size = val_size / (1.0 - test_size) if (1.0 - test_size) > 0 else 0.0
             if relative_val_size >= 1.0: # Handle edge case
                  idx_train, idx_val = [], idx_temp
             elif len(idx_temp) < 2 or relative_val_size == 0: # Handle small temp set
                  idx_train, idx_val = idx_temp, []
             else:
                  stratify_temp = y[idx_temp] if stratify_param is not None else None
                  idx_train, idx_val = train_test_split(
                     idx_temp, test_size=relative_val_size, random_state=np.random.RandomState(seed + 1), stratify=stratify_temp # Use different seed for second split
                  )

        except ValueError as e:
             # Fallback to non-stratified if stratification fails (e.g., class with 1 sample)
             print(f"Warning: Stratified split failed ({e}), falling back to non-stratified split.")
             idx_temp, idx_test = train_test_split(indices, test_size=test_size, random_state=np.random.RandomState(seed))
             relative_val_size = val_size / (1.0 - test_size) if (1.0 - test_size) > 0 else 0.0
             if relative_val_size >= 1.0:
                  idx_train, idx_val = [], idx_temp
             elif len(idx_temp) < 2 or relative_val_size == 0:
                  idx_train, idx_val = idx_temp, []
             else:
                 idx_train, idx_val = train_test_split(idx_temp, test_size=relative_val_size, random_state=np.random.RandomState(seed + 1))


        # Return slices based on type of X and y
        if isinstance(X, np.ndarray):
            X_train, y_train = X[idx_train], y[idx_train]
            X_val, y_val = X[idx_val], y[idx_val]
            X_test, y_test = X[idx_test], y[idx_test]
        elif isinstance(X, list): # Handle lists (e.g., paths)
            X_train = [X[i] for i in idx_train]
            y_train = [y[i] for i in idx_train] # Assumes y is also indexable list/array
            X_val = [X[i] for i in idx_val]
            y_val = [y[i] for i in idx_val]
            X_test = [X[i] for i in idx_test]
            y_test = [y[i] for i in idx_test]
        else:
             raise TypeError(f"Unsupported data type for splitting: {type(X)}")

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)


    def _split_indices(self, indices: List[int], test_size=0.2, val_size=0.2, seed=42):
        """Splits a list of indices into train, val, test index lists."""
        num_samples = len(indices)
        if num_samples < 3:
            print(f"Warning: Cannot split indices list with only {num_samples} samples.")
            return indices, [], []

        # Split into train+val and test indices
        # Cannot stratify here as we only have indices
        indices_temp, test_indices = train_test_split(
            indices, test_size=test_size, random_state=np.random.RandomState(seed)
        )

        # Adjust validation size relative to the remaining temp set
        relative_val_size = val_size / (1.0 - test_size) if (1.0 - test_size) > 0 else 0.0
        if relative_val_size >= 1.0:
            train_indices, val_indices = [], indices_temp
        elif len(indices_temp) < 2 or relative_val_size == 0:
             train_indices, val_indices = indices_temp, []
        else:
            train_indices, val_indices = train_test_split(
                indices_temp, test_size=relative_val_size, random_state=np.random.RandomState(seed + 1)
            )

        return train_indices, val_indices, test_indices


# --- Dataset Wrapper Classes (Ensure they handle expected inputs) ---
# These classes wrap the raw data/paths and apply transforms

class BaseDataset(TorchDataset): # Inherit from torch Dataset
    def __init__(self, X, y, is_train, **kwargs):
        self.X = X
        self.y = y
        self.is_train = is_train

    def __len__(self):
         # Handle cases where X might be empty after splits
         if isinstance(self.X, (np.ndarray, list)):
              return len(self.X)
         return 0 # Or raise error

    def get_transform(self):
        """Should be implemented by subclasses to return appropriate transform based on self.is_train."""
        raise NotImplementedError

    def __getitem__(self, idx):
        """Should be implemented by subclasses to load/transform item at idx."""
        raise NotImplementedError

    def get_scalers(self):
        """Only relevant for tabular data needing external scaling info."""
        return {}

# --- Tabular Dataset Classes ---
class BaseTabularDataset(BaseDataset):
    def __init__(self, X, y, is_train, **kwargs):
        super().__init__(X, y, is_train, **kwargs)
        # Scaling is now handled in DataPreprocessor._process_xy_dict
        # self.scaler = kwargs.get('scaler') # No longer needed here
        self.X = torch.tensor(self.X, dtype=torch.float32)
        # Determine target type based on dataset (could use config)
        if self.y.dtype == 'float' or self.y.dtype == 'float64': # Regression
             self.y = torch.tensor(self.y, dtype=torch.float32)
        else: # Classification
             self.y = torch.tensor(self.y, dtype=torch.long)


    def __getitem__(self, idx):
        # Return tensors directly
        return self.X[idx], self.y[idx]

class SyntheticDataset(BaseTabularDataset): pass
class CreditDataset(BaseTabularDataset): pass
class HeartDataset(BaseTabularDataset): pass
class WeatherDataset(BaseTabularDataset): pass # Target dtype already float


# --- Image Dataset Classes ---
class BaseImageDataset(BaseDataset):
    """Base for image datasets handling numpy arrays or paths."""
    def __init__(self, X, y, is_train, **kwargs):
        super().__init__(X, y, is_train, **kwargs)
        self.transform = self.get_transform() # Get transform based on is_train

# Keep CIFAR/EMNIST/ISIC/IXI Dataset classes mostly as they were,
# ensuring their __init__ accepts the correct input types (numpy for CIFAR/EMNIST, paths for ISIC/IXI)
# and __getitem__ correctly loads/transforms the data using self.transform.

class EMNISTDataset(BaseImageDataset):
    """EMNIST Digits dataset handler (expects numpy X, y)"""
    def get_transform(self):
        # Correct normalization for EMNIST Digits (same as MNIST)
        mean, std = (0.1307,), (0.3081,)
        transforms_list = [transforms.ToPILImage()] # Start with PIL for consistency

        if self.X.ndim == 3: # Grayscale (H, W) -> add channel dim if needed by later transforms
              pass # ToPILImage handles H,W input
        elif self.X.ndim == 4 and self.X.shape[3] == 1: # Grayscale (N, H, W, C=1)
              pass # Handled below
        elif self.X.ndim == 4 and self.X.shape[3] == 3: # RGB (N, H, W, C=3) - unexpected?
             pass # Should work

        transforms_list.extend([
            transforms.Resize((28, 28)),
        ])

        if self.is_train:
             transforms_list.extend([
                transforms.RandomRotation((-15, 15)),
                transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                # transforms.RandomPerspective(distortion_scale=0.2, p=0.5), # Can add more augmentations
            ])

        transforms_list.extend([
             transforms.ToTensor(), # Converts to (C, H, W) and scales to [0, 1]
             transforms.Normalize(mean, std)
        ])
        return transforms.Compose(transforms_list)

    def __getitem__(self, idx):
        # X is expected to be a numpy array (H, W) or (H, W, 1) from preprocessor
        image = self.X[idx]
        label = self.y[idx]

        # Apply transforms defined in get_transform
        image_tensor = self.transform(image)

        # Ensure correct label type
        label_tensor = torch.tensor(label, dtype=torch.long)
        return image_tensor, label_tensor


class CIFARDataset(BaseImageDataset):
    """CIFAR-10 dataset handler (expects numpy X, y)"""
    def get_transform(self):
        mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        transforms_list = [transforms.ToPILImage()] # Input is HWC numpy array

        if self.is_train:
            transforms_list.extend([
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                transforms.RandomHorizontalFlip(),
                # transforms.RandomRotation(15) # Optional
            ])

        transforms_list.extend([
            transforms.ToTensor(), # HWC -> CHW, scales to [0,1]
            transforms.Normalize(mean, std)
        ])
        return transforms.Compose(transforms_list)

    def __getitem__(self, idx):
        # X is expected to be HWC numpy array
        image = self.X[idx]
        label = self.y[idx]

        image_tensor = self.transform(image)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return image_tensor, label_tensor


class ISICDataset(BaseImageDataset):
    """ISIC dataset handler (expects image paths X, labels y)"""
    def __init__(self, image_paths, labels, is_train, **kwargs):
        # Uses Albumentations, needs image paths
        self.sz = 200 # Image size - could be moved to config
        super().__init__(image_paths, labels, is_train, **kwargs) # self.X=paths, self.y=labels

    def get_transform(self):
        # Keep existing Albumentations transforms based on self.is_train
        mean, std = (0.585, 0.500, 0.486), (0.229, 0.224, 0.225)
        if self.is_train:
            return albumentations.Compose([
                albumentations.RandomScale(0.07), albumentations.Rotate(50),
                albumentations.RandomBrightnessContrast(0.15, 0.1), albumentations.Flip(p=0.5),
                albumentations.Affine(shear=0.1), albumentations.RandomCrop(self.sz, self.sz),
                albumentations.CoarseDropout(max_holes=random.randint(1, 8), max_height=16, max_width=16), # Updated CoarseDropout args
                albumentations.Normalize(mean=mean, std=std, always_apply=True),
            ])
        else:
            return albumentations.Compose([
                albumentations.CenterCrop(self.sz, self.sz),
                albumentations.Normalize(mean=mean, std=std, always_apply=True),
            ])

    def __getitem__(self, idx):
        image_path = self.X[idx] # self.X contains paths
        label = self.y[idx]     # self.y contains labels

        try:
             # Read image as numpy array HWC for albumentations
             image = np.array(Image.open(image_path).convert('RGB')) # Ensure RGB
        except FileNotFoundError:
             print(f"Error: Image file not found at {image_path}. Returning None.")
             # Returning None might cause issues in DataLoader, consider placeholder or error
             return None, torch.tensor(label, dtype=torch.long) # Or raise error

        # Apply albumentations transforms
        transformed = self.transform(image=image)
        image = transformed['image'] # Result is HWC numpy array

        # Convert to tensor CHW float
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        label_tensor = torch.tensor(label, dtype=torch.long)

        return image_tensor, label_tensor


class IXITinyDataset(BaseImageDataset):
    """IXITiny dataset handler (expects image paths X, label paths y)"""
    def __init__(self, image_paths, label_paths, is_train, **kwargs):
        # Needs nibabel and monai
        try:
             import nibabel as nib
             from monai.transforms import EnsureChannelFirst, AsDiscrete,Compose,NormalizeIntensity,Resize,ToTensor as MonaiToTensor
             self.nib = nib
             self.monai_transforms = {'EnsureChannelFirst': EnsureChannelFirst, 'AsDiscrete': AsDiscrete, 'Compose': Compose, 'NormalizeIntensity': NormalizeIntensity, 'Resize': Resize, 'ToTensor': MonaiToTensor}
        except ImportError as e:
             raise ImportError(f"IXITinyDataset requires 'nibabel' and 'monai'. Install them. Original error: {e}")

        self.common_shape = (48, 60, 48) # Could move to config
        super().__init__(image_paths, label_paths, is_train, **kwargs) # self.X=img_paths, self.y=lbl_paths
        # Transform is applied in __getitem__ for MONAI

    def get_transform(self):
        # MONAI transforms often applied sequentially in __getitem__
        # This method could define parts of it if needed, based on self.is_train
        # Define base transforms here, applied in __getitem__
        Compose = self.monai_transforms['Compose']
        MonaiToTensor = self.monai_transforms['ToTensor']
        EnsureChannelFirst = self.monai_transforms['EnsureChannelFirst']
        Resize = self.monai_transforms['Resize']
        NormalizeIntensity = self.monai_transforms['NormalizeIntensity']

        self.image_transform = Compose([
            MonaiToTensor(), # Adds channel dim, scales to [0, 1]
            EnsureChannelFirst(channel_dim="no_channel"), # Ensure channel is first if ToTensor didn't do it
            Resize(self.common_shape),
            NormalizeIntensity() # Normalize after resize
        ])
        self.label_transform = Compose([
            MonaiToTensor(),
            EnsureChannelFirst(channel_dim="no_channel"),
            Resize(self.common_shape),
            self.monai_transforms['AsDiscrete'](to_onehot=2) # One-hot encode labels (assuming 2 classes)
        ])
        return None # Indicate transforms are handled internally

    def __getitem__(self, idx):
        image_path = self.X[idx]
        label_path = self.y[idx]

        try:
             image = self.nib.load(image_path).get_fdata(dtype=np.float32)
             label = self.nib.load(label_path).get_fdata(dtype=np.uint8) # Use uint8 for labels
        except FileNotFoundError as e:
             print(f"Error: Nifti file not found: {e}. Returning None.")
             # Handle appropriately, maybe return placeholder tensors?
             return None, None # Or raise error

        # Apply MONAI transforms
        image_tensor = self.image_transform(image)
        label_tensor = self.label_transform(label)

        # Ensure correct types after transforms
        return image_tensor.float(), label_tensor.float() # DICE loss often expects float

