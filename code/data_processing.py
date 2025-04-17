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
    required_keys = ['data_source', 'partitioning_strategy', 'cost_interpretation', 'dataset_class', 'num_clients']
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(f"Dataset config for '{dataset_name}' missing required keys: {missing_keys}")
    # Add more specific validation logic here as needed

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

def load_ixi_client_paths(dataset_name: str, data_dir: str, client_num: int, cost_key, config: Dict):
    """Loads file paths for one IXI client based on cost key."""
    source_args = config.get('source_args', {})
    root_dir = source_args.get('data_dir', os.path.join(data_dir, dataset_name))

    # Mapping from cost_key to list of site lists (index client_num-1 selects the list for the client)
    sites_map = {
         0.08: [['Guys'], ['HH']], 0.28: [['IOP'], ['Guys']],
         0.30: [['IOP'], ['HH']], 'all': [['IOP'], ['HH'], ['Guys']]
    } # This could also be moved to config['source_args']

    if cost_key not in sites_map:
        raise ValueError(f"Invalid cost key '{cost_key}' for IXITiny. Available: {list(sites_map.keys())}")
    if client_num > len(sites_map[cost_key]):
         raise ValueError(f"Client number {client_num} out of range for cost key '{cost_key}' in IXITiny (max={len(sites_map[cost_key])})")

    site_names_for_client = sites_map[cost_key][client_num - 1]
    print(f"Loading IXI paths for client {client_num}, sites: {site_names_for_client} (cost_key={cost_key})")

    image_files, label_files = [], []
    # Adjust paths to match typical structure if data is under DATA_DIR/IXITiny/
    image_dir = os.path.join(root_dir, 'flamby/image')
    label_dir = os.path.join(root_dir, 'flamby/label')

    if not os.path.isdir(image_dir) or not os.path.isdir(label_dir):
         raise FileNotFoundError(f"IXI data directories not found at {image_dir} or {label_dir}")

    # Use glob to find files matching site names
    for name_part in site_names_for_client:
        image_files.extend(glob.glob(os.path.join(image_dir, f'*{name_part}*.nii.gz')))
        label_files.extend(glob.glob(os.path.join(label_dir, f'*{name_part}*.nii.gz')))

    # Align files based on common ID part (e.g., IXI???HH????-Guys-?)
    # Assuming format like 'IXI002HH1234-Guys-0768_image.nii.gz' -> ID '002HH1234-Guys-0768'
    def get_ixi_id(path):
         base = os.path.basename(path)
         return base.split('_')[0] # Extracts 'IXI...'' part

    labels_dict = {get_ixi_id(path): path for path in label_files}
    images_dict = {get_ixi_id(path): path for path in image_files}
    common_keys = sorted(list(set(labels_dict.keys()) & set(images_dict.keys())))

    if not common_keys:
         print(f"Warning: No matching image/label pairs found for IXI client {client_num}, sites {site_names_for_client}")

    aligned_image_files = [images_dict[key] for key in common_keys]
    aligned_label_files = [labels_dict[key] for key in common_keys]

    return np.array(aligned_image_files), np.array(aligned_label_files)


def load_isic_client_paths(dataset_name: str, data_dir: str, client_num: int, cost_key, config: Dict):
    """Loads image paths and labels for one ISIC client based on cost key."""
    source_args = config.get('source_args', {})
    root_dir = source_args.get('data_dir', os.path.join(data_dir, dataset_name))

    # Mapping from cost_key to tuple of site indices (0-3)
    # Client_num selects which site index from the tuple to load
    site_map = {
         0.06: (2, 2), 0.15: (2, 0), 0.19: (2, 3), 0.25: (2, 1),
         0.3: (1, 3), 'all': (0, 1, 2, 3)
    } # This could also be in config['source_args']

    if cost_key not in site_map:
         raise ValueError(f"Invalid cost key '{cost_key}' for ISIC. Available: {list(site_map.keys())}")
    if client_num > len(site_map[cost_key]):
         raise ValueError(f"Client number {client_num} out of range for cost key '{cost_key}' in ISIC (max={len(site_map[cost_key])})")

    site_index = site_map[cost_key][client_num - 1]
    print(f"Loading ISIC paths for client {client_num}, site index: {site_index} (cost_key={cost_key})")

    # File containing image names and labels for the site
    site_csv_path = os.path.join(root_dir, f'site_{site_index}_files_used.csv')
    if not os.path.exists(site_csv_path):
         raise FileNotFoundError(f"ISIC site file not found: {site_csv_path}")

    # Load the CSV, potentially sampling
    sampling_config = config.get('sampling_config')
    nrows = sampling_config.get('size') if sampling_config and sampling_config.get('type') == 'fixed_total' else None

    files_df = pd.read_csv(site_csv_path, nrows=nrows)
    if nrows:
         print(f"Sampled ISIC client {client_num} data down to {len(files_df)} rows")

    # Construct full image paths
    image_dir = os.path.join(root_dir, 'ISIC_2019_Training_Input_preprocessed') # Assuming this structure
    if not os.path.isdir(image_dir):
        # Try one level up in case root_dir IS the ISIC dir
        image_dir_alt = os.path.join(os.path.dirname(root_dir), 'ISIC_2019_Training_Input_preprocessed')
        if os.path.isdir(image_dir_alt):
             image_dir = image_dir_alt
        else:
             raise FileNotFoundError(f"ISIC preprocessed image directory not found near {root_dir}")


    image_files = [os.path.join(image_dir, f"{file_stem}.jpg") for file_stem in files_df['image']]
    labels = files_df['label'].values

    # Basic check if images exist (optional, can be slow)
    # if image_files and not os.path.exists(image_files[0]):
    #      print(f"Warning: First image file not found: {image_files[0]}")

    return np.array(image_files), labels


# --- Partitioning Strategies ---

def partition_dirichlet_indices(dataset: TorchDataset, num_clients: int, alpha: float, seed: int = 42, **kwargs):
    """
    Partitions dataset indices based on labels using Dirichlet distribution.
    (Identical to the function defined in the previous planning step)

    Returns:
        Dict[int, List[int]]: Dictionary mapping client index (0 to N-1) to list of sample indices.
    """
    np.random.seed(seed)
    try:
        labels = np.array(dataset.targets) # Common attribute
    except AttributeError:
        try:
            labels = np.array([dataset[i][1] for i in range(len(dataset))])
        except Exception as e:
             raise AttributeError(f"Dataset {type(dataset)} needs labels accessible via .targets or iteration.") from e

    n_classes = len(np.unique(labels))
    n_samples = len(dataset)
    if n_samples == 0: return {i: [] for i in range(num_clients)}

    idx_by_class = [np.where(labels == i)[0] for i in range(n_classes)]
    client_indices = {i: [] for i in range(num_clients)}

    for k in range(n_classes):
        class_indices = idx_by_class[k]
        if len(class_indices) == 0: continue
        np.random.shuffle(class_indices)
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = (proportions / proportions.sum()) * len(class_indices) # Distribute count
        proportions = np.cumsum(proportions).astype(int)

        start_idx = 0
        for i in range(num_clients):
             end_idx = proportions[i]
             client_indices[i].extend(class_indices[start_idx:end_idx].tolist())
             start_idx = end_idx


    # Shuffle indices within each client's list
    for i in range(num_clients):
        np.random.shuffle(client_indices[i])

    # Debug: Print data distribution
    # print(f"Dirichlet (alpha={alpha}) distribution per client:")
    # for i in range(num_clients):
    #      client_labels = labels[client_indices[i]]
    #      label_counts = np.unique(client_labels, return_counts=True)
    #      print(f"  Client {i+1} ({len(client_indices[i])} samples): {dict(zip(label_counts[0], label_counts[1]))}")


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

