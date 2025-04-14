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


def loadData(DATASET, DATA_DIR, data_num, cost):
    try:
        if DATASET in TABULAR:
            # Load tabular data
            column_counts = {'Synthetic': 11, 'Credit': 29, 'Weather': 124, 'Heart': 11}
            

            file_path = f'{DATA_DIR}/data_{data_num}_{cost:.2f}.csv'
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            X = pd.read_csv(file_path, sep=' ', names=[i for i in range(column_counts[DATASET])])
            if DATASET != 'Heart':
                X = X.sample(DEFAULT_PARAMS[DATASET]['sizes_per_client'], replace=(DATASET == 'Credit'))

            y = X.iloc[:, -1]
            X = X.iloc[:, :-1]
            return X.values, y.values

        elif DATASET in CLASS_ADJUST:
            # Load class-adjusted data
            file_path = f'{DATA_DIR}/data_{data_num}_{cost:.2f}.npz'
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            data = np.load(file_path)
            X, y = data['data'], data['labels']
            class_size = 250

            ind = sample_per_class(y, class_size)
            X_sample, y_sample = X[ind], y[ind]

            unique_labels = np.unique(y_sample)
            y_sample_mapped = np.vectorize({label: idx for idx, label in enumerate(unique_labels)}.get)(y_sample)
            return X_sample, y_sample_mapped

        elif DATASET == 'IXITiny':
            # Load IXITiny data
            sites = {
                0.08: [['Guys'], ['HH']], 
                0.28: [['IOP'], ['Guys']], 
                0.30: [['IOP'], ['HH']],
                'all': [['IOP'], ['HH'], ['Guys']]}
            site_names = sites[cost][data_num - 1]
            image_files, label_files = [], []

            image_dir = os.path.join(DATA_DIR, 'flamby/image')
            label_dir = os.path.join(DATA_DIR, 'flamby/label')
            for name in site_names:
                image_files.extend([f'{image_dir}/{file}' for file in os.listdir(image_dir) if name in file])
                label_files.extend([f'{label_dir}/{file}' for file in os.listdir(label_dir) if name in file])

            image_files, label_files = align_image_label_files(image_files, label_files)
            return np.array(image_files), np.array(label_files)

        elif DATASET == 'ISIC':
            # Load ISIC data
            dataset_pairings = {
                0.06: (2, 2), 
                0.15: (2, 0), 
                0.19: (2, 3), 
                0.25: (2, 1), 
                0.3: (1, 3),
                'all': (0,1,2,3)}
            site = dataset_pairings[cost][data_num - 1]

            file_path = f'{DATA_DIR}/site_{site}_files_used.csv'
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            files = pd.read_csv(file_path, nrows=2000)
            image_files = [f'{DATA_DIR}/ISIC_2019_Training_Input_preprocessed/{file}.jpg' for file in files['image']]
            return np.array(image_files), files['label'].values
    except Exception as e:
        raise ValueError(f"Error loading data for dataset '{DATASET}': {e}")
    
class BaseDataset(Dataset):
    """Base class for all datasets"""
    def __init__(self, X, y, is_train, **kwargs):
        self.X = X
        self.y = y
        self.is_train = is_train
 
    def __len__(self):
        return len(self.X)
    
    def get_transform(self):
        """To be implemented by child classes"""
        raise NotImplementedError
    
    def __getitem__(self, idx):
        """To be implemented by child classes"""
        raise NotImplementedError
    
    def get_scalers(self):
        """Return any parameters that need to be shared with val/test datasets"""
        return {}

class BaseTabularDataset(BaseDataset):
    """Base class for tabular datasets"""
    def __init__(self, X, y, is_train, **kwargs):
        super().__init__(X, y, is_train, **kwargs)
        
        if self.is_train:
            self.scaler = StandardScaler().fit(self.X)
        else:
            self.scaler = kwargs.get('scaler', None)
        
        self.X = self.scaler.transform(self.X)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx:idx+1], dtype=torch.float32),
            torch.tensor(self.y[idx:idx+1], dtype=torch.float32)
        )
    
    def get_scalers(self):
        return {'scaler': self.scaler}

class SyntheticDataset(BaseTabularDataset):
    """Dataset handler for synthetic data with categorical outcomes"""
    pass  # Inherits all functionality from BaseTabularDataset

class CreditDataset(BaseTabularDataset):
    """Dataset handler for credit data with categorical outcomes"""
    pass  # Inherits all functionality from BaseTabularDataset

class HeartDataset(BaseTabularDataset):
    """Dataset handler for heart data with categorical outcomes"""
    pass  # Inherits all functionality from BaseTabularDataset

class WeatherDataset(BaseTabularDataset):
    """Dataset handler for weather data with continuous outcomes"""    
    pass

class BaseImageDataset(BaseDataset):
    """Base class for image datasets"""
    def __init__(self, X, y, is_train, **kwargs):

        super().__init__(X, y, is_train,  **kwargs)
        self.transform = self.get_transform()


class EMNISTDataset(BaseImageDataset):
    """EMNIST dataset handler"""
    def get_transform(self):
        base_transform = [
            transforms.ToPILImage(), 
            transforms.Resize((28, 28)),
        ]
        
        base_transform_2 =  [
                transforms.ToTensor(),
                transforms.Normalize((0.1736,), (0.3317,))
            ]
        if self.is_train:
            augmentation = [
                transforms.RandomRotation((-15, 15)),
                transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            ]
            return transforms.Compose(base_transform + augmentation + base_transform_2)
        else:
            # Use only the base transformations for testing
            return transforms.Compose(base_transform + base_transform_2)

    def __getitem__(self, idx):
        image = self.X[idx]
        label = self.y[idx]

        image_tensor = self.transform(image)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return image_tensor, label_tensor
        

class CIFARDataset(BaseImageDataset):
    """CIFAR-100 dataset handler"""
    def get_transform(self):
        
        base_transform = [
            transforms.ToPILImage(), 
            transforms.Resize(32),  
        ]

        base_transform_2 = [transforms.ToTensor(),  
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408], 
                std=[0.2675, 0.2565, 0.2761]    
            )]

        if self.is_train:
            # Add augmentations for training
            augmentation = [
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15)
            ]
            # Combine augmentations with the base transformations
            return transforms.Compose(base_transform + augmentation + base_transform_2)
        else:
            # Use only the base transformations for testing
            return transforms.Compose(base_transform + base_transform_2)

    def __getitem__(self, idx):
        image = self.X[idx]
        label = self.y[idx]
        image = image.transpose(1, 2, 0)

        image_tensor = self.transform(image)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return image_tensor, label_tensor
        
class ISICDataset(BaseImageDataset):
    """ISIC dataset handler for skin lesion images"""
    def __init__(self, image_paths, labels, is_train):
        self.sz = 200  # Image size
        super().__init__(image_paths, labels, is_train)
    
    def get_transform(self):
        if self.is_train:
            return albumentations.Compose([
                albumentations.RandomScale(0.07),
                albumentations.Rotate(50),
                albumentations.RandomBrightnessContrast(0.15, 0.1),
                albumentations.Flip(p=0.5),
                albumentations.Affine(shear=0.1),
                albumentations.RandomCrop(self.sz, self.sz),
                albumentations.CoarseDropout(random.randint(1, 8), 16, 16),
                albumentations.Normalize(
                    mean=(0.585, 0.500, 0.486),
                    std=(0.229, 0.224, 0.225),
                    always_apply=True
                ),
            ])
        else:
            return albumentations.Compose([
                albumentations.CenterCrop(self.sz, self.sz),
                albumentations.Normalize(
                    mean=(0.585, 0.500, 0.486),
                    std=(0.229, 0.224, 0.225),
                    always_apply=True
                ),
            ])

    def __getitem__(self, idx):
        image_path = self.X[idx] 
        label = self.y[idx]

        # Read image as numpy array for albumentations
        image = np.array(Image.open(image_path))
        
        # Apply albumentations transforms
        transformed = self.transform(image=image)
        image = transformed['image']
        
        # Convert to tensor
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        label = torch.tensor(label, dtype=torch.int64)
        
        return image, label

class IXITinyDataset(BaseImageDataset):
    """IXITiny dataset handler for 3D medical images"""
    def __init__(self, image_paths, label_paths, is_train):
        self.common_shape = (48, 60, 48)
        super().__init__(image_paths, label_paths, is_train)
        self.label_transform = self._get_label_transform()
    
    def get_transform(self):
        """Use the same transform for both training and validation"""
        default_transform = Compose([
            ToTensor(),
            EnsureChannelFirst(channel_dim="no_channel"),
            Resize(self.common_shape)
        ])
        
        intensity_transform = Compose([
            NormalizeIntensity()
        ])
        
        return lambda x: intensity_transform(default_transform(x))
    
    def _get_label_transform(self):
        """Transform for labels in medical imaging"""
        default_transform = Compose([
            ToTensor(),
            EnsureChannelFirst(channel_dim="no_channel"),
            Resize(self.common_shape)
        ])
        
        one_hot_transform = Compose([
            AsDiscrete(to_onehot=2)
        ])
    
        return lambda x: one_hot_transform(default_transform(x))

    def __getitem__(self, idx):
        image_path = self.X[idx]    
        label_path = self.y[idx]    
    
        # Load 3D medical images
        image = nib.load(image_path).get_fdata()
        label = nib.load(label_path).get_fdata()

        # Apply transforms
        image = self.transform(image)
        label = self.label_transform(label)
        
        return image.to(torch.float32), label


class DataPreprocessor:
    def __init__(self, dataset_name, batch_size):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.dataset_class = self._get_dataset_class()
        
    def _get_dataset_class(self):
        dataset_classes = {
            'Synthetic': SyntheticDataset,
            'Credit': CreditDataset,
            'Weather': WeatherDataset,
            'EMNIST': EMNISTDataset,
            'CIFAR': CIFARDataset,
            'IXITiny': IXITinyDataset,
            'ISIC': ISICDataset,
            'Heart': HeartDataset
        }
        return dataset_classes[self.dataset_name]

    def process_client_data(self, client_data):
        """Process data for multiple clients and create a joint dataset."""
        processed_data = {}
        
        # Process individual client data
        for client_id, data in client_data.items():
            processed_data[client_id] = self._process_single_client(data)
            
        return processed_data

    def _process_single_client(self, data):
        """Process data for a single client."""
        # Split data into train/val/test
        train_data, val_data, test_data = self._split_data(data['X'], data['y'])
        
        # Create train dataset first
        train_dataset = self.dataset_class(train_data[0], train_data[1], is_train=True)
        
        # Get scalers if the dataset uses them
        scaler = train_dataset.get_scalers()
        
        # Create val and test datasets
        val_dataset = self.dataset_class(
            val_data[0], 
            val_data[1], 
            is_train=False, 
            **scaler
        )
        test_dataset = self.dataset_class(
            test_data[0], 
            test_data[1], 
            is_train=False, 
            **scaler
        )
        
        # Create and return dataloaders
        return (
            DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, generator=g),
            DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False),
            DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        )

    def _split_data(self, X, y):
        test_size = 0.2
        val_size = 0.2

        # Split into train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=np.random.RandomState(42)
        )

        # Split train+val into train and val
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, 
            test_size=val_size, 
            random_state=np.random.RandomState(42)
        )

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)