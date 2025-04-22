├── code/
│   ├── configs.py          # Static configurations, paths, hyperparameters
│   ├── data_loading.py     # NEW: Functions to load initial data (base datasets, site data, paths)
│   ├── data_partitioning.py # NEW: Functions for partitioning strategies (Dirichlet, IID)
│   ├── datasets.py         # NEW: All TorchDataset wrapper classes (Base, Tabular, Image specific)
│   ├── data_processing.py  # REFACTORED: DataPreprocessor class, train/val/test splitting
│   ├── pipeline.py         # REFACTORED: Experiment orchestration, server/model creation
│   ├── results_manager.py  # NEW: ResultsManager class
│   ├── servers.py          # (As before) Server logic (FedAvg, Local)
│   ├── models.py           # (As before) Model architectures (Synthetic, Credit, Heart, etc.)
│   ├── helper.py           # (As before) Utilities (seeds, metrics, device handling)
│   ├── losses.py           # (As before, if applicable) Custom loss functions
│   └── run.py              # (As before) Entry point, arg parsing
├── data/
│   ├── Synthetic/
│   ├── Credit/
│   ├── Heart/
│   │   └── processed.site.data ... # Original Heart site files
│   ├── EMNIST/
│   ├── CIFAR10/ # Ensure consistent naming if using torchvision root
│   ├── ISIC/
│   └── IXITiny/
├── results/
│   ├── lr_tuning/
│   ├── evaluation/
│   └── ...
└── saved_models/
    └── ...