import torch.optim as optim


def create_optimizer(optimizer_cfg, model_parameters):
    """
    Create optimizer based on configuration.
    
    Args:
        optimizer_cfg: Optimizer configuration
        model_parameters: Model parameters to optimize
        
    Returns:
        optimizer: Initialized optimizer
    """
    optimizer_type = optimizer_cfg.get('type', 'adam').lower()
    lr = optimizer_cfg.get('lr', 1e-3)
    weight_decay = optimizer_cfg.get('weight_decay', 1e-4)
    
    if optimizer_type == 'adam':
        optimizer = optim.Adam(
            model_parameters, 
            lr=lr, 
            weight_decay=weight_decay,
            betas=optimizer_cfg.get('betas', (0.9, 0.999))
        )
    elif optimizer_type == 'adamw':
        optimizer = optim.AdamW(
            model_parameters,
            lr=lr,
            weight_decay=weight_decay,
            betas=optimizer_cfg.get('betas', (0.9, 0.999))
        )
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(
            model_parameters,
            lr=lr,
            weight_decay=weight_decay,
            momentum=optimizer_cfg.get('momentum', 0.9)
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    return optimizer