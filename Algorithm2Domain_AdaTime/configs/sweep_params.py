sweep_train_hparams = {
        'num_epochs':   {'values': [3, 4, 5, 6]},
        'batch_size':   {'values': [32, 64]},
        'learning_rate':{'values': [1e-2, 5e-3, 1e-3, 5e-4]},
        'disc_lr':      {'values': [1e-2, 5e-3, 1e-3, 5e-4]},
        'weight_decay': {'values': [1e-4, 1e-5, 1e-6]},
        'step_size':    {'values': [5, 10, 30]},
        'gamma':        {'values': [5, 10, 15, 20, 25]},
        'optimizer':    {'values': ['adam']},
}
sweep_alg_hparams = {
        'TARGET_ONLY': {
            'learning_rate':    {'values': [1e-2, 5e-3, 1e-3, 5e-4]},
            'src_cls_loss_wt':  {'distribution': 'uniform', 'min': 1e-1, 'max': 10},
            'domain_loss_wt':   {'distribution': 'uniform', 'min': 1e-2, 'max': 10}
        },
        'NO_ADAPT': {
            'learning_rate':    {'values': [1e-2, 5e-3, 1e-3, 5e-4]},
            'src_cls_loss_wt':  {'distribution': 'uniform', 'min': 1e-1, 'max': 10},
            'domain_loss_wt':   {'distribution': 'uniform', 'min': 1e-2, 'max': 10}
        },
        'DANN': {
            'learning_rate':    {'values': [1e-2, 5e-3, 1e-3, 5e-4]},
            'src_cls_loss_wt':  {'distribution': 'uniform', 'min': 1e-1, 'max': 10},
            'domain_loss_wt':   {'distribution': 'uniform', 'min': 1e-2, 'max': 10},
        },

        'AdvSKM': {
            'learning_rate':    {'values': [1e-2, 5e-3, 1e-3, 5e-4]},
            'src_cls_loss_wt':  {'distribution': 'uniform', 'min': 1e-1, 'max': 10},
            'domain_loss_wt':   {'distribution': 'uniform', 'min': 1e-2, 'max': 10},
        },

        'CoDATS': {
            'learning_rate':    {'values': [1e-2, 5e-3, 1e-3, 5e-4]},
            'src_cls_loss_wt':  {'distribution': 'uniform', 'min': 1e-1, 'max': 10},
            'domain_loss_wt':   {'distribution': 'uniform', 'min': 1e-2, 'max': 10},
        },

        'CDAN': {
            'learning_rate':    {'values': [1e-2, 5e-3, 1e-3, 5e-4]},
            'src_cls_loss_wt':  {'distribution': 'uniform', 'min': 1e-1, 'max': 10},
            'domain_loss_wt':   {'distribution': 'uniform', 'min': 1e-2, 'max': 10},
            'cond_ent_wt':      {'distribution': 'uniform', 'min': 1e-2, 'max': 10},
        },

        'Deep_Coral': {
            'learning_rate':    {'values': [1e-2, 5e-3, 1e-3, 5e-4]},
            'src_cls_loss_wt':  {'distribution': 'uniform', 'min': 1e-1, 'max': 10},
            'coral_wt':         {'distribution': 'uniform', 'min': 1e-1, 'max': 10},
        },

        'DIRT': {
            'learning_rate':    {'values': [1e-2, 5e-3, 1e-3, 5e-4]},
            'src_cls_loss_wt':  {'distribution': 'uniform', 'min': 1e-1, 'max': 10},
            'domain_loss_wt':   {'distribution': 'uniform', 'min': 1e-2, 'max': 10},
            'cond_ent_wt':      {'distribution': 'uniform', 'min': 1e-2, 'max': 10},
            'vat_loss_wt':      {'distribution': 'uniform', 'min': 1e-2, 'max': 10},
        },

        'HoMM': {
            'learning_rate':    {'values': [1e-2, 5e-3, 1e-3, 5e-4]},
            'src_cls_loss_wt':  {'distribution': 'uniform', 'min': 1e-1, 'max': 10},
            'hommd_wt':         {'distribution': 'uniform', 'min': 1e-2, 'max': 10},
        },

        'MMDA': {
            'learning_rate':    {'values': [1e-2, 5e-3, 1e-3, 5e-4]},
            'src_cls_loss_wt':  {'distribution': 'uniform', 'min': 1e-1, 'max': 10},
            'coral_wt':         {'distribution': 'uniform', 'min': 1e-2, 'max': 10},
            'cond_ent_wt':      {'distribution': 'uniform', 'min': 1e-2, 'max': 10},
            'mmd_wt':           {'distribution': 'uniform', 'min': 1e-2, 'max': 10},
        },

        'DSAN': {
            'learning_rate':    {'values': [1e-2, 5e-3, 1e-3, 5e-4]},
            'src_cls_loss_wt':  {'distribution': 'uniform', 'min': 1e-1, 'max': 10},
            'mmd_wt':           {'distribution': 'uniform', 'min': 1e-2, 'max': 10}
        },

        'DDC': {
            'learning_rate':    {'values': [1e-2, 5e-3, 1e-3, 5e-4]},
            'src_cls_loss_wt':  {'distribution': 'uniform', 'min': 1e-1, 'max': 10},
            'mmd_wt':           {'distribution': 'uniform', 'min': 1e-2, 'max': 10},
        },
        
        'SASA': {
            'learning_rate':    {'values': [1e-2, 5e-3, 1e-3, 5e-4]},
            'src_cls_loss_wt':  {'distribution': 'uniform', 'min': 1e-1, 'max': 10},
            'domain_loss_wt':   {'distribution': 'uniform', 'min': 1e-2, 'max': 10},
        },

        'CoTMix': {
            'learning_rate':            {'values': [1e-2, 5e-3, 1e-3, 5e-4]},
            'temporal_shift':           {'values': [5, 10, 15, 20, 30, 50]},
            'src_cls_weight':           {'distribution': 'uniform', 'min': 1e-1, 'max': 1},
            'mix_ratio':                {'distribution': 'uniform', 'min': 0.5, 'max': 0.99},
            'src_supCon_weight':        {'distribution': 'uniform', 'min': 1e-3, 'max': 1},
            'trg_cont_weight':          {'distribution': 'uniform', 'min': 1e-3, 'max': 1},
            'trg_entropy_weight':       {'distribution': 'uniform', 'min': 1e-3, 'max': 1},
        },
}

def get_sweep_train_hparams(ui_hparams=None):
    """
    Get sweep training hyperparameters.
    If ui_hparams is provided (from hyperparameter tuning UI), use those values.
    Otherwise, use the default sweep_train_hparams.
    """
    if ui_hparams is None:
        return sweep_alg_hparams
    
    # Start with default sweep_train_hparams
    merged_hparams = sweep_train_hparams.copy()
    
    # Update with UI values if provided
    for param_name, param_config in ui_hparams.items():
        if param_name in merged_hparams:
            # Ensure the format matches what sweep expects
            if isinstance(param_config, dict) and "values" in param_config:
                merged_hparams[param_name] = param_config
            else:
                # Convert to proper format if needed
                merged_hparams[param_name] = {"values": param_config}
    
    return merged_hparams

def get_combined_sweep_hparams(ui_hparams=None, algorithm=None):
    """
    Get combined sweep hyperparameters including both training and algorithm-specific parameters.
    If ui_hparams is provided (from hyperparameter tuning UI), use those values for training params.
    """
    # Get training hyperparameters (with UI overrides if provided)
    training_hparams = get_sweep_train_hparams(ui_hparams)
    
    # Get algorithm-specific hyperparameters
    alg_hparams = {}
    if algorithm and algorithm in sweep_alg_hparams:
        alg_hparams = sweep_alg_hparams[algorithm].copy()
    
    # Combine both sets of hyperparameters
    # Training parameters (especially from UI) take precedence over algorithm-specific ones
    combined_hparams = {**alg_hparams, **training_hparams}
    
    return combined_hparams