import torch
import os
import json
import warnings

# Suppress TensorFlow warnings and deprecation notices
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_USE_LEGACY_KERAS'] = 'True'
os.environ['TF_ENABLE_DEPRECATION_WARNINGS'] = '0'

# Suppress Python warnings from TensorFlow
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', module='tensorflow')

from tsgm.models.augmentations import MagnitudeWarping, WindowWarping


def augment_dataset(self, augmentation_settings, x_data, y_data, is_source):
        """
        Apply data augmentation on the source domain data
        :param data: source domain data
        :return: augmented source domain data
        """

        base_dir = os.path.dirname(__file__)
        config_path = os.path.join(base_dir, "..", "configs", "data_augmentation_configs.json")
        config_path = os.path.abspath(config_path)

        augmentation_method = None
        config = None
        if augmentation_settings.data_augmentation_configs is not None:
            config = augmentation_settings.data_augmentation_configs
        elif os.path.exists(config_path):
            with open(config_path, "r") as f:
                    config = json.load(f)

        if is_source:
            data_augmentation_options = config["source_data_augmentation"]["data_augmentation_options"]
        else:
            data_augmentation_options = config["target_data_augmentation"]["data_augmentation_options"]
        
        noises = []
        noise_labels = []
        # create new augmented samples
        for option in data_augmentation_options:

            # clone random subset of augmentation_ratio to noise
            ratio = option["add_percentage_of_augmented_samples"]
            if ratio < 1.0:
                num_samples = int(x_data.size(0) * ratio)
                indices = torch.randperm(x_data.size(0))[:num_samples]
                noise = x_data[indices].clone()

                noise_label = y_data[indices].clone()
            else:
                noise = x_data.clone()
                noise_label = y_data.clone()
            
            noise_labels.append(noise_label)

            # apply data augmentation methods
            for augmentation_step in option["augmentation_steps"]:
                if augmentation_step["type"] == "GaussianNoise":
                    noise = torch.tensor(noise, dtype=torch.float32)
                    noise = noise + torch.randn_like(noise) * augmentation_step["params"]["std"] + augmentation_step["params"]["mean"]
                
                elif augmentation_step["type"] == "MagnitudeWarping":
                    aug_model = MagnitudeWarping()
                    noise = aug_model.generate(X=noise, n_samples=len(noise), sigma=augmentation_step["params"]["sigma"])

                elif augmentation_step["type"] == "WindowWarping":
                    aug_model = WindowWarping()
                    noise = aug_model.generate(X=noise, n_samples=len(noise), scales=augmentation_step["params"]["scales"], window_ratio=augmentation_step["params"]["window_ratio"])

                else:
                    raise ValueError("Unsupported data augmentation type")
                # Add more augmentation types as needed
            
            noise = torch.tensor(noise, dtype=torch.float32)
            noises.append(noise)
            
        for i in range(0, len(noises)):
            # Concatenate the original data with the augmented data
            x_data = torch.cat((x_data, noises[i]), dim=0)
            y_data = torch.cat((y_data, noise_labels[i]), dim=0)

        return x_data, y_data
    
