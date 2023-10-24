import json
import os


def load_variables_from_json():
    """Load variables from a JSON file if it exists, otherwise use default values."""
    json_path = 'variables.json'
    if os.path.exists(json_path):
        with open(json_path, 'r') as file:
            data = json.load(file)
        return data
    else:
        # Default values (can be extended as needed)
        return {
            "model_cutoff": 12.5,
            "window_durations": [0.1667, 0.3333, 0.5, 0.6667, 1.333, 1.5, 2.333, 9.333, 12.5, 25, 42.5, 60, 75, 90, 150],
            "ideal_song_length": 210,
            # For long window durations (over the cutoff), we reduce the loudest X% frequency buckets (such that the loudest now has a volume equal to the bucket just below the threshold). Note this is a percent (so e.g. give 15 not .15)
            "long_window_reduce_loud_pct": 15,
            # Number of samples. Freq resolution increases and time resolution decreases over increasing window size.
            "window_size": 1024,  # Base value, updated depending on overlaps
            "time_resolution": None,
            "latent_dim": 3,
            "layer_scaling_threshold": 128,
            "encoder_base_filters_short": 32,
            "encoder_base_filters_long": 64,
            "decoder_base_filters": 64,  # This might should be 128...
            "decoder_max_layers": 5,
            "wandb_config_base": {
                "l": 45,
                "batch_size": 8,
                "epochs": 1,
                "alpha": 0.001,
                "beta": 1,
                "gamma": 0.00,
                "delta": 0.000,

            },
            "wandb_config_tune": {
                "l": 45,
                "batch_size": 8,
                "epochs": 7,
                "alpha": 0.001,
                "beta": 1,
                "gamma": 0.00,
                "delta": 0.000,
            }
        }


def write_variables_to_json(data):
    """Write variables to a JSON file."""
    json_path = 'variables.json'
    with open(json_path, 'w') as file:
        json.dump(data, file, indent=4)
