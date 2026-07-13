"""HIM PPO configuration for the RobotLab-style Go2W task."""


def get_go2w_him_train_cfg() -> dict:
    return {
        "seed": 1,
        "use_occupancy": False,
        "policy": {
            "init_noise_std": 1.0,
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "activation": "elu",
            "use_occupancy": False,
        },
        "algorithm": {
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.005,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": 1.0e-3,
            "schedule": "adaptive",
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
        },
        "runner": {
            "policy_class_name": "HIMActorCritic",
            "algorithm_class_name": "HIMPPO",
            "num_steps_per_env": 24,
            "max_iterations": 20000,
            "save_interval": 200,
            "experiment_name": "robotlab_go2w_him",
            "run_name": "",
            "resume": False,
            "resume_path": None,
            "profile_timing": False,
        },
    }

