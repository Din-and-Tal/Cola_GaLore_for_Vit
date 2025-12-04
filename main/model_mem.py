import os
import sys
import hydra
from contextlib import redirect_stdout

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from train.Trainer import Trainer


def print_model_layers(model, indent=0, prefix=""):
    """Recursively print all layers of a PyTorch model with full hierarchy."""
    indent_str = "│   " * indent

    for name, module in model.named_children():
        # Get module class name and parameter count
        class_name = module.__class__.__name__
        num_params = sum(p.numel() for p in module.parameters(recurse=False))
        trainable_params = sum(
            p.numel() for p in module.parameters(recurse=False) if p.requires_grad
        )

        # Build the display string
        full_name = f"{prefix}.{name}" if prefix else name
        param_info = f"params: {num_params:,}" if num_params > 0 else "no params"
        trainable_info = (
            f" (trainable: {trainable_params:,})"
            if trainable_params != num_params and num_params > 0
            else ""
        )

        print(f"{indent_str}├─ {name}: {class_name} | {param_info}{trainable_info}")

        # Print module's own parameters (weights, biases, etc.)
        for param_name, param in module.named_parameters(recurse=False):
            print(
                f"{indent_str}│   └─ {param_name}: {list(param.shape)}, requires_grad={param.requires_grad}"
            )

        # Recursively print children
        if len(list(module.children())) > 0:
            print_model_layers(module, indent + 1, full_name)


def print_model_summary(model):
    """Print a complete summary of model architecture and parameters."""
    print("\n" + "=" * 80)
    print(f"Model: {model.__class__.__name__}")
    print("=" * 80)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print("-" * 80)
    print("Layer Hierarchy:")
    print("-" * 80)

    print_model_layers(model)

    print("=" * 80 + "\n")


def main():
    # config_names = ["vit_adamw","vit_glora","vit_glora_layer","cola_adamw","cola_glora","cola_glora_layer"]
    # config_names = ["v2_cola_adamw", "v2_cola_glora", "v2_cola_glora_layer"]
    config_names = ["vit_adamw"]

    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(__file__), "model_layers")
    os.makedirs(output_dir, exist_ok=True)

    for config_name in config_names:
        output_file = os.path.join(output_dir, f"{config_name}.txt")

        with open(output_file, "w", encoding="utf-8") as f:
            with redirect_stdout(f):
                print(f"\n{'='*50}")
                print(f"Running config: {config_name}")
                print(f"{'='*50}")

                with hydra.initialize(version_base=None, config_path="../conf"):
                    current_cfg = hydra.compose(config_name=config_name)
                    current_cfg.USE_WANDB = False
                    trainer = Trainer(current_cfg)
                    data, target = next(iter(trainer.loaders.train))
                    print_model_summary(trainer.model)
                    # benchmark_training_memory(model=trainer.model,lossFunc=trainer.lossFunc,optimizer=trainer.optimizer,input_data=data,target=target)
                    # summary(trainer.model, input_size=data.shape)

        print(f"Saved summary to: {output_file}")


if __name__ == "__main__":
    main()
