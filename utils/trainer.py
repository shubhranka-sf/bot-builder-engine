from rasa.model_training import train
import os

def train_rasa_model(model_name):
    # Path to your project directory containing YAML files
    project_dir = "./"  # Replace with your project directory

    # Paths to your configuration files
    config_path = os.path.join(project_dir, "config.yml")
    domain_path = os.path.join(project_dir, "domain.yml")
    nlu_path = os.path.join(project_dir, "data/nlu.yml")
    stories_path = os.path.join(project_dir, "data/stories.yml")
    # rules_path = os.path.join(project_dir, "data/rules.yml")

    # Ensure all files exist
    for path in [config_path, domain_path, nlu_path, stories_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

    # Output directory for the trained model
    output_path = os.path.join(project_dir, "models")

    # List of training data files
    training_files = [nlu_path, stories_path]

    # Train the model using the train function
    training_result = train(
        domain=domain_path,
        config=config_path,
        training_files=training_files,
        output=output_path,
        force_training=False,  # Retrain even if model exists
        fixed_model_name=model_name,  # Use default timestamp-based name
        persist_nlu_training_data=False  # Don’t save NLU training data
        )

        # Check the training result
    if training_result.model:
        return training_result.model
    else:
        print("Training failed. No model was created.")
        raise Exception("Training failed. No model was created.")