from flask import Flask, request, jsonify
import yaml
import subprocess
import os
import json
from rasa.core.agent import Agent
import asyncio
from rasa.model_training import train
from rasa.shared.importers.importer import TrainingDataImporter
app = Flask(__name__)
agent = None

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

@app.route('/train', methods=['POST'])
def train_model():
    # Get JSON input
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    # Generate NLU YAML
    nlu_data = {"version": "3.1", "nlu": []}
    for intent in data.get("intents", []):
        intent_data = {
            "intent": intent["name"],
            "examples": "|\n" + "\n".join(f"- {ex}" for ex in intent["examples"])
        }
        nlu_data["nlu"].append(intent_data)

    # Generate Domain YAML
    domain_data = {
        "version": "3.1",
        "intents": [intent["name"] for intent in data.get("intents", [])],
        "entities": [],
        "slots": {},
        "responses": {},
        "actions": []
    }
    for intent in data.get("intents", []):
        intent_name = intent["name"]
        # Responses
        response_key = f"utter_{intent_name}"
        domain_data["responses"][response_key] = [{"text": intent["response"]}]
        domain_data["actions"].append(response_key)
        # Entities and slots
        if "entities" in intent:
            for entity in intent["entities"]:
                entity_name = entity["entity"]
                if entity_name not in domain_data["entities"]:
                    domain_data["entities"].append(entity_name)
                    domain_data["slots"][entity_name] = {
                        "type": "text",
                        "influence_conversation": True,
                        "mappings": [
                            {
                                "type": "from_entity",
                                "entity": entity_name
                            }
                        ]
                    }

    # Generate Stories YAML
    stories_data = {"version": "3.1", "stories": []}
    for conv in data.get("conversations", []):
        story = {
            "story": conv["name"],
            "steps": []
        }
        for step in conv["steps"]:
            story["steps"].append({"intent": step["user"]})
            story["steps"].append({"action": f"utter_{step['bot']}"})
        stories_data["stories"].append(story)

    if not os.path.exists("data"):
        os.makedirs("data")
    # Write YAML files
    with open("data/nlu.yml", "w") as f:
        yaml.dump(nlu_data, f, default_flow_style=False)
    with open("domain.yml", "w") as f:
        yaml.dump(domain_data, f, default_flow_style=False)
    with open("data/stories.yml", "w") as f:
        yaml.dump(stories_data, f, default_flow_style=False)

# Run rasa train
    try:
        result = train_rasa_model(data['model_name'])
        return jsonify({
            "message": "Model trained successfully",
            "model_name": result,
            "output": result
        }), 200
    except Exception as e:
        print(e.stderr)
        return jsonify({
            "error": "Training failed",
            "details": e.stderr
        }), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "OK"}), 200

@app.route('/predict', methods=['POST'])
async def predict():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    if not data.get("model_name"):
        return jsonify({"error": "No model name provided"}), 400
    
    try:
        model_name = data.get("model_name")
        global agent
        if not agent:
            agent = Agent.load(f"./models/{model_name}")
        response = await agent.handle_text(text_message=data["text"], sender_id="user1")
        return jsonify({"response": response}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0")