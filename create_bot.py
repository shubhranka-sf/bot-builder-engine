from flask import Flask, request, jsonify
import yaml
import subprocess
import os
import json
app = Flask(__name__)

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
        result = subprocess.run(
            ["rasa", "train"],
        )
        return jsonify({
            "message": "Model trained successfully",
            "model_path": "/app/models/full_model.tar.gz",
            "output": result.stdout
        }), 200
    except subprocess.CalledProcessError as e:
        print(e.stderr)
        return jsonify({
            "error": "Training failed",
            "details": e.stderr
        }), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "OK"}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0")