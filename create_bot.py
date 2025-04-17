from controllers.train import train_model
from flask import Flask, request, jsonify
from rasa.core.agent import Agent
app = Flask(__name__)
agent = None

@app.route('/train', methods=['POST'])
def train_model_route():
    # Get JSON input
    data = request.get_json()
    return train_model(data)

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