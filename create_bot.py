from flask import Flask, request, jsonify, make_response
from controllers.train import train_model
from rasa.core.agent import Agent

app = Flask(__name__)
agent = None

# Function to add CORS headers to a response
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'  # Allow all origins
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    response.headers['Access-Control-Max-Age'] = '86400'  # Cache preflight for 24 hours
    return response

# Debug logging for all requests
@app.before_request
def log_request():
    print(f"Request: {request.method} {request.path}")
    print(f"Origin: {request.headers.get('Origin')}")
    print(f"Headers: {request.headers}")

@app.route('/train', methods=['POST', 'OPTIONS'])
def train_model_route():
    if request.method == "OPTIONS":
        print("Handling OPTIONS for /train")
        response = make_response()
        return add_cors_headers(response)
    data = request.get_json()
    print(f"POST data: {data}")
    response = make_response(train_model(data))
    global agent
    agent = None
    return add_cors_headers(response)

@app.route('/health', methods=['GET', 'OPTIONS'])
def health():
    if request.method == "OPTIONS":
        print("Handling OPTIONS for /health")
        response = make_response()
        return add_cors_headers(response)
    response = make_response(jsonify({"status": "OK"}))
    return add_cors_headers(response)

@app.route('/predict', methods=['POST', 'OPTIONS'])
async def predict():
    if request.method == "OPTIONS":
        print("Handling OPTIONS for /predict")
        response = make_response()
        return add_cors_headers(response)
    data = request.get_json()
    if not data or "text" not in data:
        response = make_response(jsonify({"error": "No text provided"}), 400)
        return add_cors_headers(response)
    if not data.get("model_name"):
        response = make_response(jsonify({"error": "No model name provided"}), 400)
        return add_cors_headers(response)
    if not data.get("sender_id"):
        response = make_response(jsonify({"error": "No sender ID provided"}), 400)
        return add_cors_headers(response)
    try:
        model_name = data.get("model_name")
        global agent
        if not agent:
            agent = Agent.load(f"./models/{model_name}")
        response_data = await agent.handle_text(text_message=data["text"], sender_id=data["sender_id"])
        response = make_response(jsonify({"response": response_data}))
        return add_cors_headers(response)
    except Exception as e:
        response = make_response(jsonify({"error": str(e)}), 500)
        return add_cors_headers(response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)