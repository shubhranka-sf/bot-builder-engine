def predict(data):
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