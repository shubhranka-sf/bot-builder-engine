from flask import jsonify
import os
import yaml
from utils.trainer import train_rasa_model

def train_model(data):
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    # Create directories
    if not os.path.exists("data"):
        os.makedirs("data")
    
    # Generate NLU YAML with all features
    nlu_data = {"version": "3.1", "nlu": []}
    
    # Process intents and their examples
    for intent in data.get("intents", []):
        intent_data = {
            "intent": intent["name"],
            "examples": "|\n" + "\n".join(f"- {ex}" for ex in intent["examples"])
        }
        nlu_data["nlu"].append(intent_data)
    
    # Add lookup tables if available
    if "lookup_tables" in data and data["lookup_tables"]:
        lookup_tables = []
        for table in data["lookup_tables"]:
            lookup_tables.append({
                "lookup": table["name"],
                "examples": "|\n" + "\n".join(f"- {item}" for item in table["elements"])
            })
        nlu_data["nlu"].extend(lookup_tables)
    
    # Add regex features if available
    if "regexes" in data and data["regexes"]:
        regexes = []
        for regex in data["regexes"]:
            regexes.append({
                "regex": regex["name"],
                "examples": "|\n" + f"- {regex['pattern']}"
            })
        nlu_data["nlu"].extend(regexes)

    # Generate Domain YAML with all features
    domain_data = {
        "version": "3.1",
        "intents": [],
        "entities": [],
        "slots": {},
        "responses": {},
        "actions": [],
        "forms": {},
        "session_config": {
            "session_expiration_time": 60,
            "carry_over_slots_to_new_session": True
        }
    }
    
    # Process intents for domain
    for intent in data.get("intents", []):
        intent_obj = intent["name"]
        # Check if any examples have entities to add use_entities config
        has_entities = any("[" in example for example in intent["examples"])
        if has_entities:
            intent_obj = {
                intent["name"]: {
                    "use_entities": True
                }
            }
        domain_data["intents"].append(intent_obj)
    
    # Process entities for domain
    entity_names = []
    for entity in data.get("entities", []):
        entity_obj = {"name": entity["name"]}
        if entity.get("roles"):
            entity_obj["roles"] = entity["roles"]
        if entity.get("groups"):
            entity_obj["groups"] = entity["groups"]
        domain_data["entities"].append(entity_obj)
        entity_names.append(entity["name"])
    
    # Process slots for domain
    for slot in data.get("slots", []):
        slot_config = {
            "type": slot["type"],
            "influence_conversation": slot.get("influence_conversation", True),
            "mappings": slot["mappings"]
        }
        
        # Add type-specific configurations
        if slot["type"] == "categorical" and "values" in slot:
            slot_config["values"] = slot["values"]
        
        if slot["type"] in ["float", "int"] and "min_value" in slot:
            slot_config["min_value"] = slot["min_value"]
            
        if slot["type"] in ["float", "int"] and "max_value" in slot:
            slot_config["max_value"] = slot["max_value"]
            
        if "initial_value" in slot and slot["initial_value"] is not None:
            slot_config["initial_value"] = slot["initial_value"]
            
        domain_data["slots"][slot["name"]] = slot_config
    
    # Process forms for domain
    for form in data.get("forms", []):
        domain_data["forms"][form["name"]] = {
            "required_slots": form["required_slots"]
        }
    
    # Process actions for domain
    custom_actions = []
    for action in data.get("actions", []):
        if action["type"] == "text":
            if "variations" in action:
                domain_data["responses"][action["name"]] = [{"text": text} for text in action["variations"]]
            else:
                domain_data["responses"][action["name"]] = [{"text": action["value"]}]
        elif action["type"] == "action":
            custom_actions.append(action["name"])
        elif action["type"] == "button":
            response = {"buttons": []}
            for button in action.get("buttons", []):
                response["buttons"].append({
                    "title": button["title"],
                    "payload": button["payload"]
                })
            if "text" in action:
                response["text"] = action["text"]
            domain_data["responses"][action["name"]] = [response]
        elif action["type"] == "image":
            response = {
                "image": action["image"]
            }
            if "text" in action:
                response["text"] = action["text"]
            domain_data["responses"][action["name"]] = [response]
        elif action["type"] == "custom":
            responses = []
            for response in action.get("responses", []):
                response_obj = {"text": response["text"]}
                if "condition" in response:
                    response_obj["condition"] = response["condition"]
                responses.append(response_obj)
            domain_data["responses"][action["name"]] = responses
    
    # Add all custom actions to domain
    domain_data["actions"].extend(custom_actions)
    
    # Add form actions to domain
    for form in data.get("forms", []):
        if form["name"] not in domain_data["actions"]:
            domain_data["actions"].append(form["name"])
    
    # Add fallback responses if provided
    if "responses" in data:
        if "fallback" in data["responses"]:
            domain_data["responses"]["utter_fallback"] = [{"text": data["responses"]["fallback"]}]
        if "out_of_scope" in data["responses"]:
            domain_data["responses"]["utter_out_of_scope"] = [{"text": data["responses"]["out_of_scope"]}]
    
    # Generate Stories YAML
    stories_data = {"version": "3.1", "stories": []}
    for story in data.get("stories", []):
        story_yaml = {
            "story": story["name"],
            "steps": []
        }
        for step in story["steps"]:
            if step["node"] == "intent":
                intent_step = {"intent": step["name"]}
                if "entities" in step:
                    intent_step["entities"] = []
                    for entity in step["entities"]:
                        intent_step["entities"].append({
                            entity["entity"]: entity["value"]
                        })
                story_yaml["steps"].append(intent_step)
            elif step["node"] == "action":
                story_yaml["steps"].append({"action": step["name"]})
            elif step["node"] == "slot":
                story_yaml["steps"].append({"slot_was_set": [{step["name"]: step["value"]}]})
        stories_data["stories"].append(story_yaml)
    
    # Generate Rules YAML if rules are provided
    if "rules" in data and data["rules"]:
        rules_data = {"version": "3.1", "rules": []}
        for rule in data.get("rules", []):
            rule_yaml = {
                "rule": rule["name"],
                "steps": []
            }
            # Add condition if provided
            if "condition" in rule:
                rule_yaml["condition"] = [rule["condition"]]
                
            for step in rule["steps"]:
                if step["node"] == "intent":
                    rule_yaml["steps"].append({"intent": step["name"]})
                elif step["node"] == "action":
                    rule_yaml["steps"].append({"action": step["name"]})
                elif step["node"] == "slot":
                    rule_yaml["steps"].append({"slot_was_set": [{step["name"]: step["value"]}]})
            rules_data["rules"].append(rule_yaml)
        
        # Write rules YAML file
        with open("data/rules.yml", "w") as f:
            yaml.dump(rules_data, f, default_flow_style=False)

    # Write YAML files
    with open("data/nlu.yml", "w") as f:
        yaml.dump(nlu_data, f, default_flow_style=False)
    with open("domain.yml", "w") as f:
        yaml.dump(domain_data, f, default_flow_style=False)
    with open("data/stories.yml", "w") as f:
        yaml.dump(stories_data, f, default_flow_style=False)

    # Generate custom actions file if there are custom actions
    if any(action["type"] == "action" and "code" in action for action in data.get("actions", [])):
        if not os.path.exists("actions"):
            os.makedirs("actions")
        
        with open("actions/actions.py", "w") as f:
            f.write("from typing import Any, Text, Dict, List\n")
            f.write("from rasa_sdk import Action, Tracker\n")
            f.write("from rasa_sdk.executor import CollectingDispatcher\n")
            f.write("from rasa_sdk.events import SlotSet\n\n")
            
            for action in data.get("actions", []):
                if action["type"] == "action" and "code" in action:
                    f.write(action["code"] + "\n\n")

    # Run rasa train
    try:
        result = train_rasa_model(data.get('model_name', 'default_model'))
        return jsonify({
            "message": "Model trained successfully",
            "model_name": result,
            "output": result
        }), 200
    except Exception as e:
        print(e)
        return jsonify({
            "error": "Training failed",
            "details": str(e)
        }), 500