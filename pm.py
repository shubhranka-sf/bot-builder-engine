files = [
    # "./controllers/train.py", 
# "./utils/trainer.py", 
# "create_bot.py", 
    "config.yml",
    "./data/nlu.yml",
    "./data/stories.yml",
    "./domain.yml", 
    "./data/rules.yml"
]

promptFile = "prompt.txt"

with open(promptFile, "w") as f:
    for file in files:
        with open(file, "r") as f2:
            f.write(f"File: {file}\n\n")
            f.write(f2.read())
            f.write("\n\n")

    f.write("I am making a rasa bot builder. I am taking input as json and building the bot. I will ask you some questions to build the bot. \n\n")