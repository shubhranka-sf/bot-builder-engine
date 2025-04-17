FROM python:3.10-slim

WORKDIR /app

RUN pip install --upgrade pip
RUN pip install 'flask[async]'
RUN pip install rasa
RUN pip install pyyaml

COPY . .

# Change the following line to your actual entrypoint if needed
CMD ["python", "create_bot.py"]