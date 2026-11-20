# Text Oracle: Advanced Predictive Text Generation for Dynamic Communication

# Importing Required Libraries
import numpy as np
import pandas as pd
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from flask import Flask, request, jsonify

# Initialize Flask App
app = Flask(__name__)

# Load Pre-trained GPT-2 Model and Tokenizer
MODEL_NAME = "gpt2"  # You can switch to larger models like "gpt2-large" or fine-tuned GPT-3 models
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)

# Check for GPU Support
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

@app.route("/generate", methods=["POST"])
def generate_text():
    try:
        # Get input data
        input_data = request.json
        prompt = input_data.get("prompt", "")
        max_length = int(input_data.get("max_length", 50))

        # Generate text using GPT-2
        inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
        outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2, temperature=0.7)

        # Decode and return the generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return jsonify({"generated_text": generated_text})

    except Exception as e:
        return jsonify({"error": str(e)})

# Example of a simple HTML Frontend
@app.route("/", methods=["GET"])
def home():
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Text Oracle</title>
    </head>
    <body>
        <h1>Text Oracle: Advanced Predictive Text Generation</h1>
        <form action="/generate" method="post" id="generate-form">
            <label for="prompt">Enter a prompt:</label><br>
            <textarea id="prompt" name="prompt" rows="5" cols="50"></textarea><br><br>
            <label for="max_length">Max Length:</label>
            <input type="number" id="max_length" name="max_length" value="50"><br><br>
            <button type="submit">Generate</button>
        </form>
    </body>
    </html>
    '''

# Run Flask Server
if __name__ == "__main__":
    app.run(debug=True)
