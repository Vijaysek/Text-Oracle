# 🔮 Text Oracle: Advanced Predictive Text Generation for Dynamic Communication

## 📌 Project Overview

**Text Oracle** is an AI-powered predictive text generation application developed using Python, Flask, PyTorch, and the GPT-2 language model.

The application generates meaningful text based on a user-provided prompt. It uses a pre-trained GPT-2 model from Hugging Face Transformers to understand the input context and predict the next sequence of words.

The project provides a Flask-based web application that allows users to submit text prompts and generate AI-generated text.

The main objective is to demonstrate how Natural Language Processing (NLP) and Generative AI can be integrated into a web application to build an intelligent text-generation system.

## 🎯 Project Objectives

- Develop an AI-based predictive text generation application.
- Understand how pre-trained GPT-2 models generate text.
- Integrate Hugging Face Transformers with Python.
- Use PyTorch to load and execute deep learning models.
- Develop REST API endpoints using Flask.
- Generate context-aware text based on user prompts.
- Provide a simple web interface for interacting with the application.

## ✨ Key Features

### 1. AI-Based Text Generation
Uses the pre-trained GPT-2 language model to generate text based on the user's input prompt.

### 2. Context-Aware Predictions
GPT-2 uses the provided text context to predict and generate the next sequence of words.

### 3. Adjustable Maximum Length
Users can specify the maximum generated sequence length through the `max_length` parameter.

### 4. Repetition Control
Uses `no_repeat_ngram_size=2` to reduce repeated two-word sequences during generation.

### 5. Temperature-Based Generation
Uses a temperature value of `0.7` to control the randomness of generated text.

### 6. GPU and CPU Support
Automatically detects whether CUDA-compatible GPU hardware is available.

- GPU available: Uses CUDA.
- GPU unavailable: Uses CPU.

### 7. Flask REST API
Provides a `/generate` POST endpoint that accepts prompts and returns generated text in JSON format.

### 8. Web Interface
Provides a basic HTML interface for entering prompts and specifying maximum text length.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Flask | Web application and API development |
| GPT-2 | Pre-trained language generation model |
| Hugging Face Transformers | Loading GPT-2 model and tokenizer |
| PyTorch | Deep learning model execution |
| NumPy | Numerical computing |
| Pandas | Data manipulation library (imported but not currently used) |
| HTML | Basic frontend interface |

## 🧠 Understanding the GPT-2 Model

GPT-2 (Generative Pre-trained Transformer 2) is a transformer-based language model developed by OpenAI.

It generates text by predicting the next token based on previously provided tokens.

### How GPT-2 Works

1. The user enters a text prompt.
2. The tokenizer converts the text into numerical token IDs.
3. The token IDs are passed to the GPT-2 model.
4. GPT-2 processes the input using its transformer architecture.
5. The model predicts the next tokens.
6. The generation process continues until the configured stopping condition is reached.
7. The tokenizer converts the generated token IDs back into readable text.
8. The application returns the generated text.

### Example

**Input Prompt:**
```text
Artificial Intelligence is changing the world because
```

**Possible Generated Output:**
```text
Artificial Intelligence is changing the world because it helps
people automate tasks, analyze information, and develop intelligent
solutions for different industries.
```

*Note: GPT-2 output is probabilistic. Actual generated text may differ between requests.*

## ⚙️ Project Architecture

```text
                 User
                  |
                  |
            Enter Text Prompt
                  |
                  |
             Flask Web App
                  |
                  |
             /generate API
                  |
                  |
             GPT-2 Tokenizer
                  |
                  |
             Token IDs
                  |
                  |
             GPT-2 Model
                  |
                  |
           Text Generation
                  |
                  |
             GPT-2 Output
                  |
                  |
              Tokenizer
                  |
                  |
          Generated Text
                  |
                  |
             JSON Response
```

## 📂 Project Structure

```text
Text-Oracle/
│
├── app.py                  # Main Flask application
│
├── requirements.txt        # Project dependencies
│
└── README.md               # Project documentation
```

The GPT-2 model and tokenizer are downloaded from Hugging Face when the application first loads them. They may be cached locally for subsequent executions.

## 💻 System Requirements

- Python 3.9 or later
- Minimum 4 GB RAM (8 GB or more recommended)
- Internet connection for the initial model download
- Windows, Linux, or macOS
- CUDA-compatible GPU (optional)

**Note:** Running GPT-2 on a CPU is supported, but generation may be slower than on a compatible GPU.

## 🚀 Installation and Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/Vijaysek/NLP-Generative-AI-Projects.git
```

### Step 2: Navigate to the Project Folder

```bash
cd NLP-Generative-AI-Projects
```

If the Text Oracle Python file is stored in a separate folder, navigate to that folder before continuing.

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

### Step 4: Activate the Virtual Environment

For Windows:

```bash
venv\Scripts\activate
```

For Linux/macOS:

```bash
source venv/bin/activate
```

### Step 5: Install Required Libraries

```bash
pip install flask numpy pandas torch transformers
```

### Step 6: Run the Application

If the Python file is named `app.py`:

```bash
python app.py
```

If you use another filename, replace `app.py` with the actual filename.

### Step 7: Open the Application

Open the following address in your browser:

```text
http://127.0.0.1:5000/
```

The Text Oracle application will be available locally.

## 🔗 API Documentation

### Generate Text API

**Endpoint:**
```text
POST /generate
```

**Description:**

Accepts a text prompt and generates text using the GPT-2 language model.

**Request Format:**

The API expects JSON input.

```json
{
    "prompt": "Artificial Intelligence is",
    "max_length": 50
}
```

**Successful Response:**

```json
{
    "generated_text": "Artificial Intelligence is ..."
}
```

**Error Response:**

```json
{
    "error": "Error description"
}
```

### API Parameters

| Parameter | Description | Default Value |
|---|---|---|
| prompt | Text provided by the user | Empty string |
| max_length | Maximum total sequence length, including input tokens | 50 |

**Important:** `max_length` represents the total sequence length, including the original prompt, not just the number of newly generated tokens.

## 🔍 Important Code Components

### 1. Loading the GPT-2 Model

```python
MODEL_NAME = "gpt2"

tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
```

Loads the pre-trained GPT-2 tokenizer and language model.

### 2. GPU and CPU Detection

```python
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model.to(device)
```

Automatically identifies whether a CUDA-compatible GPU is available and places the model on the appropriate device.

### 3. Tokenization

```python
inputs = tokenizer.encode(
    prompt,
    return_tensors="pt"
).to(device)
```

Converts the user's input text into token IDs that GPT-2 can process.

### 4. Text Generation

```python
outputs = model.generate(
    inputs,
    max_length=max_length,
    num_return_sequences=1,
    no_repeat_ngram_size=2,
    temperature=0.7
)
```

Generates text using the input tokens and the configured generation parameters.

### 5. Decoding Generated Text

```python
generated_text = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)
```

Converts generated token IDs into readable text.

### 6. Returning the API Response

```python
return jsonify({
    "generated_text": generated_text
})
```

Returns the generated result in JSON format.

## 🧪 Testing the API

You can test the `/generate` endpoint using Postman.

1. Open Postman.
2. Create a new POST request.
3. Enter the URL:

```text
http://127.0.0.1:5000/generate
```

4. Select **Body → raw → JSON**.
5. Enter:

```json
{
    "prompt": "Machine Learning is useful because",
    "max_length": 50
}
```

6. Click Send.
7. View the generated text in the response.

## 📚 Learning Outcomes

Through this project, the following concepts can be understood:

- Natural Language Processing (NLP).
- Generative Artificial Intelligence.
- Transformer architecture.
- GPT-2 language models.
- Tokenization and token IDs.
- Pre-trained language models.
- Text generation using PyTorch.
- Hugging Face Transformers.
- Flask application development.
- REST API integration.
- CPU and GPU model execution.
- JSON request and response handling.

## 🔮 Future Enhancements

The project can be improved with the following features:

- Develop a modern and responsive frontend.
- Add user authentication.
- Support multiple GPT-based language models.
- Add adjustable temperature and top-k sampling.
- Implement text history and saved prompts.
- Add multilingual text generation.
- Integrate a database to store generated results.
- Add streaming text generation.
- Deploy the application to a cloud platform.
- Add input validation and improved error handling.

## ⚠️ Current Implementation Notes

- The application currently uses the standard pre-trained GPT-2 model.
- It does not perform custom model training or fine-tuning.
- The model requires an initial download from Hugging Face.
- The HTML form currently submits form data, whereas the `/generate` endpoint reads JSON using `request.json`. To make the browser form work, connect it using JavaScript `fetch()` with a JSON request or modify the Flask endpoint to accept form data.
- The application runs with Flask's debug mode enabled. Disable debug mode before production deployment.
- The current implementation is a basic demonstration and should be enhanced with input validation and production security before public deployment.

## 👨‍💻 Author

**Project:** Text Oracle – Advanced Predictive Text Generation for Dynamic Communication

**GitHub:** [Vijaysek](https://github.com/Vijaysek)

## 📄 License

This project is intended for educational and learning purposes. Add an appropriate open-source license if you plan to distribute it publicly.

---

⭐ If you find this project interesting, consider giving the repository a star!