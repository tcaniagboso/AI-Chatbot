"""
app.py

Main FastAPI server for GPT-2 inspired text generation.

This module exposes an HTTP API endpoint for interacting with a Transformer-based chatbot.
It initializes the core components—Tokenizer, TransformerModel, DecodingStrategy, TextGenerator, and Controller—
and wraps them behind a FastAPI application.

Endpoints:
-----------
GET /
    Health check endpoint. Returns a welcome message.

POST /generate
    Accepts a JSON body with a prompt string and returns generated text.
    Input: { "prompt": "Once upon a time" }
    Output: { "output": "Once upon a time, there was a..." }

Technologies Used:
------------------
- FastAPI for serving HTTP requests and built-in Swagger documentation
- PyTorch for model training/inference
- SentencePiece for subword tokenization
- Logging for request tracking and error handling

How to Run:
-----------
Start the API locally:
    uvicorn app:app --reload

Test from PowerShell:
    Invoke-RestMethod -Uri "http://127.0.0.1:8000/generate" `
        -Method Post `
        -Headers @{ "Content-Type" = "application/json" } `
        -Body '{"prompt": "Once upon a time"}'

Test from Bash:
    curl -X POST "http://127.0.0.1:8000/generate" \
         -H "Content-Type: application/json" \
         -d '{"prompt": "Once upon a time"}'

Swagger UI:
-----------
Access interactive docs at: http://127.0.0.1:8000/docs
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from controller.controller import Controller
from tokenizer.tokenizer import Tokenizer
from transformer.model import TransformerModel
from generator.text_generator import TextGenerator
from generator.decoding_strategy.decoding_strategy_factory import DecodingStrategyFactory, DecodingStrategyType
from checkpoint_manager.checkpoint_manager import CheckpointManager
import logging
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize model pipeline
tokenizer = Tokenizer()
model = TransformerModel() 
decoding_strategy = DecodingStrategyFactory.create_decoding_strategy(decoding_strategy=DecodingStrategyType.GREEDY)

# Load weights from latest checkpoint (if available)
CheckpointManager(model=model).load_best_model()

generator = TextGenerator(model, tokenizer, decoding_strategy)
controller = Controller(model, tokenizer, generator)

# FastAPI app
app = FastAPI(title="GPT-2 Text Generation API", version="1.0")

# Input schema
class PromptRequest(BaseModel):
    prompt: str

# Output schema
class GenerationResponse(BaseModel):
    output: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the AI Chatbot API"}

@app.post("/generate", response_model=GenerationResponse)
def generate_text(request: PromptRequest):
    start_time = time.time()  # Start timer
    try:
        logger.info(f"Received request: {request.prompt}")
        if not request.prompt.strip():
            raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

        generated_text = controller.predict_next_words(request.prompt)

        duration = time.time() - start_time
        logger.info(f"Generated response in {duration:.3f} seconds")

        return GenerationResponse(output=generated_text)

    except Exception as e:
        logger.error(f"Error during generation: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while generating text.")
