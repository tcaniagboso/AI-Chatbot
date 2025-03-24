**Transformer-Based Text Generation (Next-Word Prediction)**

  This project implements a decoder-only Transformer model for next-word prediction using **PyTorch**. It includes training, evaluation (perplexity), and text generation with support for 
  multiple decoding strategies: **greedy**, **top-k sampling**, and **nucleus (top-p) sampling**.

**Features**
  1. Custom Transformer model trained from scratch
  
  2. SentencePiece tokenizer (BPE-based)
  
  3. Wikitext-2 dataset by default (OpenWebText also supported)
  
  4. Modular code structure
  
  5. Configurable hyperparameters
  
  6. Text generation via CLI with multiple decoding strategies
  
  8. Logging and checkpointing support

**Getting Started**
  1. **Install Dependencies**

     run `pip install -r requirements.txt`
     
  3. **Train the Tokenizer**

     Make sure you have your dataset ready (e.g., Wikitext-2). To train a tokenizer:

     run `python tokenizer/train_sentencepiece.py`
     
  5. **Train the Transformer Model**

     Update model hyperparameters in config.py if needed. Update the number of epochs in the `trainer/trainer.py`, by default it has been set to 3, then:

     run `python trainer/trainer.py`
     
  7. **Run the Text Generator**

     Once the model is trained, generate text interactively:

     run `python controller/controller.py`

**Configuration**

  Edit the `transformer/config.py` file to set training parameters:
  
  ```
  batch_size = 8              # Number of sequences per batch
  block_size = 256            # Maximum context length
  learning_rate = 3e-4
  d_model = 512               # Embedding size
  n_head = 8                  # Number of attention heads
  n_layer = 6                 # Transformer layers
  dropout = 0.2
  vocab_size = 32000          # Based on trained tokenizer
  device = "cuda" if torch.cuda.is_available() else "cpu"
  ```

**Dataset Support**

  1. Wikitext-2 (default)
  
  2. OpenWebText (optional)

**Notes**

  1. All training logs and checkpoints are saved automatically.
  
  2. Make sure your tokenizer is trained before training the model or generating text.
  
  3. Hyperparameters are easily tunable in `transformer/config.py`.
  
  4. Text generation is done using next-word prediction, not factual answering. For QA-style and Classification-style tasks, fine-tuning is required.
