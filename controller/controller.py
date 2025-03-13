from tokenizer.tokenizer import Tokenizer, SingletonTokenizer
from transformer.model import TransformerModel
from generator.text_generator import TextGenerator
from transformer.config import device

import os
import torch

class Controller:
    """
    A simple interactive text generation controller that takes user input,
    processes it through a trained Transformer model, and generates a response.

    This class serves as the main interface for interacting with the text generation system.
    It continuously prompts the user for input, encodes the input using a tokenizer, 
    generates a response using a Transformer model, and then decodes the generated output.

    Attributes:
        model (TransformerModel): The trained Transformer model for next-word prediction.
        tokenizer (Tokenizer): The tokenizer used for encoding user input and decoding model output.
        generator (TextGenerator): The text generation engine that generates text based on input.

    Methods:
        run():
            Starts an interactive loop where the user inputs a text prompt, and the model generates a response.
    """
    def __init__(self, model : TransformerModel, tokenizer : Tokenizer, generator : TextGenerator):
        """
        Initializes the Controller with a trained model, tokenizer, and text generator.

        Args:
            model (TransformerModel): The trained Transformer model.
            tokenizer (Tokenizer): The tokenizer used for encoding and decoding text.
            generator (TextGenerator): The text generator that predicts the next words.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.generator = generator
    
    def run(self):
        """
        Starts an interactive text generation session.

        The user provides input, which is tokenized and fed into the model.
        The model generates a response using the chosen decoding strategy, and the result is displayed.
        """
        while True:
            user_input = input("User Input: ")

            input_ids = self.tokenizer.encode(user_input)
            output_ids = self.generator.generate_text(input_ids, decoding_strategy='greedy', temperature=0.8)
            output = self.tokenizer.decode(output_ids)

            print(f"Model Response: {output}")
            print()

def load_best_model(model, best_path = 'checkpoints/best/best_model.pt'):
        """
        Loads the best (lowest validation loss) model for evaluation or deployment.
        """
        if os.path.exists(best_path):
            checkpoint = torch.load(best_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model from {best_path}")
        else:
            print("No best model found. You may need to train first.")


if __name__ == '__main__':
    """
    Initializes the model, tokenizer, and generator, and starts the interactive text generation session.
    """
    print(f"Device: {device}")
    tokenizer = SingletonTokenizer()

    model = TransformerModel()

    # Load weights from latest checkpoint (if available)
    load_best_model(model)

    generator = TextGenerator(model, tokenizer)
    controller = Controller(model, tokenizer, generator)
    controller.run()