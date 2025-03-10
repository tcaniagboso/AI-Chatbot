from tokenizer.tokenizer import Tokenizer, SingletonTokenizer
from transformer.model import TransformerModel
from generator.text_generator import TextGenerator
from transformer.config import device

import os
import torch

class Controller:
    def __init__(self, model : TransformerModel, tokenizer : Tokenizer, generator : TextGenerator):
        self.model = model
        self.tokenizer = tokenizer
        self.generator = generator
    
    def run(self):
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
    print(f"Device: {device}")
    tokenizer = SingletonTokenizer()

    model = TransformerModel()

    # Load weights from latest checkpoint (if available)
    load_best_model(model)

    generator = TextGenerator(model, tokenizer)
    controller = Controller(model, tokenizer, generator)
    controller.run()