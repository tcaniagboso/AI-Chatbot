from tokenizer.tokenizer import Tokenizer
from transformer.model import TransformerModel
from generator.text_generator import TextGenerator
from transformer.config import device
from checkpoint_manager.checkpoint_manager import CheckpointManager

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
        self.model: TransformerModel = model
        self.tokenizer: Tokenizer = tokenizer
        self.generator: TextGenerator = generator
    
    def run(self) -> None:
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

if __name__ == '__main__':
    """
    Initializes the model, tokenizer, and generator, and starts the interactive text generation session.
    """
    print(f"Device: {device}")
    tokenizer = Tokenizer()

    model = TransformerModel()

    # Load weights from latest checkpoint (if available)
    CheckpointManager(model=model).load_best_model()

    generator = TextGenerator(model, tokenizer)
    controller = Controller(model, tokenizer, generator)
    controller.run()