from enum import Enum
from generator.decoding_strategy.decoding_strategy import DecodingStrategy
from generator.decoding_strategy.greedy_decoding import GreedyDecoding
from generator.decoding_strategy.top_k_decoding import TopKDecoding
from generator.decoding_strategy.nucleus_decoding import NucleusDecoding

class DecodingStrategyType(Enum):
    """
    Enumeration of supported decoding strategy types.

    Attributes:
        GREEDY (str): Greedy decoding strategy.
        TOP_K (str): Top-k sampling strategy.
        NUCLEUS (str): Nucleus (top-p) sampling strategy.
    """
    GREEDY = "greedy"
    TOP_K = "top_k"
    NUCLEUS = "nucleus"


class DecodingStrategyFactory:
    """
    Factory class for creating decoding strategy instances.

    This factory encapsulates the logic for instantiating different types of
    decoding strategies (Greedy, Top-K, Nucleus), based on the given strategy type
    and additional hyperparameters.

    Methods:
        create_decoding_strategy(decoding_strategy, temperature, k, p):
            Creates and returns a decoding strategy instance based on the input strategy type.
    """

    @staticmethod
    def create_decoding_strategy(
        decoding_strategy: DecodingStrategyType = DecodingStrategyType.GREEDY,
        temperature: float = 0.9,
        k: int = 50,
        p: float = 0.9
    ) -> DecodingStrategy:
        """
        Creates a decoding strategy instance based on the provided type and parameters.

        Args:
            decoding_strategy (DecodingStrategyType): Type of decoding strategy to create.
            temperature (float): Sampling temperature to control randomness (used in all strategies).
            k (int): Number of top tokens to consider for Top-K decoding.
            p (float): Cumulative probability threshold for Nucleus decoding.

        Returns:
            DecodingStrategy: An instance of a concrete decoding strategy.

        Raises:
            ValueError: If an unknown decoding strategy type is provided.
        """
        if decoding_strategy == DecodingStrategyType.TOP_K:
            return TopKDecoding(k=k, temperature=temperature)
        elif decoding_strategy == DecodingStrategyType.GREEDY:
            return GreedyDecoding(temperature=temperature)
        elif decoding_strategy == DecodingStrategyType.NUCLEUS:
            return NucleusDecoding(p=p, temperature=temperature)
        else:
            raise ValueError(f"Unknown DecodingStrategyType")
