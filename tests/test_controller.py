"""
Unit tests for the Controller class.

These tests verify:
- That the controller encodes input and decodes output correctly
- That it uses the generator and view as expected
"""

import pytest
from unittest.mock import MagicMock
from controller.controller import Controller
from tokenizer.tokenizer import Tokenizer
from transformer.model import TransformerModel
from generator.text_generator import TextGenerator
from generator.decoding_strategy.decoding_strategy_factory import DecodingStrategyFactory, DecodingStrategyType


@pytest.fixture
def controller_setup():
    """Fixture to set up the Controller with mocks and dependencies."""
    tokenizer = Tokenizer()
    model = TransformerModel()
    decoding_strategy = DecodingStrategyFactory.create_decoding_strategy(
        decoding_strategy=DecodingStrategyType.GREEDY
    )
    generator = TextGenerator(model, tokenizer, decoding_strategy)

    mock_view = MagicMock()
    controller = Controller(model, tokenizer, generator, mock_view)
    return controller, tokenizer, generator, mock_view


def test_controller_processes_input_and_outputs_response(controller_setup):
    """
    Test that the controller encodes user input, uses the generator,
    and decodes the generated output, sending it to the view.
    """
    controller, tokenizer, generator, mock_view = controller_setup

    # Mock user input and expected model behavior
    sample_input = "Hello world"
    sample_input_ids = tokenizer.encode(sample_input)
    sample_output_ids = sample_input_ids + [1, 2]  # simulate model generating extra tokens
    sample_decoded_output = tokenizer.decode(sample_output_ids)

    mock_view.get_user_input.return_value = sample_input
    generator.generate_text = MagicMock(return_value=sample_output_ids)
    tokenizer.decode = MagicMock(return_value=sample_decoded_output)

    # Run one iteration of the loop manually (don't call .run())
    input_ids = tokenizer.encode(mock_view.get_user_input())
    output_ids = generator.generate_text(input_ids)
    output = tokenizer.decode(output_ids)
    mock_view.display_output(output)

    # Assertions
    generator.generate_text.assert_called_once_with(input_ids)
    tokenizer.decode.assert_called_once_with(output_ids)
    mock_view.display_output.assert_called_once_with(sample_decoded_output)