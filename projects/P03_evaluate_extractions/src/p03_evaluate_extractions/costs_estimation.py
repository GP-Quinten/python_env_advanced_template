from typing import List, Dict, Any
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage, AssistantMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.exceptions import TokenizerException  # Import TokenizerException

model_prices: Dict[str, Dict[str, float]] = {
    "mistral-large-latest": {"input_per_million": 1.8, "output_per_million": 5.4},
    "mistral-medium-latest": {"input_per_million": 0.4, "output_per_million": 2.0},
    "mistral-small-latest": {"input_per_million": 0.1, "output_per_million": 0.3},
}


def estimate(
    prompts: List[str], generations: List[str], model: str = "mistral-large-latest"
) -> Dict[str, float]:
    """
    Estimate the cost of generating responses based on the number of tokens in prompts and generations.
    """
    if model not in model_prices:
        raise ValueError(f"Model {model} not found in price list.")
    else:
        print(f"Estimating costs for model: {model}")

    input_price: float = model_prices[model]["input_per_million"]
    output_price: float = model_prices[model]["output_per_million"]

    # Use the correct tokenizer for the selected model
    try:
        # print(f"Loading tokenizer for model: {model}")
        tokenizer: MistralTokenizer = MistralTokenizer.from_model(
            model.replace("-latest", ""), strict=True
        )
    except TokenizerException:
        # Fallback to Mistral-large tokenizer if Mistral-medium tokenizer is not available
        print(
            f"Tokenizer for model {model} not found. Using Mistral-large tokenizer as a fallback."
        )
        tokenizer: MistralTokenizer = MistralTokenizer.from_model(
            "mistral-large", strict=True
        )

    total_input_tokens: int = 0
    total_output_tokens: int = 0

    # Count input tokens (user prompts)
    for prompt in prompts:
        request = ChatCompletionRequest(
            messages=[UserMessage(content=prompt)], model=model
        )
        tokenized = tokenizer.encode_chat_completion(request)
        total_input_tokens += len(tokenized.tokens)

    # Count output tokens (generations)
    for generation in generations:
        request = ChatCompletionRequest(
            messages=[UserMessage(content=generation)], model=model
        )
        tokenized = tokenizer.encode_chat_completion(request)
        total_output_tokens += len(tokenized.tokens)

    # Calculate costs (per million tokens)
    input_cost: float = (total_input_tokens / 1_000_000) * input_price
    output_cost: float = (total_output_tokens / 1_000_000) * output_price

    return {
        "avg_input_tokens": total_input_tokens / len(prompts) if prompts else 0,
        "avg_output_tokens": (
            total_output_tokens / len(generations) if generations else 0
        ),
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost,
    }


# # Example usage
# result = estimate_generation_costs(
#     prompts=["Hello, how are you today?",],
#     generations=["I'm doing well, thank you!"],
#     model="mistral-large-latest"
# )
