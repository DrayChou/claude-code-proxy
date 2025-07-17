import tiktoken
import logging
from typing import Dict, Any, List, Union, Optional
from src.core.config import config

logger = logging.getLogger(__name__)


class TokenManager:
    """Handles token counting and truncation for OpenAI models."""

    def __init__(self):
        self.encoders = {}

    def get_encoder(self, model: str):
        """Get or create tiktoken encoder for the specified model."""
        if model not in self.encoders:
            try:
                # Try to get encoding for the specific model
                self.encoders[model] = tiktoken.encoding_for_model(model)
            except KeyError:
                # Fallback to cl100k_base encoding for most GPT models
                logger.warning(
                    f"Model {model} not found, using cl100k_base encoding")
                self.encoders[model] = tiktoken.get_encoding("cl100k_base")
        return self.encoders[model]

    def count_tokens(self, text: str, model: str) -> int:
        """Count tokens in a text string for a specific model."""
        if not text:
            return 0
        encoder = self.get_encoder(model)
        return len(encoder.encode(text))

    def count_message_tokens(self, messages: List[Dict[str, Any]], model: str) -> int:
        """Count tokens in a list of messages."""
        encoder = self.get_encoder(model)
        total_tokens = 0

        # Add message overhead tokens (varies by model)
        tokens_per_message = 3  # Default for most models
        tokens_per_name = 1

        if "gpt-3.5-turbo" in model:
            tokens_per_message = 4
            tokens_per_name = -1
        elif "gpt-4" in model:
            tokens_per_message = 3
            tokens_per_name = 1

        for message in messages:
            total_tokens += tokens_per_message

            # Count tokens in role
            if "role" in message:
                total_tokens += len(encoder.encode(message["role"]))

            # Count tokens in content
            if "content" in message:
                content = message["content"]
                if isinstance(content, str):
                    total_tokens += len(encoder.encode(content))
                elif isinstance(content, list):
                    # Handle multimodal content
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text_content = item.get("text", "")
                            total_tokens += len(encoder.encode(text_content))
                        elif isinstance(item, dict) and item.get("type") == "image_url":
                            # Rough estimation for images (adjust based on your needs)
                            total_tokens += 85  # Base token cost for images

            # Count tokens in name field if present
            if "name" in message:
                total_tokens += tokens_per_name
                total_tokens += len(encoder.encode(message["name"]))

            # Count tokens in function calls if present
            if "tool_calls" in message:
                for tool_call in message["tool_calls"]:
                    if "function" in tool_call:
                        func = tool_call["function"]
                        if "name" in func:
                            total_tokens += len(encoder.encode(func["name"]))
                        if "arguments" in func:
                            total_tokens += len(
                                encoder.encode(func["arguments"]))

        # Add completion priming tokens
        total_tokens += 3
        return total_tokens

    def truncate_text(self, text: str, max_tokens: int, model: str,
                      preserve_ending: bool = False) -> str:
        """Truncate text to fit within token limit."""
        if not text:
            return text

        encoder = self.get_encoder(model)
        tokens = encoder.encode(text)

        if len(tokens) <= max_tokens:
            return text

        if preserve_ending:
            # Keep the end of the text
            truncated_tokens = tokens[-max_tokens:]
        else:
            # Keep the beginning of the text
            truncated_tokens = tokens[:max_tokens]

        return encoder.decode(truncated_tokens)

    def should_proactively_compress(self, messages: List[Dict[str, Any]], model: str) -> bool:
        """Check if messages should be proactively compressed."""
        from src.core.config import config
        
        context_window = self.get_context_window_size(model)
        current_tokens = self.count_message_tokens(messages, model)
        threshold = int(context_window * config.proactive_compression_threshold)
        
        return current_tokens >= threshold

    def proactively_compress_messages(self, messages: List[Dict[str, Any]], model: str) -> List[Dict[str, Any]]:
        """Proactively compress messages before hitting token limit."""
        from src.core.config import config
        
        context_window = self.get_context_window_size(model)
        target_tokens = int(context_window * config.compression_target_ratio)
        
        logger.info(f"Proactively compressing messages to {target_tokens} tokens")
        return self.truncate_messages(messages, target_tokens, model)

    def truncate_messages(self, messages: List[Dict[str, Any]],
                          max_tokens: int, model: str) -> List[Dict[str, Any]]:
        """Truncate messages to fit within token limit."""
        if not messages:
            return messages

        current_tokens = self.count_message_tokens(messages, model)

        if current_tokens <= max_tokens:
            return messages

        logger.warning(
            f"Messages exceed token limit ({current_tokens} > {max_tokens}), truncating...")

        # Enhanced strategy: Keep system messages, preserve recent conversation flow
        truncated_messages = []
        system_messages = []
        user_assistant_messages = []

        # Separate system messages from user/assistant messages
        for msg in messages:
            if msg.get("role") == "system":
                system_messages.append(msg)
            else:
                user_assistant_messages.append(msg)

        # Always include system messages (they're usually short and important)
        truncated_messages.extend(system_messages)

        # Calculate remaining token budget
        system_tokens = self.count_message_tokens(system_messages, model)
        remaining_tokens = max_tokens - system_tokens

        # Preserve recent conversational flow by keeping pairs of user-assistant messages
        if user_assistant_messages:
            # Start from the most recent messages and work backwards
            i = len(user_assistant_messages) - 1
            recent_messages = []
            
            while i >= 0 and remaining_tokens > 0:
                msg = user_assistant_messages[i]
                msg_tokens = self.count_message_tokens([msg], model)
                
                # If this message fits, add it
                if msg_tokens <= remaining_tokens:
                    recent_messages.insert(0, msg)
                    remaining_tokens -= msg_tokens
                    i -= 1
                else:
                    # Try to truncate the message content if it's a text message
                    if (msg.get("role") in ["user", "assistant"] and
                            isinstance(msg.get("content"), str)):
                        
                        # Reserve tokens for message overhead
                        content_token_budget = remaining_tokens - 10
                        if content_token_budget > 100:  # Only truncate if we have reasonable space
                            truncated_content = self.truncate_text(
                                msg["content"], content_token_budget, model, preserve_ending=True
                            )
                            truncated_msg = msg.copy()
                            truncated_msg["content"] = f"[...truncated...]\n{truncated_content}"
                            recent_messages.insert(0, truncated_msg)
                    break
            
            # Add the preserved messages after system messages
            truncated_messages.extend(recent_messages)

        final_tokens = self.count_message_tokens(truncated_messages, model)
        logger.info(
            f"Truncated messages from {current_tokens} to {final_tokens} tokens")

        return truncated_messages

    def get_context_window_size(self, model: str) -> int:
        """Get the context window size for a model."""
        model_limits = {
            "gpt-4o": 128000,
            "gpt-4o-2024-08-06": 128000,
            "gpt-4o-mini": 128000,
            "gpt-4-turbo": 128000,
            "gpt-4": 8192,
            "gpt-3.5-turbo": 16385,
            "gpt-3.5-turbo-16k": 16385,
        }

        # Find best match for model name
        for known_model, limit in model_limits.items():
            if known_model in model:
                return limit

        # Default to conservative limit
        return 8192


# Global token manager instance
token_manager = TokenManager()
