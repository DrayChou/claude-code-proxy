import tiktoken
import logging
import json
import asyncio
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
        
        # Try AI compression first if enabled
        if config.enable_ai_compression:
            try:
                return self.ai_compress_messages(messages, model, target_tokens)
            except Exception as e:
                logger.warning(f"AI compression failed, falling back to truncation: {e}")
        
        # Fallback to truncation
        return self.truncate_messages(messages, target_tokens, model)

    def ai_compress_messages(self, messages: List[Dict[str, Any]], model: str, target_tokens: int) -> List[Dict[str, Any]]:
        """Use AI to intelligently compress messages."""
        if not messages:
            return messages
            
        # Separate system messages from conversation messages
        system_messages = []
        conversation_messages = []
        
        for msg in messages:
            if msg.get("role") == "system":
                system_messages.append(msg)
            else:
                conversation_messages.append(msg)
        
        # Always keep system messages
        compressed_messages = system_messages[:]
        
        # Calculate tokens used by system messages
        system_tokens = self.count_message_tokens(system_messages, model)
        remaining_tokens = target_tokens - system_tokens
        
        if remaining_tokens <= 0 or not conversation_messages:
            return compressed_messages
            
        # If conversation is small enough, keep it as is
        conversation_tokens = self.count_message_tokens(conversation_messages, model)
        if conversation_tokens <= remaining_tokens:
            compressed_messages.extend(conversation_messages)
            return compressed_messages
            
        # Get AI summary of older messages
        try:
            summary = self._get_ai_summary(conversation_messages, model)
            if summary:
                # Create summary message
                summary_message = {
                    "role": "assistant",
                    "content": f"[Previous conversation summary: {summary}]"
                }
                
                # Keep recent messages + summary
                recent_messages = []
                summary_tokens = self.count_message_tokens([summary_message], model)
                remaining_for_recent = remaining_tokens - summary_tokens
                
                # Intelligently select recent messages to preserve
                recent_candidates = conversation_messages[-15:]  # Consider last 15 messages
                
                # Prioritize recent important messages
                important_recent = []
                regular_recent = []
                
                for msg in recent_candidates:
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        content_lower = content.lower()
                        is_important = (
                            msg.get("role") == "user" or  # User questions are important
                            any(keyword in content_lower for keyword in [
                                'error', 'exception', 'failed', 'warning', 'bug',
                                'file', 'function', 'class', 'config', 'test'
                            ]) or
                            len(content) > 300  # Substantial messages
                        )
                        
                        if is_important:
                            important_recent.append(msg)
                        else:
                            regular_recent.append(msg)
                
                # Add important messages first, then regular ones if space allows
                for msg_list in [important_recent, regular_recent]:
                    for msg in reversed(msg_list):  # Most recent first
                        msg_tokens = self.count_message_tokens([msg], model)
                        if msg_tokens <= remaining_for_recent:
                            recent_messages.insert(0, msg)
                            remaining_for_recent -= msg_tokens
                        elif remaining_for_recent > 100:  # Try to truncate if some space left
                            content = msg.get("content", "")
                            if isinstance(content, str) and len(content) > 200:
                                truncated_content = content[-150:]  # Keep ending
                                truncated_msg = msg.copy()
                                truncated_msg["content"] = f"[...]{truncated_content}"
                                msg_tokens = self.count_message_tokens([truncated_msg], model)
                                if msg_tokens <= remaining_for_recent:
                                    recent_messages.insert(0, truncated_msg)
                                    remaining_for_recent -= msg_tokens
                
                # Build final compressed messages
                compressed_messages.append(summary_message)
                compressed_messages.extend(recent_messages)
                
                final_tokens = self.count_message_tokens(compressed_messages, model)
                logger.info(f"AI compressed messages to {final_tokens} tokens (target: {target_tokens})")
                return compressed_messages
                
        except Exception as e:
            logger.error(f"AI compression failed: {e}")
            
        # If AI compression fails, fallback to truncation
        return self.truncate_messages(messages, target_tokens, model)

    def _get_ai_summary(self, messages: List[Dict[str, Any]], model: str) -> Optional[str]:
        """Get AI summary of conversation messages."""
        try:
            # Import here to avoid circular dependency
            import httpx
            
            # Get compression model
            compression_model_name = getattr(config, f"{config.compression_model}_model")
            
            # Prepare conversation text for summarization, prioritizing important content
            conversation_text = ""
            important_keywords = [
                'error', 'exception', 'traceback', 'failed', 'warning',
                'config', 'setting', 'environment', 'variable',
                'api', 'endpoint', 'database', 'schema',
                'test', 'debug', 'fix', 'bug',
                'file', 'path', 'function', 'class', 'method',
                'import', 'dependency', 'library', 'version'
            ]
            
            # Separate messages into important and regular
            important_messages = []
            regular_messages = []
            
            for msg in messages[:-5]:  # Process all but last 5 messages
                role = msg.get("role", "")
                content = msg.get("content", "")
                if isinstance(content, str) and content.strip():
                    content_lower = content.lower()
                    is_important = (
                        role == "system" or  # System messages are always important
                        any(keyword in content_lower for keyword in important_keywords) or
                        len(content) > 500  # Long messages likely contain important details
                    )
                    
                    if is_important:
                        important_messages.append(f"[IMPORTANT] {role}: {content}")
                    else:
                        regular_messages.append(f"{role}: {content}")
            
            # Prioritize important messages in the summary
            if important_messages:
                conversation_text += "=== CRITICAL INFORMATION ===\n"
                conversation_text += "\n\n".join(important_messages[:10])  # Limit to 10 most important
                conversation_text += "\n\n"
            
            if regular_messages:
                conversation_text += "=== GENERAL CONVERSATION ===\n"
                conversation_text += "\n\n".join(regular_messages[:5])  # Limit regular messages
            
            if not conversation_text.strip():
                return None
                
            # Limit input size for summary request
            if len(conversation_text) > 8000:  # Rough character limit
                conversation_text = conversation_text[:8000] + "..."
            
            # Create summary request
            summary_request = {
                "model": compression_model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": """You are an expert code assistant that creates intelligent conversation summaries for ongoing development sessions. Your goal is to preserve CRITICAL information while reducing token usage.

PRIORITY 1 - ALWAYS PRESERVE:
• Current task/objective and its status
• File paths, function names, and class names mentioned
• Error messages, stack traces, and debugging information  
• Configuration settings and environment variables
• API endpoints, database schemas, and data structures
• Dependencies, imports, and library versions
• Test results and validation outcomes

PRIORITY 2 - PRESERVE WHEN RELEVANT:
• Code patterns and architectural decisions
• Performance metrics and optimization details
• Security considerations and access controls
• Deployment and infrastructure details
• User requirements and business logic

PRIORITY 3 - SUMMARIZE OR OMIT:
• Casual conversation and acknowledgments
• Repetitive explanations of basic concepts
• Multiple similar examples (keep one representative)
• Verbose explanations (condense to key points)

FORMAT: Create a structured summary with:
1. **Current Task**: [What we're working on]
2. **Key Files**: [Important file paths and components]
3. **Critical Info**: [Errors, configs, decisions]
4. **Context**: [Relevant background for continuing work]

Keep summary under 200 words but ensure no critical development context is lost."""
                    },
                    {
                        "role": "user", 
                        "content": f"Summarize this development conversation, preserving all critical technical information:\n\n{conversation_text}"
                    }
                ],
                "max_tokens": 300,
                "temperature": 0.1
            }
            
            # Make API call
            with httpx.Client(
                base_url=config.openai_base_url,
                headers={"Authorization": f"Bearer {config.openai_api_key}"},
                timeout=30.0
            ) as client:
                response = client.post("/chat/completions", json=summary_request)
                response.raise_for_status()
                
                result = response.json()
                summary = result["choices"][0]["message"]["content"]
                logger.debug(f"Generated AI summary: {summary[:100]}...")
                return summary
                
        except Exception as e:
            logger.error(f"Failed to get AI summary: {e}")
            return None

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
