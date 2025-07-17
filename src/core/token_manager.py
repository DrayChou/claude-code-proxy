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

        # Strategy: Keep system message and recent messages
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

        # Add user/assistant messages from the end (most recent first)
        for msg in reversed(user_assistant_messages):
            msg_tokens = self.count_message_tokens([msg], model)

            if msg_tokens <= remaining_tokens:
                truncated_messages.insert(-len(system_messages)
                                          if system_messages else 0, msg)
                remaining_tokens -= msg_tokens
            else:
                # Try to truncate the message content if it's a text message
                if (msg.get("role") in ["user", "assistant"] and
                        isinstance(msg.get("content"), str)):

                    # Reserve tokens for message overhead
                    content_token_budget = remaining_tokens - 10
                    if content_token_budget > 50:  # Only truncate if we have reasonable space
                        truncated_content = self.truncate_text(
                            msg["content"], content_token_budget, model, preserve_ending=False
                        )
                        truncated_msg = msg.copy()
                        truncated_msg["content"] = truncated_content
                        truncated_messages.insert(-len(system_messages)
                                                  if system_messages else 0, truncated_msg)
                break

        final_tokens = self.count_message_tokens(truncated_messages, model)
        logger.info(
            f"Truncated messages from {current_tokens} to {final_tokens} tokens")
        
        logger.debug(f"原始截断策略返回 {len(truncated_messages)} 条消息")
        return truncated_messages

    def truncate_messages_smart(self, messages: List[Dict[str, Any]],
                                max_tokens: int, model: str,
                                preserve_recent_turns: int = 2,
                                compression_ratio: float = 0.3
                                ) -> List[Dict[str, Any]]:
        """
        智能截断策略：保护最近对话轮次，压缩历史内容
        
        Args:
            messages: 消息列表
            max_tokens: 最大token限制
            model: 模型名称
            preserve_recent_turns: 保护最近几轮完整对话
            compression_ratio: 历史内容压缩比例
        """
        if not messages:
            return messages
        
        current_tokens = self.count_message_tokens(messages, model)
        if current_tokens <= max_tokens:
            logger.debug(f"消息token数量({current_tokens})在限制内，无需压缩")
            return messages
        
        logger.info(f"启动智能压缩: {current_tokens} -> {max_tokens} tokens")
        logger.debug(f"压缩参数: 保留 {preserve_recent_turns} 轮对话, 压缩比 {compression_ratio}")
        
        # 分离不同类型的消息
        system_messages = []
        conversation_messages = []
        
        for msg in messages:
            if msg.get("role") == "system":
                system_messages.append(msg)
            else:
                conversation_messages.append(msg)
        
        # 始终保留系统消息
        result_messages = system_messages.copy()
        system_tokens = self.count_message_tokens(system_messages, model)
        remaining_tokens = max_tokens - system_tokens
        
        logger.debug(f"系统消息占用 {system_tokens} tokens，剩余 {remaining_tokens} tokens")
        
        if remaining_tokens <= 0:
            logger.warning("系统消息已超出token限制！")
            return system_messages
        
        # 提取最近的对话轮次
        recent_messages = self._extract_recent_turns(conversation_messages, preserve_recent_turns)
        recent_tokens = self.count_message_tokens(recent_messages, model)
        
        logger.debug(f"最近 {preserve_recent_turns} 轮对话占用 {recent_tokens} tokens")
        
        # 如果最近对话已经超限，采用保守策略
        if recent_tokens >= remaining_tokens:
            logger.warning("最近对话已接近token限制，采用保守截断")
            return result_messages + self._conservative_truncate(recent_messages, remaining_tokens, model)
        
        # 计算历史消息
        history_messages = conversation_messages[:-len(recent_messages)] if recent_messages else conversation_messages
        available_for_history = remaining_tokens - recent_tokens
        
        logger.debug(f"历史消息可用 {available_for_history} tokens")
        
        # 压缩历史消息
        compressed_history = []
        if history_messages and available_for_history > 100:  # 至少保留一些历史
            compressed_history = self._compress_historical_messages(
                history_messages, available_for_history, model, compression_ratio
            )
        
        # 组装最终结果
        result_messages.extend(compressed_history)
        result_messages.extend(recent_messages)
        
        final_tokens = self.count_message_tokens(result_messages, model)
        logger.info(f"智能压缩完成: {current_tokens} -> {final_tokens} tokens")
        
        # 验证压缩效果
        if final_tokens > max_tokens:
            logger.warning(f"压缩后仍超限: {final_tokens} > {max_tokens}")
        else:
            compression_rate = (current_tokens - final_tokens) / current_tokens * 100
            logger.debug(f"压缩成功，压缩率: {compression_rate:.1f}%")
        
        return result_messages
    
    def _extract_recent_turns(self, messages: List[Dict[str, Any]], 
                             preserve_turns: int) -> List[Dict[str, Any]]:
        """提取最近几轮完整的用户-助手对话"""
        if not messages or preserve_turns <= 0:
            return []
        
        # 从后往前找完整的对话轮次
        recent_messages = []
        turn_count = 0
        i = len(messages) - 1
        
        while i >= 0 and turn_count < preserve_turns:
            current_msg = messages[i]
            recent_messages.insert(0, current_msg)
            
            # 如果遇到用户消息，说明完成了一轮对话
            if current_msg.get("role") == "user":
                turn_count += 1
            
            i -= 1
        
        return recent_messages
    
    def _compress_historical_messages(self, messages: List[Dict[str, Any]],
                                    available_tokens: int, model: str,
                                    compression_ratio: float) -> List[Dict[str, Any]]:
        """压缩历史消息"""
        if not messages:
            return []
        
        current_history_tokens = self.count_message_tokens(messages, model)
        target_tokens = int(available_tokens * compression_ratio)
        
        logger.debug(f"历史消息压缩: {current_history_tokens} -> {target_tokens} tokens")
        
        if current_history_tokens <= target_tokens:
            return messages
        
        # 简单策略：从最老的消息开始删除
        compressed = []
        tokens_used = 0
        
        # 从最新的历史消息开始保留
        for msg in reversed(messages):
            msg_tokens = self.count_message_tokens([msg], model)
            if tokens_used + msg_tokens <= target_tokens:
                compressed.insert(0, msg)
                tokens_used += msg_tokens
            else:
                # 尝试截断这个消息的内容
                if (msg.get("role") in ["user", "assistant"] and 
                    isinstance(msg.get("content"), str) and 
                    target_tokens - tokens_used > 50):
                    
                    remaining_budget = target_tokens - tokens_used - 10
                    truncated_content = self.truncate_text(
                        msg["content"], remaining_budget, model, preserve_ending=True
                    )
                    truncated_msg = msg.copy()
                    truncated_msg["content"] = f"[历史对话摘要]...{truncated_content}"
                    compressed.insert(0, truncated_msg)
                break
        
        return compressed
    
    def _conservative_truncate(self, messages: List[Dict[str, Any]],
                              max_tokens: int, model: str) -> List[Dict[str, Any]]:
        """保守的截断策略，确保不超过token限制"""
        if not messages:
            return []
        
        result = []
        tokens_used = 0
        
        # 从最新消息开始保留
        for msg in reversed(messages):
            msg_tokens = self.count_message_tokens([msg], model)
            if tokens_used + msg_tokens <= max_tokens:
                result.insert(0, msg)
                tokens_used += msg_tokens
            else:
                break
        
        return result

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
