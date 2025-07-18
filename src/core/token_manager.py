import tiktoken
import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from .monitor import system_monitor

logger = logging.getLogger(__name__)


class TokenManager:
    """优化的Token管理器 - 分层截断策略"""

    def __init__(self):
        self.encoders = {}
        self.importance_keywords = {
            'critical': ['error', 'exception', 'failed', 'bug', 'fix', 'crash', 'traceback'],
            'high': ['function', 'class', 'config', 'api', 'database', 'schema', 'endpoint'],
            'medium': ['test', 'debug', 'performance', 'optimization', 'import', 'dependency'],
            'contextual': ['file', 'path', 'method', 'variable', 'parameter', 'return']
        }

    def get_encoder(self, model: str):
        """Get or create tiktoken encoder for the specified model."""
        if model not in self.encoders:
            try:
                # Claude模型映射到对应的tiktoken编码
                claude_model_mapping = {
                    "claude-3-5-haiku-20241022": "cl100k_base",
                    "claude-sonnet-4-20250514": "cl100k_base",
                    "claude-3-5-sonnet-20241022": "cl100k_base",
                    "claude-3-opus-20240229": "cl100k_base",
                    "claude-3-haiku-20240307": "cl100k_base",
                }

                # 检查是否为 Claude 模型
                if model in claude_model_mapping:
                    logger.debug(
                        f"使用 Claude 模型映射：{model} -> "
                        f"{claude_model_mapping[model]}"
                    )
                    self.encoders[model] = tiktoken.get_encoding(
                        claude_model_mapping[model]
                    )
                else:
                    # 尝试获取特定模型的编码
                    self.encoders[model] = tiktoken.encoding_for_model(model)
            except KeyError:
                # 回退到 cl100k_base 编码
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
                            # Rough estimation for images
                            total_tokens += 85  # Base token cost

            # Count tokens in name field if present
            if "name" in message:
                total_tokens += tokens_per_name
                total_tokens += len(encoder.encode(message["name"]))

        # Add 3 tokens for assistant response priming
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

    def smart_truncate(self, messages: List[Dict[str, Any]], max_tokens: int, model: str) -> List[Dict[str, Any]]:
        """智能分层截断策略 - 主入口"""
        if not messages:
            return messages

        original_tokens = self.count_message_tokens(messages, model)
        
        if original_tokens <= max_tokens:
            return messages

        # 记录压缩开始
        system_monitor.record_compression(original_tokens, 0)  # 先记录原始token数
        
        # 计算压力等级
        pressure_ratio = original_tokens / max_tokens
        
        logger.info(f"开始智能截断: {original_tokens} -> {max_tokens} tokens (压力比: {pressure_ratio:.2f})")
        
        if pressure_ratio <= 1.2:  # 轻微超限 (20%以内)
            result = self._light_truncation(messages, max_tokens, model)
            logger.info("使用轻度截断策略")
        elif pressure_ratio <= 1.8:  # 中等压力 (80%以内)
            result = self._medium_truncation(messages, max_tokens, model)
            logger.info("使用中度截断策略")
        else:  # 重度压力
            result = self._heavy_truncation(messages, max_tokens, model)
            logger.info("使用重度截断策略")

        # 记录最终结果
        final_tokens = self.count_message_tokens(result, model)
        system_monitor.record_compression(original_tokens, final_tokens)
        
        compression_ratio = (original_tokens - final_tokens) / original_tokens
        logger.info(f"截断完成: {original_tokens} -> {final_tokens} tokens (压缩率: {compression_ratio:.1%})")
        
        return result

    def _light_truncation(self, messages: List[Dict[str, Any]], max_tokens: int, model: str) -> List[Dict[str, Any]]:
        """轻度截断：只移除最老的普通消息"""
        target_tokens = int(max_tokens * 0.95)  # 留5%缓冲
        
        # 分离消息类型
        system_msgs = [msg for msg in messages if msg.get('role') == 'system']
        other_msgs = [msg for msg in messages if msg.get('role') != 'system']
        
        # 保护最近3轮对话
        recent_msgs = self._get_recent_messages(other_msgs, turns=3)
        old_msgs = [msg for msg in other_msgs if msg not in recent_msgs]
        
        # 构建结果：系统消息 + 部分旧消息 + 最近消息
        result = system_msgs[:]
        current_tokens = self.count_message_tokens(result + recent_msgs, model)
        
        # 从最新的旧消息开始添加
        for msg in reversed(old_msgs):
            msg_tokens = self.count_message_tokens([msg], model)
            if current_tokens + msg_tokens <= target_tokens:
                result.append(msg)
                current_tokens += msg_tokens
            else:
                break
        
        result.extend(recent_msgs)
        return self._sort_messages_by_order(result, messages)

    def _medium_truncation(self, messages: List[Dict[str, Any]], max_tokens: int, model: str) -> List[Dict[str, Any]]:
        """中度截断：基于重要性的智能选择"""
        target_tokens = int(max_tokens * 0.90)  # 留10%缓冲
        
        # 分离消息类型
        system_msgs = [msg for msg in messages if msg.get('role') == 'system']
        other_msgs = [msg for msg in messages if msg.get('role') != 'system']
        
        # 保护最近2轮对话
        recent_msgs = self._get_recent_messages(other_msgs, turns=2)
        history_msgs = [msg for msg in other_msgs if msg not in recent_msgs]
        
        # 对历史消息评分
        scored_msgs = [(msg, self._score_message(msg)) for msg in history_msgs]
        scored_msgs.sort(key=lambda x: x[1], reverse=True)  # 按重要性排序
        
        # 构建结果
        result = system_msgs[:]
        result.extend(recent_msgs)
        current_tokens = self.count_message_tokens(result, model)
        
        # 添加重要的历史消息
        for msg, score in scored_msgs:
            msg_tokens = self.count_message_tokens([msg], model)
            if current_tokens + msg_tokens <= target_tokens:
                result.append(msg)
                current_tokens += msg_tokens
            else:
                break
        
        return self._sort_messages_by_order(result, messages)

    def _heavy_truncation(self, messages: List[Dict[str, Any]], max_tokens: int, model: str) -> List[Dict[str, Any]]:
        """重度截断：保留核心信息 + 摘要"""
        target_tokens = int(max_tokens * 0.85)  # 留15%缓冲
        
        # 分离消息类型
        system_msgs = [msg for msg in messages if msg.get('role') == 'system']
        other_msgs = [msg for msg in messages if msg.get('role') != 'system']
        
        # 只保留最近1轮对话
        recent_msgs = self._get_recent_messages(other_msgs, turns=1)
        history_msgs = [msg for msg in other_msgs if msg not in recent_msgs]
        
        # 生成历史摘要
        summary = self._create_simple_summary(history_msgs)
        
        # 构建结果
        result = system_msgs[:]
        
        if summary:
            summary_msg = {
                "role": "assistant",
                "content": f"[对话摘要] {summary}"
            }
            result.append(summary_msg)
        
        result.extend(recent_msgs)
        
        # 如果仍然超限，截断最长的消息内容
        current_tokens = self.count_message_tokens(result, model)
        if current_tokens > target_tokens:
            result = self._truncate_long_messages(result, target_tokens, model)
        
        return result

    def _score_message(self, message: Dict[str, Any]) -> float:
        """基于内容重要性评分消息"""
        content = str(message.get('content', '')).lower()
        score = 0.0
        
        # 角色基础分
        role = message.get('role', '')
        if role == 'system':
            score = 1.0  # 系统消息最重要
        elif role == 'user':
            score = 0.6  # 用户问题重要
        else:
            score = 0.4  # 助手回复
        
        # 关键词加分
        for level, keywords in self.importance_keywords.items():
            keyword_count = sum(1 for kw in keywords if kw in content)
            if level == 'critical':
                score += keyword_count * 0.3
            elif level == 'high':
                score += keyword_count * 0.2
            elif level == 'medium':
                score += keyword_count * 0.1
            elif level == 'contextual':
                score += keyword_count * 0.05
        
        # 长度加分（适度）
        if len(content) > 200:
            score += 0.1
        
        return min(score, 1.0)  # 限制最高分

    def _create_simple_summary(self, messages: List[Dict[str, Any]]) -> str:
        """创建简单的规则摘要（无需API调用）"""
        if not messages:
            return ""
        
        # 提取关键信息
        key_points = []
        file_mentions = set()
        error_mentions = []
        important_actions = []
        
        for msg in messages:
            content = str(msg.get('content', ''))
            
            # 提取文件路径
            files = re.findall(r'[\w/\\.-]+\.(py|js|ts|json|yaml|yml|txt|md|html|css)', content)
            file_mentions.update(files[:3])  # 最多3个文件
            
            # 提取错误信息
            if any(kw in content.lower() for kw in ['error', 'exception', 'failed']):
                error_part = content[:100] + "..." if len(content) > 100 else content
                error_mentions.append(error_part)
            
            # 提取重要操作
            if any(kw in content.lower() for kw in ['config', 'setup', 'install', 'deploy']):
                action_part = content[:80] + "..." if len(content) > 80 else content
                important_actions.append(action_part)
        
        # 构建摘要
        summary_parts = []
        if file_mentions:
            summary_parts.append(f"涉及文件: {', '.join(list(file_mentions)[:3])}")
        if error_mentions:
            summary_parts.append(f"错误信息: {error_mentions[0]}")
        if important_actions:
            summary_parts.append(f"重要操作: {important_actions[0]}")
        
        # 添加消息数量信息
        summary_parts.append(f"历史消息{len(messages)}条")
        
        return " | ".join(summary_parts)

    def _get_recent_messages(self, messages: List[Dict[str, Any]], turns: int) -> List[Dict[str, Any]]:
        """获取最近的N轮对话"""
        if not messages:
            return []
        
        recent = []
        turn_count = 0
        
        for msg in reversed(messages):
            if msg.get('role') == 'system':
                continue
                
            recent.insert(0, msg)
            
            if msg.get('role') == 'user':
                turn_count += 1
                if turn_count >= turns:
                    break
        
        return recent

    def _sort_messages_by_order(self, result_messages: List[Dict[str, Any]], 
                               original_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """按原始顺序排序消息"""
        # 创建原始消息的索引映射
        original_indices = {id(msg): i for i, msg in enumerate(original_messages)}
        
        # 按原始顺序排序
        result_messages.sort(key=lambda msg: original_indices.get(id(msg), float('inf')))
        
        return result_messages

    def _truncate_long_messages(self, messages: List[Dict[str, Any]], 
                               max_tokens: int, model: str) -> List[Dict[str, Any]]:
        """截断过长的消息内容"""
        current_tokens = self.count_message_tokens(messages, model)
        
        if current_tokens <= max_tokens:
            return messages
        
        # 找到最长的可截断消息
        truncatable_msgs = []
        for i, msg in enumerate(messages):
            if (msg.get('role') in ['user', 'assistant'] and 
                isinstance(msg.get('content'), str) and
                len(msg['content']) > 100):
                truncatable_msgs.append((i, msg, len(msg['content'])))
        
        # 按长度排序
        truncatable_msgs.sort(key=lambda x: x[2], reverse=True)
        
        # 逐个截断直到满足要求
        result = messages[:]
        for i, msg, length in truncatable_msgs:
            if current_tokens <= max_tokens:
                break
            
            # 计算需要截断的token数
            excess_tokens = current_tokens - max_tokens + 50  # 留50token缓冲
            
            # 截断消息
            original_content = msg['content']
            truncated_content = self.truncate_text(
                original_content, 
                self.count_tokens(original_content, model) - excess_tokens,
                model, 
                preserve_ending=True
            )
            
            result[i] = msg.copy()
            result[i]['content'] = f"[截断] {truncated_content}"
            
            current_tokens = self.count_message_tokens(result, model)
        
        return result

    def get_context_window_size(self, model: str) -> int:
        """Get the context window size for a model."""
        context_windows = {
            # GPT models
            "gpt-3.5-turbo": 4096,
            "gpt-3.5-turbo-16k": 16384,
            "gpt-4": 8192,
            "gpt-4-32k": 32768,
            "gpt-4-turbo": 128000,
            "gpt-4o": 128000,
            # Claude models
            "claude-3-haiku-20240307": 200000,
            "claude-3-sonnet-20240229": 200000,
            "claude-3-opus-20240229": 200000,
            "claude-3-5-sonnet-20240620": 200000,
            "claude-3-5-sonnet-20241022": 200000,
            "claude-3-5-haiku-20241022": 200000,
            "claude-sonnet-4-20250514": 200000,
        }
        
        return context_windows.get(model, 8192)  # Default to 8K if unknown


# Global token manager instance
token_manager = TokenManager()