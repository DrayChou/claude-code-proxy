import asyncio
import logging
import random
from typing import Callable, Any

logger = logging.getLogger(__name__)


class RetryHandler:
    """处理 API 调用重试，包括 429 错误和其他可重试错误"""

    def __init__(self, max_retries: int = 3, base_delay: float = 1.0,
                 max_delay: float = 60.0, exponential_base: float = 2.0,
                 jitter: bool = True):
        """
        初始化重试处理器

        Args:
            max_retries: 最大重试次数
            base_delay: 基础延迟时间（秒）
            max_delay: 最大延迟时间（秒）
            exponential_base: 指数退避基数
            jitter: 是否添加随机抖动
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retry_stats = {
            "total_attempts": 0,
            "successful_retries": 0,
            "failed_retries": 0,
            "rate_limit_hits": 0
        }

    async def execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """
        执行带重试机制的函数

        Args:
            func: 要执行的函数
            *args: 位置参数
            **kwargs: 关键字参数

        Returns:
            函数执行结果

        Raises:
            最后一次重试的异常
        """
        last_exception = None

        for attempt in range(self.max_retries + 1):
            self.retry_stats["total_attempts"] += 1

            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)

                if attempt > 0:
                    self.retry_stats["successful_retries"] += 1
                    logger.info(f"重试成功，第{attempt}次尝试")

                return result

            except Exception as e:
                last_exception = e

                # 检查是否应该重试
                if not self._should_retry(e, attempt):
                    raise e

                self.retry_stats["failed_retries"] += 1

                if self._is_rate_limit_error(e):
                    self.retry_stats["rate_limit_hits"] += 1
                    logger.warning(
                        f"触发429限流，准备重试 "
                        f"(尝试 {attempt + 1}/{self.max_retries + 1})"
                    )
                else:
                    logger.warning(
                        f"发生可重试错误，准备重试 "
                        f"(尝试 {attempt + 1}/{self.max_retries + 1}): {e}"
                    )

                if attempt < self.max_retries:
                    delay = self._calculate_delay(attempt)
                    logger.debug(f"等待 {delay:.2f} 秒后重试...")
                    await asyncio.sleep(delay)

        # 所有重试都失败
        logger.error(f"所有{self.max_retries + 1}次尝试均失败")
        raise last_exception

    def _should_retry(self, exception: Exception, attempt: int) -> bool:
        """判断是否应该重试"""
        if attempt >= self.max_retries:
            return False

        # 429 Too Many Requests
        if self._is_rate_limit_error(exception):
            return True

        # 5xx 服务器错误
        if self._is_server_error(exception):
            return True

        # 网络相关错误
        if self._is_network_error(exception):
            return True

        return False

    def _is_rate_limit_error(self, exception: Exception) -> bool:
        """检查是否为429限流错误"""
        error_str = str(exception).lower()
        return (
            "429" in error_str or
            "rate limit" in error_str or
            "too many requests" in error_str or
            "quota exceeded" in error_str
        )

    def _is_server_error(self, exception: Exception) -> bool:
        """检查是否为5xx服务器错误"""
        error_str = str(exception).lower()
        return (
            "500" in error_str or
            "502" in error_str or
            "503" in error_str or
            "504" in error_str or
            "server error" in error_str or
            "service unavailable" in error_str
        )

    def _is_network_error(self, exception: Exception) -> bool:
        """检查是否为网络错误"""
        error_str = str(exception).lower()
        return (
            "connection" in error_str or
            "timeout" in error_str or
            "network" in error_str or
            "socket" in error_str
        )

    def _calculate_delay(self, attempt: int) -> float:
        """计算重试延迟时间（指数退避 + 抖动）"""
        delay = self.base_delay * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay)

        if self.jitter:
            # 添加±25%的随机抖动
            jitter_range = delay * 0.25
            delay += random.uniform(-jitter_range, jitter_range)

        return max(delay, 0.1)  # 最小延迟0.1秒

    def get_stats(self) -> dict:
        """获取重试统计信息"""
        return self.retry_stats.copy()

    def reset_stats(self):
        """重置统计信息"""
        self.retry_stats = {
            "total_attempts": 0,
            "successful_retries": 0,
            "failed_retries": 0,
            "rate_limit_hits": 0
        }


# 全局重试处理器实例
retry_handler = RetryHandler()
