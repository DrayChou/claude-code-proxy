import logging
import time
from typing import Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
import json

logger = logging.getLogger(__name__)


@dataclass
class TokenUsageStats:
    """Token使用统计"""
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    requests_count: int = 0
    error_count: int = 0
    rate_limit_hits: int = 0
    compression_count: int = 0
    compression_saved_tokens: int = 0


@dataclass
class PerformanceMetrics:
    """性能指标"""
    request_start_time: float = field(default_factory=time.time)
    response_time: float = 0.0
    token_processing_time: float = 0.0
    compression_time: float = 0.0
    retry_count: int = 0


class SystemMonitor:
    """系统监控和预警系统"""

    def __init__(self):
        self.token_stats = TokenUsageStats()
        self.performance_metrics: Dict[str, PerformanceMetrics] = {}
        self.alerts: list = []
        self.thresholds = {
            "max_tokens_per_minute": 100000,
            "max_requests_per_minute": 100,
            "max_response_time": 30.0,
            "max_error_rate": 0.05,
            "max_compression_ratio": 0.8
        }
        self.start_time = datetime.now()

    def start_request(self, request_id: str) -> None:
        """开始记录请求"""
        self.performance_metrics[request_id] = PerformanceMetrics()

    def end_request(self, request_id: str) -> None:
        """结束记录请求"""
        if request_id in self.performance_metrics:
            metrics = self.performance_metrics[request_id]
            metrics.response_time = time.time() - metrics.request_start_time
            del self.performance_metrics[request_id]

    def record_token_usage(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int
    ) -> None:
        """记录token使用情况"""
        self.token_stats.total_tokens += total_tokens
        self.token_stats.prompt_tokens += prompt_tokens
        self.token_stats.completion_tokens += completion_tokens
        self.token_stats.requests_count += 1

    def record_error(self) -> None:
        """记录错误"""
        self.token_stats.error_count += 1

    def record_rate_limit(self) -> None:
        """记录限流"""
        self.token_stats.rate_limit_hits += 1

    def record_compression(
        self, original_tokens: int, compressed_tokens: int
    ) -> None:
        """记录压缩"""
        self.token_stats.compression_count += 1
        saved = original_tokens - compressed_tokens
        self.token_stats.compression_saved_tokens += saved

    def check_alerts(self) -> list:
        """检查预警条件"""
        alerts = []

        # 检查token使用率
        max_tokens = self.thresholds["max_tokens_per_minute"]
        if self.token_stats.total_tokens > max_tokens:
            alerts.append({
                "type": "token_limit",
                "message": f"Token usage exceeded limit: "
                f"{self.token_stats.total_tokens}",
                "severity": "warning"
            })

        # 检查请求频率
        max_requests = self.thresholds["max_requests_per_minute"]
        if self.token_stats.requests_count > max_requests:
            alerts.append({
                "type": "request_limit",
                "message": f"Request count exceeded limit: "
                f"{self.token_stats.requests_count}",
                "severity": "warning"
            })

        # 检查错误率
        if self.token_stats.requests_count > 0:
            error_rate = (
                self.token_stats.error_count / self.token_stats.requests_count
            )
            if error_rate > self.thresholds["max_error_rate"]:
                alerts.append({
                    "type": "error_rate",
                    "message": f"Error rate too high: {error_rate:.2%}",
                    "severity": "error"
                })

        # 检查限流频率
        if self.token_stats.rate_limit_hits > 10:
            alerts.append({
                "type": "rate_limit",
                "message": f"Too many rate limit hits: "
                f"{self.token_stats.rate_limit_hits}",
                "severity": "warning"
            })

        return alerts

    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        uptime = datetime.now() - self.start_time

        # 计算平均响应时间
        avg_response_time = 0.0
        if self.performance_metrics:
            times = [
                time.time() - m.request_start_time
                for m in self.performance_metrics.values()
            ]
            avg_response_time = sum(times) / len(times) if times else 0.0

        return {
            "uptime": str(uptime),
            "token_stats": {
                "total_tokens": self.token_stats.total_tokens,
                "prompt_tokens": self.token_stats.prompt_tokens,
                "completion_tokens": self.token_stats.completion_tokens,
                "requests_count": self.token_stats.requests_count,
                "error_count": self.token_stats.error_count,
                "error_rate": (
                    self.token_stats.error_count
                    / self.token_stats.requests_count
                    if self.token_stats.requests_count > 0 else 0.0
                ),
                "rate_limit_hits": self.token_stats.rate_limit_hits,
                "compression_stats": {
                    "count": self.token_stats.compression_count,
                    "saved_tokens": self.token_stats.compression_saved_tokens,
                    "efficiency": (
                        self.token_stats.compression_saved_tokens
                        / self.token_stats.total_tokens
                        if self.token_stats.total_tokens > 0 else 0.0
                    )
                }
            },
            "performance": {
                "active_requests": len(self.performance_metrics),
                "avg_response_time": avg_response_time
            },
            "alerts": self.check_alerts()
        }

    def log_status(self) -> None:
        """记录系统状态"""
        status = self.get_system_status()
        logger.info(
            f"System status: "
            f"{json.dumps(status, indent=2, ensure_ascii=False)}"
        )

    def reset_stats(self) -> None:
        """重置统计信息"""
        self.token_stats = TokenUsageStats()
        self.performance_metrics.clear()
        self.alerts.clear()
        self.start_time = datetime.now()


# 全局监控实例
system_monitor = SystemMonitor()
