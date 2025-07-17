#!/usr/bin/env python3
"""
测试token压缩策略的脚本
"""
import sys
import os
import logging
# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.core.token_manager import TokenManager
# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
def create_test_messages():
    """创建测试消息列表"""
    messages = [
        {"role": "system", "content": "你是一个有用的AI助手。"},
        {"role": "user", "content": "请帮我写一个Python函数来计算斐波那契数列。" * 20},
        {"role": "assistant", "content": "好的，我来帮你写一个斐波那契数列的函数。" * 30},
        {"role": "user", "content": "能否优化一下这个函数的性能？" * 15},
        {"role": "assistant", "content": "当然可以，我们可以使用动态规划来优化。" * 25},
        {"role": "user", "content": "太好了，能否再解释一下时间复杂度？" * 10},
        {"role": "assistant", "content": "时间复杂度从O(2^n)优化到了O(n)。" * 20},
        {"role": "user", "content": "非常感谢你的详细解释！"},  # 最新的用户消息
    ]
    return messages
def test_compression_strategy():
    """测试压缩策略"""
    print("🔧 开始测试智能压缩策略...")
    
    token_manager = TokenManager()
    model = "gpt-4"
    max_tokens = 1000  # 设置较小的限制来触发压缩
    
    # 创建测试消息
    test_messages = create_test_messages()
    print(f"📝 创建了 {len(test_messages)} 条测试消息")
    
    # 计算原始token数量
    original_tokens = token_manager.count_message_tokens(test_messages, model)
    print(f"📊 原始消息token数量: {original_tokens}")
    
    if original_tokens <= max_tokens:
        print("⚠️  消息未超过限制，增加内容...")
        # 如果没超过限制，增加更多内容
        for i in range(len(test_messages)):
            if test_messages[i]["role"] != "system":
                test_messages[i]["content"] = test_messages[i]["content"] * 5
        original_tokens = token_manager.count_message_tokens(test_messages, model)
        print(f"📊 增加内容后token数量: {original_tokens}")
    
    print(f"🎯 Token限制: {max_tokens}")
    print(f"📈 超出比例: {((original_tokens - max_tokens) / max_tokens * 100):.1f}%")
    
    # 测试原始截断策略
    print("\n🔄 测试原始截断策略...")
    truncated_messages = token_manager.truncate_messages(test_messages.copy(), max_tokens, model)
    truncated_tokens = token_manager.count_message_tokens(truncated_messages, model)
    print(f"✂️  原始策略结果: {truncated_tokens} tokens, {len(truncated_messages)} 条消息")
    
    # 测试智能压缩策略
    print("\n🧠 测试智能压缩策略...")
    smart_messages = token_manager.truncate_messages_smart(test_messages.copy(), max_tokens, model)
    smart_tokens = token_manager.count_message_tokens(smart_messages, model)
    print(f"🎯 智能策略结果: {smart_tokens} tokens, {len(smart_messages)} 条消息")
    
    # 分析结果
    print("\n📋 结果分析:")
    print(f"原始消息: {original_tokens} tokens, {len(test_messages)} 条")
    print(f"原始截断: {truncated_tokens} tokens, {len(truncated_messages)} 条")
    print(f"智能压缩: {smart_tokens} tokens, {len(smart_messages)} 条")
    
    # 验证最新消息是否保留
    if smart_messages:
        last_message = smart_messages[-1]
        original_last = test_messages[-1]
        if last_message["content"] == original_last["content"]:
            print("✅ 最新用户消息完整保留")
        else:
            print("❌ 最新用户消息被修改")
    
    # 检查是否超过限制
    if smart_tokens <= max_tokens:
        print("✅ 智能压缩策略成功控制在token限制内")
    else:
        print(f"❌ 智能压缩策略仍超出限制: {smart_tokens} > {max_tokens}")
    
    return {
        'original': (original_tokens, len(test_messages)),
        'truncated': (truncated_tokens, len(truncated_messages)),
        'smart': (smart_tokens, len(smart_messages)),
        'success': smart_tokens <= max_tokens
    }
if __name__ == "__main__":
    try:
        result = test_compression_strategy()
        print(f"\n🎉 测试完成! 智能压缩策略{'成功' if result['success'] else '失败'}")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()