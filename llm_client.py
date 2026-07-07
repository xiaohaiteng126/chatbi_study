"""
LLM 客户端模块

封装 OpenAI 兼容的 LLM 调用，用于 Text2SQL 场景。
"""

from openai import OpenAI
from config import LLM_CONFIG


class LLMClient:
    """LLM 客户端，通过 OpenAI 兼容协议调用"""

    def __init__(self):
        self.client = OpenAI(
            api_key=LLM_CONFIG["api_key"],
            base_url=LLM_CONFIG["base_url"]
        )
        self.model = LLM_CONFIG["model"]
        self.temperature = LLM_CONFIG["temperature"]
        self.max_tokens = LLM_CONFIG["max_tokens"]

    def generate_sql(self, system_msg: str, prompt: str) -> str:
        """
        调用 LLM 生成 SQL

        Args:
            system_msg: 系统消息
            prompt: 用户提示词

        Returns:
            生成的 SQL 字符串
        """
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
