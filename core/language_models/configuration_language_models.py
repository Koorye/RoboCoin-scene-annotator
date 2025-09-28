import draccus
from dataclasses import dataclass


@dataclass
class LanguageModelConfig(draccus.ChoiceRegistry):
    pass


@LanguageModelConfig.register_subclass("base_language_model")
@dataclass
class BaseLanguageModelConfig(LanguageModelConfig):
    pass


@LanguageModelConfig.register_subclass("ollama")
@dataclass
class OllamaLanguageModelConfig(BaseLanguageModelConfig):
    model: str = "deepseek-r1:8b"
    think: bool = False  # Whether to use "think" mode for more detailed responses


@LanguageModelConfig.register_subclass("webapi")
@dataclass
class WebApiLanguageModelConfig(BaseLanguageModelConfig):
    api_url: str = "https://api.suanli.cn/v1/chat/completions"
    api_key: str = "sk-vPTMzbUSQzG6jJiE0QQNENu2f7zwcD8m6FGr8GDTD4Vqy1Dw"
    model: str = "free:Qwen3-30B-A3B"