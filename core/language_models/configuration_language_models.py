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