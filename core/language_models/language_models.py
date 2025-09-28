from abc import ABC, abstractmethod

from .configuration_language_models import (
    BaseLanguageModelConfig,
    LanguageModelConfig,
    OllamaLanguageModelConfig,
)


class BaseLanguageModel(ABC):

    config_class = BaseLanguageModelConfig
    name = "base_language_model"

    def __init__(
        self,
        config: BaseLanguageModelConfig,
    ):
        self.config = config
        self.model = self._load_model()
    
    @abstractmethod
    def _load_model(self):
        pass

    @abstractmethod
    def _generate(self, prompt: str, **kwargs) -> str:
        pass

    def generate(self, prompt: str, **kwargs) -> str:
        return self._generate(prompt, **kwargs)
    

class OllamaLanguageModel(BaseLanguageModel):

    config_class = OllamaLanguageModelConfig
    name = "ollama"

    def __init__(
        self,
        config: BaseLanguageModelConfig,
    ):
        super().__init__(config=config)
        import ollama
        self.ollama = ollama
    
    def _load_model(self):
        return self.config.model

    def _generate(self, prompt: str, **kwargs) -> str:
        import ollama

        return ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            think=self.config.think,
        )['message']['content']


def get_language_model(config: LanguageModelConfig) -> BaseLanguageModel:
    if isinstance(config, OllamaLanguageModelConfig):
        return OllamaLanguageModel(config)
    else:
        raise ValueError(f"Unknown language model type: {config.name}")