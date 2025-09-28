from abc import ABC, abstractmethod

from .configuration_language_models import (
    BaseLanguageModelConfig,
    LanguageModelConfig,
    OllamaLanguageModelConfig,
    WebApiLanguageModelConfig,
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
        self.config = config
    
    def _load_model(self):
        return self.config.model

    def _generate(self, prompt: str, **kwargs) -> str:
        import ollama

        return ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            think=self.config.think,
        )['message']['content']


class WebApiLanguageModel(BaseLanguageModel):

    config_class = BaseLanguageModelConfig
    name = "webapi"

    def __init__(
        self,
        config: BaseLanguageModelConfig,
    ):
        super().__init__(config=config)
        self.config = config
    
    def _load_model(self):
        return self.config.model

    def _generate(self, prompt: str, **kwargs) -> str:
        import requests

        response = requests.post(
            self.config.api_url,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
            }
        )
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content'].split('</think>')[-1].strip()


def get_language_model(config: LanguageModelConfig) -> BaseLanguageModel:
    if isinstance(config, OllamaLanguageModelConfig):
        return OllamaLanguageModel(config)
    elif isinstance(config, WebApiLanguageModelConfig):
        return WebApiLanguageModel(config)
    else:
        raise ValueError(f"Unknown language model type: {config.name}")