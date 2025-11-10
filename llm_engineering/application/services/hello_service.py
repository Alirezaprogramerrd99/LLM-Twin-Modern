# llm_engineering/application/services/hello_service.py
from dataclasses import dataclass
from loguru import logger
from llm_engineering.application.settings import Settings

@dataclass
class HelloService:
    settings: Settings

    def greet(self, name: str | None = None) -> dict:
        who = (name or "friend").strip()
        level = "DEBUG" if self.settings.debug else "INFO"
        logger.log(level, "Greeting requested: name='{}', env='{}'", who, self.settings.app_env)
        return {
            "message": f"Hello, {who}!",
            "environment": self.settings.app_env,
            "debug": self.settings.debug,
        }
