from typing import Dict, Type, Any, Optional
from dataclasses import dataclass


@dataclass
class GeneratorInfo:
    name: str
    generator_class: Type[Any]
    config_class: Type[Any]
    description: str
    default_config: Dict[str, Any]


class GeneratorRegistry:
    _generators: Dict[str, GeneratorInfo] = {}

    @classmethod
    def register(
        cls,
        name: str,
        description: str = "",
        default_config: Optional[Dict[str, Any]] = None,
    ):
        def decorator(generator_class):
            if not hasattr(generator_class, "Config"):
                raise ValueError(
                    f"Generator {generator_class.__name__} must have a Config inner class"
                )

            config_class = generator_class.Config
            # Check inheritance at runtime
            from datagen.core.generators import BaseGeneratorConfig

            if not issubclass(config_class, BaseGeneratorConfig):
                raise ValueError(f"Config class must inherit from BaseGeneratorConfig")

            cls._generators[name] = GeneratorInfo(
                name=name,
                generator_class=generator_class,
                config_class=config_class,
                description=description,
                default_config=default_config or {},
            )
            return generator_class

        return decorator

    @classmethod
    def get_generator_info(cls, name: str) -> GeneratorInfo:
        if name not in cls._generators:
            raise ValueError(
                f"Unknown generator: {name}. Available: {list(cls._generators.keys())}"
            )
        return cls._generators[name]

    @classmethod
    def list_generators(cls) -> Dict[str, GeneratorInfo]:
        return cls._generators.copy()

    @classmethod
    def create_generator_config(
        cls, name: str, config_dict: dict, sources_config: dict, models_config: dict
    ):
        generator_info = cls.get_generator_info(name)
        return generator_info.config_class.from_config(
            generator_config=config_dict,
            sources_config=sources_config,
            models_config=models_config,
        )
