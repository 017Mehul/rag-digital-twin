"""
Configuration utilities for the RAG Digital Twin system.
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any
from src.models.rag_config import RAGConfig
from src.exceptions import ConfigurationError, ErrorCode


def load_config(config_path: str) -> RAGConfig:
    """
    Load configuration from a YAML or JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        RAGConfig instance
        
    Raises:
        ConfigurationError: If configuration cannot be loaded or is invalid
    """
    try:
        config_file = Path(config_path)
        
        if not config_file.exists():
            raise ConfigurationError(
                f"Configuration file not found: {config_path}",
                ErrorCode.CONFIG_MISSING_REQUIRED,
                "config_path"
            )
        
        # Load configuration based on file extension
        with open(config_file, 'r') as f:
            if config_file.suffix.lower() in ['.yaml', '.yml']:
                config_data = yaml.safe_load(f)
            elif config_file.suffix.lower() == '.json':
                config_data = json.load(f)
            else:
                raise ConfigurationError(
                    f"Unsupported configuration file format: {config_file.suffix}",
                    ErrorCode.CONFIG_INVALID,
                    "file_format"
                )
        
        # Flatten nested configuration structure
        flattened_config = _flatten_config(config_data)
        
        # Substitute environment variables
        resolved_config = _resolve_environment_variables(flattened_config)
        
        # Create RAGConfig instance
        return RAGConfig.from_dict(resolved_config)
        
    except ConfigurationError:
        raise
    except yaml.YAMLError as e:
        raise ConfigurationError(
            f"Invalid YAML configuration: {e}",
            ErrorCode.CONFIG_INVALID,
            cause=e
        )
    except json.JSONDecodeError as e:
        raise ConfigurationError(
            f"Invalid JSON configuration: {e}",
            ErrorCode.CONFIG_INVALID,
            cause=e
        )
    except Exception as e:
        raise ConfigurationError(
            f"Failed to load configuration: {e}",
            ErrorCode.CONFIG_INVALID,
            cause=e
        )


def validate_config(config: RAGConfig) -> bool:
    """
    Validate a RAGConfig instance.
    
    Args:
        config: RAGConfig to validate
        
    Returns:
        True if valid
        
    Raises:
        ConfigurationError: If configuration is invalid
    """
    try:
        return config.validate()
    except ValueError as e:
        raise ConfigurationError(
            f"Configuration validation failed: {e}",
            ErrorCode.CONFIG_VALIDATION_FAILED,
            cause=e
        )


def _flatten_config(config_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten nested configuration structure to match RAGConfig fields.
    
    Args:
        config_data: Nested configuration dictionary
        
    Returns:
        Flattened configuration dictionary
    """
    flattened = {}
    
    # Map nested structure to flat structure
    mapping = {
        ('embedding', 'provider'): 'embedding_provider',
        ('embedding', 'model'): 'embedding_model',
        ('embedding', 'provider_config'): 'embedding_provider_config',
        ('embedding', 'fallbacks'): 'embedding_fallbacks',
        ('llm', 'provider'): 'llm_provider',
        ('llm', 'model'): 'llm_model',
        ('llm', 'provider_config'): 'llm_provider_config',
        ('llm', 'fallbacks'): 'llm_fallbacks',
        ('document_processing', 'chunk_size'): 'chunk_size',
        ('document_processing', 'chunk_overlap'): 'chunk_overlap',
        ('retrieval', 'top_k_results'): 'top_k_results',
        ('retrieval', 'similarity_threshold'): 'similarity_threshold',
        ('retrieval', 'max_context_length'): 'max_context_length',
        ('response', 'max_tokens'): 'max_response_tokens',
        ('response', 'temperature'): 'temperature',
        ('system', 'batch_size'): 'batch_size',
        ('system', 'max_retries'): 'max_retries',
        ('system', 'timeout_seconds'): 'timeout_seconds',
        ('paths', 'data_directory'): 'data_directory',
        ('paths', 'embeddings_directory'): 'embeddings_directory',
        ('paths', 'logs_directory'): 'logs_directory',
    }
    
    # Apply mappings
    for (section, key), target_key in mapping.items():
        if section in config_data and key in config_data[section]:
            flattened[target_key] = config_data[section][key]
    
    # Handle direct mappings (for backward compatibility)
    direct_keys = [
        'embedding_provider', 'embedding_model', 'embedding_provider_config', 'embedding_fallbacks',
        'llm_provider', 'llm_model', 'llm_provider_config', 'llm_fallbacks',
        'chunk_size', 'chunk_overlap', 'top_k_results', 'similarity_threshold',
        'max_context_length', 'max_response_tokens', 'temperature',
        'batch_size', 'max_retries', 'timeout_seconds',
        'data_directory', 'embeddings_directory', 'logs_directory'
    ]
    
    for key in direct_keys:
        if key in config_data:
            flattened[key] = config_data[key]
    
    return flattened


def _resolve_environment_variables(config_data: Any) -> Any:
    """
    Resolve environment variable references in configuration values.
    
    Args:
        config_data: Configuration dictionary
        
    Returns:
        Configuration with resolved environment variables
    """
    if isinstance(config_data, dict):
        return {key: _resolve_environment_variables(value) for key, value in config_data.items()}

    if isinstance(config_data, list):
        return [_resolve_environment_variables(value) for value in config_data]

    if isinstance(config_data, str) and config_data.startswith('${') and config_data.endswith('}'):
        env_var = config_data[2:-1]
        resolved_value = os.getenv(env_var)

        if resolved_value is None:
            raise ConfigurationError(
                f"Environment variable not found: {env_var}",
                ErrorCode.CONFIG_MISSING_REQUIRED,
                env_var
            )

        return resolved_value

    return config_data


def get_default_config_path() -> str:
    """
    Get the default configuration file path.
    
    Returns:
        Path to default configuration file
    """
    return "config/rag_config.yaml"


def create_default_config(output_path: str) -> None:
    """
    Create a default configuration file.
    
    Args:
        output_path: Path where to create the configuration file
    """
    config_file = Path(output_path)
    config_file.parent.mkdir(parents=True, exist_ok=True)

    default_config = {
        "embedding": {
            "provider": "openai",
            "model": "text-embedding-3-small",
            "provider_config": {
                "api_key": "${OPENAI_API_KEY}",
                "dimension": 1536,
            },
            "fallbacks": [
                {
                    "provider": "huggingface",
                    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                    "config": {
                        "dimension": 1536,
                    },
                }
            ],
        },
        "llm": {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "provider_config": {
                "api_key": "${OPENAI_API_KEY}",
            },
            "fallbacks": [
                {
                    "provider": "huggingface",
                    "model_name": "distilgpt2",
                    "config": {},
                }
            ],
        },
        "document_processing": {
            "chunk_size": 1000,
            "chunk_overlap": 200,
        },
        "retrieval": {
            "top_k_results": 5,
            "similarity_threshold": 0.7,
            "max_context_length": 4000,
        },
        "response": {
            "max_tokens": 500,
            "temperature": 0.1,
        },
        "system": {
            "batch_size": 10,
            "max_retries": 3,
            "timeout_seconds": 30,
        },
        "paths": {
            "data_directory": "data",
            "embeddings_directory": "embeddings",
            "logs_directory": "logs",
        },
    }

    with open(config_file, "w", encoding="utf-8") as handle:
        if config_file.suffix.lower() in [".yaml", ".yml"]:
            yaml.safe_dump(default_config, handle, sort_keys=False)
        elif config_file.suffix.lower() == ".json":
            json.dump(default_config, handle, indent=2)
        else:
            raise ConfigurationError(
                f"Unsupported configuration file format: {config_file.suffix}",
                ErrorCode.CONFIG_INVALID,
                "file_format"
            )
