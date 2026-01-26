"""
Centralized model configuration with task-specific model selection.

This module provides:
1. Task-specific model selection (resume parsing, job extraction, matching, etc.)
2. Explicit JSON mode control
3. Locked critical paths to deterministic models
4. Runtime assertions for model capabilities
"""
import os
import logging
from typing import Optional, Literal
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

# Task types for model selection
TaskType = Literal[
    "resume_parsing",
    "job_extraction", 
    "matching",
    "sponsorship",
    "summarization",
    "ocr_fallback"
]

# Models that don't support temperature customization
MODELS_WITHOUT_TEMPERATURE = [
    "o1", "o1-mini", "o1-preview", "o1-2024",
    "gpt-5-mini", "gpt-5",
]

# Models that support JSON mode
MODELS_WITH_JSON_MODE = [
    "gpt-4", "gpt-4o", "gpt-4o-mini", "gpt-4-turbo",
    "o1", "o1-mini", "o1-preview",
]

# Locked models for critical paths (cannot be overridden)
LOCKED_MODELS = {
    "resume_parsing": "gpt-4o-mini",  # Fast, deterministic
    "ocr_fallback": "gpt-4o-mini",    # Fast, deterministic
    "matching": "gpt-4o",              # High accuracy required
    "job_extraction": "gpt-4o",       # High accuracy required
}

# Default models per task (can be overridden via env vars)
DEFAULT_MODELS = {
    "resume_parsing": "gpt-4o-mini",
    "job_extraction": "gpt-4o",
    "matching": "gpt-4o",
    "sponsorship": "gpt-4o-mini",
    "summarization": "gpt-4o-mini",
    "ocr_fallback": "gpt-4o-mini",
}

# Default temperatures per task
DEFAULT_TEMPERATURES = {
    "resume_parsing": 0,
    "job_extraction": 0,
    "matching": 0,
    "sponsorship": 0,
    "summarization": 0.3,
    "ocr_fallback": 0,
}


def select_model_for_task(
    task_type: TaskType,
    allow_override: bool = False
) -> str:
    """
    Select model for a specific task.
    
    Args:
        task_type: Type of task (resume_parsing, job_extraction, etc.)
        allow_override: If True, allows env var override even for locked models (dev/test only)
    
    Returns:
        Model name to use for this task
    """
    # Check if model is locked
    if task_type in LOCKED_MODELS and not allow_override:
        locked_model = LOCKED_MODELS[task_type]
        logger.debug(f"Using locked model for {task_type}: {locked_model}")
        return locked_model
    
    # Check for task-specific env var override
    env_var_name = f"OPENAI_MODEL_{task_type.upper()}"
    env_model = os.getenv(env_var_name)
    
    if env_model:
        logger.info(f"Using env override for {task_type}: {env_model} (from {env_var_name})")
        return env_model
    
    # Use default for task
    default_model = DEFAULT_MODELS.get(task_type, "gpt-4o")
    logger.debug(f"Using default model for {task_type}: {default_model}")
    return default_model


def supports_temperature(model_name: str) -> bool:
    """Check if model supports temperature customization."""
    model_lower = model_name.lower()
    return not any(no_temp in model_lower for no_temp in MODELS_WITHOUT_TEMPERATURE)


def supports_json_mode(model_name: str) -> bool:
    """Check if model supports JSON mode."""
    model_lower = model_name.lower()
    return any(json_model in model_lower for json_model in MODELS_WITH_JSON_MODE)


def build_chat_model(
    model_name: str,
    temperature: float = 0,
    json_mode: bool = False,
    task_type: Optional[TaskType] = None
) -> ChatOpenAI:
    """
    Build ChatOpenAI model with explicit configuration.
    
    Args:
        model_name: Name of the model
        temperature: Temperature setting (only used if model supports it)
        json_mode: Whether to enable JSON response format
        task_type: Task type for logging/validation
    
    Returns:
        ChatOpenAI model instance
    
    Raises:
        ValueError: If json_mode is requested but model doesn't support it
    """
    # Validate JSON mode support
    if json_mode and not supports_json_mode(model_name):
        raise ValueError(
            f"Model {model_name} does not support JSON mode. "
            f"Supported models: {', '.join(MODELS_WITH_JSON_MODE)}"
        )
    
    # Build model kwargs
    model_kwargs = {"model": model_name}
    
    # Set temperature if supported
    if supports_temperature(model_name):
        model_kwargs["temperature"] = temperature
        if task_type:
            logger.debug(f"{task_type}: Using temperature={temperature} for {model_name}")
    else:
        if task_type:
            logger.debug(f"{task_type}: Model {model_name} does not support temperature, using default")
    
    # Set JSON mode if requested
    if json_mode:
        model_kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
        if task_type:
            logger.debug(f"{task_type}: JSON mode enabled for {model_name}")
    
    return ChatOpenAI(**model_kwargs)


def get_model_for_task(
    task_type: TaskType,
    temperature: Optional[float] = None,
    json_mode: bool = False,
    allow_override: bool = False
) -> ChatOpenAI:
    """
    Get model configuration for a specific task.
    
    This is the main entry point for getting models - it combines:
    - Task-specific model selection
    - Task-specific temperature defaults
    - Explicit JSON mode control
    
    Args:
        task_type: Type of task
        temperature: Temperature override (uses task default if None)
        json_mode: Whether to enable JSON mode
        allow_override: Allow override of locked models (dev/test only)
    
    Returns:
        ChatOpenAI model instance configured for the task
    """
    # Select model for task
    model_name = select_model_for_task(task_type, allow_override=allow_override)
    
    # Use task-specific temperature if not provided
    if temperature is None:
        temperature = DEFAULT_TEMPERATURES.get(task_type, 0)
    
    # Build and return model
    return build_chat_model(
        model_name=model_name,
        temperature=temperature,
        json_mode=json_mode,
        task_type=task_type
    )


# Backward compatibility: Keep old get_model_config function
def get_model_config(
    model_name: str,
    default_temperature: float = 0,
    json_mode: bool = False
) -> ChatOpenAI:
    """
    Legacy function for backward compatibility.
    
    Use get_model_for_task() for new code.
    
    Args:
        model_name: Name of the model
        default_temperature: Temperature setting
        json_mode: Whether to enable JSON mode
    
    Returns:
        ChatOpenAI model instance
    """
    return build_chat_model(
        model_name=model_name,
        temperature=default_temperature,
        json_mode=json_mode
    )


# Runtime assertions for model capabilities
def assert_model_supports_json(model_name: str, task_type: Optional[TaskType] = None):
    """
    Assert that model supports JSON mode. Raises ValueError if not.
    
    Args:
        model_name: Name of the model
        task_type: Task type for error message context
    
    Raises:
        ValueError: If model doesn't support JSON mode
    """
    if not supports_json_mode(model_name):
        task_context = f" for {task_type}" if task_type else ""
        raise ValueError(
            f"Model {model_name} does not support JSON mode{task_context}. "
            f"Supported models: {', '.join(MODELS_WITH_JSON_MODE)}"
        )


def assert_model_supports_temperature(model_name: str, temperature: float, task_type: Optional[TaskType] = None):
    """
    Assert that model supports temperature customization. Logs warning if not.
    
    Args:
        model_name: Name of the model
        temperature: Desired temperature
        task_type: Task type for logging context
    """
    if not supports_temperature(model_name) and temperature != 0:
        task_context = f" for {task_type}" if task_type else ""
        logger.warning(
            f"Model {model_name} does not support temperature customization{task_context}. "
            f"Requested temperature={temperature} will be ignored."
        )


def validate_model_for_task(model_name: str, task_type: TaskType, json_mode: bool = False):
    """
    Validate that model is appropriate for the task.
    
    Args:
        model_name: Name of the model
        task_type: Type of task
        json_mode: Whether JSON mode is required
    
    Raises:
        ValueError: If model doesn't meet task requirements
    """
    if json_mode:
        assert_model_supports_json(model_name, task_type)
    
    # Check if model is locked for this task
    if task_type in LOCKED_MODELS:
        locked_model = LOCKED_MODELS[task_type]
        if model_name != locked_model:
            logger.warning(
                f"Model {model_name} differs from locked model {locked_model} for {task_type}. "
                f"Using {model_name} but locked model is recommended for consistency."
            )
