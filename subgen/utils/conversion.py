import os
from typing import Any, Callable, Optional, TypeVar

T = TypeVar('T')


def convert_to_bool(in_bool: Any) -> bool:
    # Convert the input to string and lower case, then check against true values
    return str(in_bool).lower() in ('true', 'on', '1', 'y', 'yes')


def get_env_with_fallback(
    new_name: str,
    old_name: str,
    default_value: Optional[Any] = None,
    convert_func: Optional[Callable[[Any], T]] = None,
) -> Any:
    """
    Get environment variable with backwards compatibility fallback.

    Args:
        new_name: The new standardized environment variable name
        old_name: The legacy environment variable name for backwards compatibility
        default_value: Default value if neither variable is set
        convert_func: Optional function to convert the value (e.g., convert_to_bool, int)

    Returns:
        The environment variable value, converted if convert_func is provided
    """
    # Try new name first, then fall back to old name
    value = os.getenv(new_name) or os.getenv(old_name)

    if value is None:
        value = default_value

    # Apply conversion function if provided
    if convert_func and value is not None:
        return convert_func(value)

    return value
