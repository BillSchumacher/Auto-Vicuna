"""Conversation utilities."""
from typing import List, Optional, Tuple

from fastchat.conversation import Conversation, SeparatorStyle, conv_templates


def make_conversation(
    system: str,
    roles: List[str],
    messages: List[Tuple[str, str]],
    seperator_1: str = "###",
    seperator_2: Optional[str] = None,
    seperator_style: SeparatorStyle = SeparatorStyle.SINGLE,
) -> Conversation:
    """Make a conversation.

    Args:
        system (str): System prompt, this is where you should define how you want
          the model to respond.
        roles (List[str]): List of roles. System, User, etc.
        messages (List[Tuple[str, str]]): List of messages or context.
        seperator_1 (str, optional): Seperator 1. Defaults to '###'.
        seperator_2 (Optional[str], optional): Seperator 2. Defaults to None.
        seperator_style (SeparatorStyle, optional): Seperator style.
            Defaults to SeparatorStyle.SINGLE.

    Returns:
        Conversation: Conversation object.
    """
    return Conversation(
        system=system,
        roles=roles,
        messages=messages,  # type: ignore
        offset=len(messages),
        sep_style=seperator_style,
        sep=seperator_1,
        sep2=seperator_2,  # type: ignore
    )


def conversation_from_template(template_name: str) -> Conversation:
    """Get a conversation from a template.

    Args:
        template_name (str): Template name.

    Returns:
        Conversation: Conversation object.

    Raises:
        ValueError: If template not found.
    """
    try:
        return conv_templates[template_name].copy()
    except KeyError as e:
        raise ValueError(f"Template {template_name} not found.") from e
