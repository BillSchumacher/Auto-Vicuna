"""Chat loop for auto_vicuna."""
from ast import Module
from typing import List, Optional

from fastchat.conversation import Conversation, SeparatorStyle
from fastchat.serve.cli import SimpleChatIO
from fastchat.serve.inference import ChatIO, generate_stream
from fastchat.serve.serve_chatglm import chatglm_generate_stream


def chat_one_shot(
    model,
    tokenizer,
    model_name: str,
    device: str,
    conversation: Conversation,
    message: str,
    temperature: float = 0.7,
    max_new_tokens: int = 512,
    plugins: Optional[List[Module]] = None,
    chatio: ChatIO = SimpleChatIO(),
    debug: bool = False,
) -> Optional[str]:
    """Chat one shot, returns a single message.

    Unless no input was provided, in which case it returns None.

    Args:
        model: Model.
        tokenizer: Tokenizer.
        model_name (str): Model name.
        device (str): Device.
        conversation (Conversation): Conversation.
        message (str): Message.
        temperature (float): Temperature.
        max_new_tokens (int): Max new tokens.
        plugins: Plugins.
        chatio (ChatIO): ChatIO.
        debug (bool): Debug.

    Returns:
        Optional[str]: Output.
    """
    if not message:
        return None
    if plugins is None:
        plugins = []

    conversation.append_message(conversation.roles[0], message)
    conversation.append_message(conversation.roles[1], None)

    return chat_output(
        model,
        tokenizer,
        model_name,
        device,
        conversation,
        temperature,
        max_new_tokens,
        plugins,
        chatio,
        debug,
    )


def chat_loop(
    model,
    tokenizer,
    model_name: str,
    device: str,
    conversation: Conversation,
    temperature: float = 0.7,
    max_new_tokens: int = 512,
    plugins: Optional[List[Module]] = None,
    chatio: ChatIO = SimpleChatIO(),
    debug: bool = False,
) -> None:
    """Infinite chat loop.

    Args:
        model: Model.
        tokenizer: Tokenizer.
        model_name (str): Model name.
        device (str): Device.
        conversation (Conversation): Conversation.
        temperature (float): Temperature.
        max_new_tokens (int): Max new tokens.
        plugins: Plugins.
        chatio (ChatIO): ChatIO.
        debug (bool): Debug.

    Returns:
        None"""
    if plugins is None:
        plugins = []

    while True:
        try:
            inp = chatio.prompt_for_input(conversation.roles[0])
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break

        conversation.append_message(conversation.roles[0], inp)
        conversation.append_message(conversation.roles[1], None)

        chat_output(
            model,
            tokenizer,
            model_name,
            device,
            conversation,
            temperature,
            max_new_tokens,
            plugins,
            chatio,
            debug,
        )


def chat_output(
    model,
    tokenizer,
    model_name: str,
    device: str,
    conversation: Conversation,
    temperature: float = 0.7,
    max_new_tokens: int = 512,
    plugins: Optional[List[Module]] = None,
    chatio: ChatIO = SimpleChatIO(),
    debug: bool = False,
):
    is_chatglm = "chatglm" in str(type(model)).lower()
    if is_chatglm:
        prompt = conversation.messages[conversation.offset :]
        generate_stream_func = chatglm_generate_stream
        skip_echo_len = len(conversation.messages[-2][1]) + 1
    else:
        generate_stream_func = generate_stream
        prompt = conversation.get_prompt()
        skip_echo_len = len(prompt.replace("</s>", " ")) + 1

    params = {
        "model": model_name,
        "prompt": prompt,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "stop": conversation.sep
        if conversation.sep_style == SeparatorStyle.SINGLE
        else conversation.sep2,
    }

    chatio.prompt_for_output(conversation.roles[1])
    output_stream = generate_stream_func(model, tokenizer, params, device)
    outputs = chatio.stream_output(output_stream, skip_echo_len)
    # NOTE: strip is important to align with the training data.
    output = outputs.strip()

    for plugin in plugins:
        output = plugin.on_response(output)
    if debug:
        print("\n", {"prompt": prompt, "outputs": outputs}, "\n")
    return output
