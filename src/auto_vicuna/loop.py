"""Chat loop for auto_vicuna."""
from fastchat.conversation import conv_templates, SeparatorStyle
from fastchat.serve.serve_chatglm import chatglm_generate_stream
from fastchat.serve.inference import generate_stream, ChatIO


def chat_loop(
    model, tokenizer, model_name: str, device: str,
    conv_template: str, temperature: float, max_new_tokens: int,
    plugins, chatio: ChatIO, debug: bool
):

    is_chatglm = "chatglm" in str(type(model)).lower()

    # Chat
    conv = conv_templates[conv_template].copy()
    while True:
        try:
            inp = chatio.prompt_for_input(conv.roles[0])
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break

        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)

        if is_chatglm:
            prompt = conv.messages[conv.offset:]
            generate_stream_func = chatglm_generate_stream
            skip_echo_len = len(conv.messages[-2][1]) + 1
        else:
            generate_stream_func = generate_stream
            prompt = conv.get_prompt()
            skip_echo_len = len(prompt.replace("</s>", " ")) + 1

        params = {
            "model": model_name,
            "prompt": prompt,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "stop": conv.sep if conv.sep_style == SeparatorStyle.SINGLE else conv.sep2,
        }

        chatio.prompt_for_output(conv.roles[1])
        output_stream = generate_stream_func(model, tokenizer, params, device)
        outputs = chatio.stream_output(output_stream, skip_echo_len)
        # NOTE: strip is important to align with the training data.
        conv.messages[-1][-1] = output = outputs.strip()

        for plugin in plugins:
            output = plugin.on_response(output)
        if debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")
