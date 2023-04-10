from fastchat.conversation import conv_templates, SeparatorStyle
from auto_vicuna.model import generate_stream


def main_loop(model, tokenizer, conv_template, temperature,
              max_new_tokens, plugins, debug, model_path):
    # Chat
    print(model)
    conv = conv_templates[conv_template].copy()
    while True:
        try:
            inp = input("User:")
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break

        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)

        prompt = conv.get_prompt()
        skip_echo_len = len(prompt) + 1
        params = {
            "model": model_path,
            "prompt": prompt,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "stop": conv.sep if conv.sep_style == SeparatorStyle.SINGLE else conv.sep2,
        }
        output_stream = generate_stream(
            model, tokenizer, params, model.device
        )

        pre = 0
        for outputs in output_stream:
            outputs = outputs[skip_echo_len:].strip()
            outputs = outputs.split(" ")
            now = len(outputs)
            if now - 1 > pre:
                print(" ".join(outputs[pre:now-1]), end=" ", flush=True)
                pre = now - 1

        print(" ".join(outputs[pre:now]), flush=True)
        output = " ".join(outputs)
        for plugin in plugins:
            output = plugin.on_response(output)
        conv.append_message(conv.roles[1], output)
        conv.offset += 2

    return conv
