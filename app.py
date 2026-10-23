# 1. Install required libraries (run in Colab / terminal)
# pip install transformers accelerate torch torchvision gradio

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import gradio as gr

# 2. Load open-source AI model (Qwen2.5-7B-Instruct)
print("🤖 Loading AI model...")
model_id = "Qwen/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load model in native bfloat16 precision
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
print("✨ Model load complete!")

# 3. Chat function
def chat_with_open_ai(message, history):
    messages = []
    for user_msg, bot_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": bot_msg})

    messages.append({"role": "user", "content": message})

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    outputs = pipe(
        prompt,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

    generated_text = outputs[0]["generated_text"]
    response = generated_text[len(prompt):].strip()

    return response

# 4. Build Gradio UI
with gr.Blocks(theme=gr.themes.Monochrome()) as demo:
    gr.Markdown("# 🔥 My Open-Source AI Platform")
    gr.Markdown("Running open-source AI (Qwen2.5) directly. Chat freely without limits!")

    gr.ChatInterface(
        fn=chat_with_open_ai,
        textbox=gr.Textbox(placeholder="Type a message...", container=False, scale=7)
    )

# 5. Launch
demo.launch(
    share=True,
    debug=True,
    auth=("your_username", "your_password")
)
