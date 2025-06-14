import gradio as gr
from langchain.chains import ConversationChain
from typing import Optional, Tuple
from chains.slot_memory import SlotMemory
from chains.prompt import CHAT_PROMPT
from configs.params import ModelParams
from local_model import local_load_model
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer

model_config = ModelParams()
tokenizer = AutoTokenizer.from_pretrained(
    model_config.model_path,
    trust_remote_code=True,
    padding_side="left",
    pad_token="<|endoftext|>",
    eos_token="<|endoftext|>",
)



chain: ConversationChain


def initial_chain():
    llm = local_load_model()
    memory = SlotMemory(llm=llm)
    global chain
    chain = ConversationChain(llm=llm, memory=memory, prompt=CHAT_PROMPT)


def clear_session():
    initial_chain()
    return [], []


def slot_format(slot_dict):
    result = f"name: {slot_dict['name']}\norigin: {slot_dict['origin']}\ndestination: {slot_dict['destination']}\ndeparture_time: {slot_dict['departure_time']}\n"
    return result


def predict(command, history: Optional[list]):
    history = history or []
    print(f"User input: {command}")

    # Format the prompt with thinking mode OFF
    formatted_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": command}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False  # 👈 This turns off thinking mode
    )
    print("Formatted prompt:\n", formatted_prompt)

    result = chain.invoke({"input": formatted_prompt})
    print(f"Raw chain response: {result}")
    response_text = result.get("response", "No response generated.")
    print(f"Extracted text: {response_text}")
    current_slot = chain.memory.current_slots

    history.append({"role": "user", "content": command})
    history.append({"role": "assistant", "content": response_text})
    return history, history, '', slot_format(current_slot)



if __name__ == "__main__":
    title = """
    # Dialogue Slot Filling Demo
    """
    with gr.Blocks() as demo:
        gr.Markdown(title)

        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.JSON(value=[])
                user_input = gr.Textbox(show_label=False, placeholder="Input...", container=False)
                with gr.Row():
                    submitBtn = gr.Button("🚀Submit", variant="primary")
                    emptyBtn = gr.Button("🧹Clear History")
            slot_show = gr.Textbox(label="current_slot", lines=20, interactive=False, scale=1)

        initial_chain()
        state = gr.State([])

        submitBtn.click(fn=predict, inputs=[user_input, state], outputs=[chatbot, state, user_input, slot_show])
        emptyBtn.click(fn=clear_session, inputs=[], outputs=[chatbot, state])

    demo.queue().launch(share=False, inbrowser=True, server_name="0.0.0.0", server_port=8000)
