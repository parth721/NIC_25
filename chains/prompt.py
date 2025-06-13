from langchain.prompts.prompt import PromptTemplate

# =============== CHAT PROMPT ===============
# Format: Qwen ChatML-style

_DEFAULT_TEMPLATE = """<|im_start|>system
You are an AI flight booking assistant. Follow these rules:
1. Collect missing information: name, origin, destination, departure_time
2. When all slots are filled, output "Booking Successful" with details:
   name: [name]
   origin: [origin]
   destination: [destination]
   departure time: [departure time]
3. Never repeat the human's exact words
4. Never output Current Slots directly
5. If information is missing, ask specific questions:
   - If name is null: "May I have your name?"
   - If origin is null: "Where will you be departing from?"
   - If destination is null: "What is your destination city?"
   - If departure_time is null: "When would you like to depart?"

Current Slots:
{slots}
Information Check: {check}<|im_end|>
<|im_start|>user
{history}
{input}<|im_end|>
<|im_start|>assistant
"""

CHAT_PROMPT = PromptTemplate(
    input_variables=["history", "input", "slots", "check"],
    template=_DEFAULT_TEMPLATE
)

# =============== SLOT EXTRACTION PROMPT ===============
# Format: Qwen ChatML-style with JSON output block safely escaped

_DEFAULT_SLOT_EXTRACTION_TEMPLATE = """<|im_start|>system
You are an information extraction AI. Strictly follow these rules:
1. Extract flight booking details from the conversation
2. Output ONLY valid JSON with keys: name, origin, destination, departure_time
3. Use "null" for missing values
4. Format dates as yyyy/mm/dd hh:mi (24-hour format)
5. Current datetime: {current_datetime}<|im_end|>
<|im_start|>user
## Conversation History ##
{history}

## Current Slots ##
{slots}

## Last Human Message ##
{input}<|im_end|>
<|im_start|>assistant
{{
    "name": "null",
    "origin": "null",
    "destination": "null",
    "departure_time": "null"
}}
"""

SLOT_EXTRACTION_PROMPT = PromptTemplate(
    input_variables=["history", "input", "slots", "current_datetime"],
    template=_DEFAULT_SLOT_EXTRACTION_TEMPLATE,
)
