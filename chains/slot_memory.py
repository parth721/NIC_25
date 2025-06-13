import copy
import json
from typing import Any, Dict, List, ClassVar
from pydantic import Field
from datetime import datetime
from langchain.chains.llm import LLMChain
from langchain.memory.chat_memory import BaseChatMemory
from langchain.memory.entity import BaseEntityStore, InMemoryEntityStore
from langchain.memory.utils import get_prompt_input_key
from langchain.prompts.base import BasePromptTemplate
from langchain_core.language_models.base import BaseLanguageModel
from langchain.schema.messages import get_buffer_string
from chains.prompt import SLOT_EXTRACTION_PROMPT
import re
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from typing import Dict


class SlotMemory(BaseChatMemory):
    # Public LangChain-compatible fields
    llm: BaseLanguageModel
    slot_extraction_prompt: BasePromptTemplate = SLOT_EXTRACTION_PROMPT
    k: int = 10
    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    chat_history_key: str = "history"
    slot_key: str = "slots"
    inform_check_key: str = "check"
    return_messages: bool = False
    entity_store: BaseEntityStore = Field(default_factory=InMemoryEntityStore)

    # ✅ Class-level constants (Pydantic-safe)
    default_slots: ClassVar[Dict[str, str]] = {
        "name": "null",
        "origin": "null",
        "destination": "null",
        "departure_time": "null"
    }

    # ✅ Runtime-only mutable attributes (init in __init__ or __post_init__)
    current_slots: Dict[str, str] = Field(default_factory=lambda: copy.deepcopy(SlotMemory.default_slots))
    inform_check: bool = False
    current_datetime: str = Field(default_factory=lambda: datetime.now().strftime("%Y/%m/%d %H:%M"))

    @property
    def buffer(self) -> Any:
        """String buffer of memory."""
        if self.return_messages:
            return self.chat_memory.messages
        else:
            return get_buffer_string(
                self.chat_memory.messages,
                human_prefix=self.human_prefix,
                ai_prefix=self.ai_prefix,
            )

    @property
    def memory_variables(self) -> List[str]:
        """Always return list of memory variable keys."""
        return [self.slot_key, self.chat_history_key, self.inform_check_key]

    def information_check(self) -> None:
        """Determine if all slots are filled."""
        self.inform_check = all(value != "null" for value in self.current_slots.values())

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and update slots based on latest input."""
        buffer_string = get_buffer_string(
            self.chat_memory.messages[-self.k * 2:],
            human_prefix=self.human_prefix,
            ai_prefix=self.ai_prefix,
        )

        prompt_input_key = self.input_key or get_prompt_input_key(inputs, self.memory_variables)
        slots = self.current_slots
        chain = self.slot_extraction_prompt | self.llm
        output = chain.invoke({
           "history": buffer_string,
           "input": inputs[prompt_input_key],
           "slots": slots,
           "current_datetime": self.current_datetime
        })
        
# ----   json_match = re.search(r'\{.*\}', output, re.DOTALL)

        json_match = re.search(r'\{[\s\S]*\}', output)
        if json_match:
            try:
                output_json = json.loads(json_match.group(0))
            except json.JSONDecodeError:
                output_json = self.current_slots
        else:
            output_json = self.current_slots
            
            
#        try:
#           output_json = json.loads(output)
#        except Exception:
#            print(f"[SlotMemory Warning] Could not parse model output:\n{output}")
# -----      output_json = slots

        for k, v in output_json.items():
            if v and v != "null":
                self.current_slots[k] = v

        self.information_check()
        return {
            self.chat_history_key: buffer_string,
            self.slot_key: str(self.current_slots),
            self.inform_check_key: str(self.inform_check)
        }

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Store new user and model turns into chat memory."""
        super().save_context(inputs, outputs)

    def clear(self) -> None:
        """Reset memory and slots."""
        self.chat_memory.clear()
        self.entity_store.clear()
        self.current_slots = copy.deepcopy(self.default_slots)
        self.current_datetime = datetime.now().strftime("%Y/%m/%d %H:%M")


####special educator, class educator, 6-a anupam biswas 4