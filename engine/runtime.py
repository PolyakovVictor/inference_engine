from typing import Any
from pathlib import Path

from tinygrad import Variable
from tinygrad.engine.jit import TinyJit
from tinygrad.tensor import Tensor
from tinygrad.nn.state import safe_load, load_state_dict

from engine.helper import DEBUG, resolve_chat_template
from engine.transformer import Transformer
from engine.tokenizer import Tokenizer
from engine.config import ModelConfig


CHAT_TEMPLATES = {
    "legacy": {
        "system": "<|system|>\n{content}</s>\n",
        "user":   "<|user|>\n{content}</s>\n",
        "assistant": "<|assistant|>\n{content}</s>\n",
        "generation_prompt": "<|assistant|>\n",
    },
    "chatml": {
        "system": "<|im_start|>system\n{content}<|im_end|>\n",
        "user":   "<|im_start|>user\n{content}<|im_end|>\n",
        "assistant": "<|im_start|>assistant\n{content}<|im_end|>\n",
        "generation_prompt": "<|im_start|>assistant\n",
    },
    "llama3": {
        "system": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>",
        "user":   "<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>",
        "assistant": "<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>",
        "generation_prompt": "<|start_header_id|>assistant<|end_header_id|>\n\n",
    },
}

class Runtime:
    def __init__(self, model_path: str | Path, chat_template: str | None = None) -> None:
        self.model_dir = Path(model_path)
        self.config = ModelConfig.from_json(self.model_dir / "config.json")
        self.tokenizer = Tokenizer(self.model_dir / "tokenizer.json")
        self.model = Transformer(self.config)
        self.load_weights()
        self.max_context = self.config.max_position_embeddings
        self._jit_decode = TinyJit(self._decode_step)
        self.start_pos = 0
        self.messages: list[dict[str, str]] = []
        if chat_template is not None and chat_template not in CHAT_TEMPLATES: raise ValueError(f"Unknown chat_template: {chat_template}. Available: {list(CHAT_TEMPLATES)}")
        self.chat_template = CHAT_TEMPLATES[chat_template or resolve_chat_template(self.model_dir)]
        self.chat_template_name = chat_template

    def _decode_step(self, tokens: Tensor, start_pos: Any) -> Tensor:
        return self.model(tokens=tokens, start_pos=start_pos).realize()
    
    def _feed_tokens(self, tokens: list[int]) -> Tensor | None:
        if not tokens: return None
        input_tensor = Tensor([tokens])
        logits = self.model(input_tensor, start_pos=self.start_pos)
        self.start_pos += len(tokens)
        if DEBUG > 0: print(f"[DEBUG] Fed {len(tokens)} tokens, start_post now = {self.start_pos}")
        return logits

    def load_weights(self) -> None:
        state_dict = safe_load(str(self.model_dir / "model.safetensors"))
        cleaned_state = {k.removeprefix("model."): v for k,v in state_dict.items()}
        if "lm_head.weight" not in cleaned_state and "embed_tokens.weight" in cleaned_state:
            cleaned_state["lm_head.weight"] = cleaned_state["embed_tokens.weight"]
        load_state_dict(self.model, cleaned_state, strict=False)
    
    def sample(self, logits: Tensor, temperature: float = 0.7) -> int:
        if temperature == 0: return int(logits.argmax().item())
        probs = (logits / temperature).softmax(axis=-1)
        return int(probs.multinomial().item())

    def start_conversation(self, system_prompt: str | None = None) -> None:
        self.model.reset_cache()
        self.start_pos = 0
        self.messages = []
        if system_prompt: self.add_message("system", system_prompt)

    def generate(self, user_text: str, max_new_tokens: int = 50, temperature: float = 0.7) -> str:
        self.messages.append({"role": "user", "content": user_text})
        delta = self.format_turn("user", user_text, add_generation_prompt=True)
        tokens = self.tokenizer.encode(delta, add_bos=(self.start_pos == 0))

        if not tokens: return ""
        logits = self._feed_tokens(tokens)
        assert logits is not None
        next_tokens = self.sample(logits[0,-1], temperature=temperature)
        generated: list[int] = [next_tokens]

        if DEBUG > 0: print(f"[DEBUG] Prefill tokens: {tokens}, start_pos now: {self.start_pos}")
        for _ in range(max_new_tokens - 1):
            if next_tokens in self.tokenizer.eos_ids: break
            input_tensor = Tensor([[next_tokens]])
            var = Variable("start_pos", 1, self.max_context).bind(self.start_pos)
            logits = self._jit_decode(input_tensor, var)
            next_tokens = self.sample(logits[0,-1], temperature=temperature)
            generated.append(next_tokens)
            self.start_pos += 1
        response = self.tokenizer.decode(generated)
        for eos in self.tokenizer.eos_tokens: response = response.replace(eos, "")
        response = response.strip()
        self.messages.append({"role": "assistant", "content": response})
        return response
 
    def format_turn(self, role: str, content: str, add_generation_prompt: bool = False) -> str:
        if role not in ("system", "user", "assistant"): raise ValueError(f"Unknown role: {role}")
        text = self.chat_template[role].format(content=content)
        if add_generation_prompt: text += self.chat_template["generation_prompt"]
        return text

    def add_message(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        text = self.format_turn(role, content, add_generation_prompt=False)
        tokens = self.tokenizer.encode(text, add_bos=(self.start_pos == 0))
        self._feed_tokens(tokens)

    def clear(self) -> None: self.start_conversation()
