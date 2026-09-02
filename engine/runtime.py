import os
os.environ["DEFAULT_FLOAT"] = "half"

import json, os

from typing import Any
from pathlib import Path

from tinygrad import Variable, Context, Device
from tinygrad.dtype import dtypes
from tinygrad.engine.jit import TinyJit
from tinygrad.tensor import Tensor
from tinygrad.llm.gguf import gguf_load
from tinygrad.nn.state import safe_load, load_state_dict, torch_load
from tinygrad.extra.models.llama import convert_from_huggingface, convert_from_gguf, fix_bf16
from tinygrad.extra.models.llama import Transformer as TinyTransofrmer

from engine.helper import DEBUG, resolve_chat_template
from engine.transformer import Transformer, Int8Linear
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
    
    def concat_weights(self, models, device=None):
        def convert(name) -> Tensor:
            disk_tensors: list[Tensor] = [model[name] for model in models]
            if len(disk_tensors) == 1 or len(disk_tensors[0].shape) == 1:
                return disk_tensors[0].to(device=device)
            axis = 1 if name.endswith((".attention.wo.weight", ".feed_forward.w2.weight")) else 0
            lazy_tensors = [data.to(device=device) for data in disk_tensors]
            return lazy_tensors[0].cat(*lazy_tensors[1:], dim=axis)
        return {name: convert(name) for name in {name: None for model in models for name in model}}

    def load(self, fn:str):
        if fn.endswith('.index.json'):
            with open(fn) as fp: weight_map = json.load(fp)['weight_map']
            parts = {n: self.load(str(Path(fn).parent / Path(n).name)) for n in set(weight_map.values())}
            return {k: parts[n][k] for k, n in weight_map.items()}
        elif fn.endswith(".gguf"):
            gguf_tensor = Tensor.empty(os.stat(fn).st_size, dtype=dtypes.uint8, device=f"disk:{fn}").to(Device.DEFAULT)
            return gguf_load(gguf_tensor)[1]
        elif fn.endswith(".safetensors"):
            return safe_load(fn)
        else:
            return torch_load(fn)

    def load_weights(self) -> None:
        if self.model_dir.is_dir():
            if (self.model_dir / "model.safetensors.index.json").exists(): 
                weights = self.load(str(self.model_dir / "model.safetensors.index.json"))
            elif (self.model_dir / "model.safetensors").exists(): 
                weights = self.load(str(self.model_dir / "model.safetensors"))
            else: raise FileNotFoundError(...)
            # else: weights = self.concat_weights([self.load(str(self.model_dir / f"consolidated.{i:02d}.pth")) for i in range(MODEL_PARAMS[model_size]["files"])], device[0] if isinstance(device, tuple) else device)
        else:
            weights = self.load(str(self.model_dir))
        if any(k.startswith("model.") for k in weights):
            cleaned = {}
            for k,v in weights.items():
                cleaned[k.removeprefix("model.")] = v
            weights = cleaned
        
        if "lm_head.weight" not in weights and "embed_tokens.weight" in weights: 
            weights["lm_head.weight"] = weights["embed_tokens.weight"]

        print("Streaming and quantizing weights tensor-by-tensor...")
        quantized_weights = {}

        for k, v in weights.items():
            v_real = v.to(Device.DEFAULT).realize()
            
            v_real = fix_bf16({k: v_real})[k]

            is_linear = any(x in k for x in ["self_attn", "mlp", "lm_head"])
            
            if is_linear and k.endswith(".weight"):
                # 2. Квантуємо на льоту
                v_real = v_real.cast(dtypes.float16)
                scale = v_real.abs().max(axis=1) / 127.0
                int8_weight = (v_real.T / scale).T.round().cast(dtype=dtypes.int8).contiguous().realize()
                scale_real = scale.contiguous().realize()

                quantized_weights[k] = int8_weight
                quantized_weights[k.replace('.weight', '.scale')] = scale_real
            else:
                quantized_weights[k] = v_real.cast(dtypes.float16).contiguous().realize()

            del v_real
            del v
        
        del weights

        print("Weights quantized and loaded to RAM/GPU.")
        load_state_dict(self.model, quantized_weights, strict=False, consume=True, realize=False)

    def sample(self, logits: Tensor, temperature: float = 0.7) -> int:
        print(f"[sample] logits.shape={logits.shape} dtype={logits.dtype}")
        flat = logits[0,1] if logits.ndim == 3 else logits
        print(f"[sample] flat.shape={flat.shape}")
        flat = flat.cast(dtypes.float32).contiguous().reshape(-1)
        if temperature <= 1e-5: return int(flat.argmax().item())
        probs = (flat / temperature).softmax(axis=-1)
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
        print(f"[generate] model logits shape = {logits.dtype}")
        assert logits is not None
        logits = logits.cast(dtypes.float16).contiguous().realize()
        print("logits realized successfully")
        next_tokens = self.sample(logits, temperature=temperature)
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
