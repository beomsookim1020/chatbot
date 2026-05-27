from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class HuggingFaceGenerationConfig:
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    max_new_tokens: int = 512
    temperature: float = 0.1
    top_p: float = 0.9
    repetition_penalty: float = 1.05
    device: str | None = None
    torch_dtype: str = "auto"


class HuggingFaceGenerator:
    """Local Hugging Face chat generator for retrieval-grounded answers."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        top_p: float = 0.9,
        repetition_penalty: float = 1.05,
        device: str | None = None,
        torch_dtype: str = "auto",
    ):
        self.config = HuggingFaceGenerationConfig(
            model_name=model_name,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            device=device,
            torch_dtype=torch_dtype,
        )
        self._tokenizer = None
        self._model = None
        self._device = None
        self._processor_mode = False

    def generate(self, query: str, contexts: list[str] | str, system_prompt: str | None = None) -> str:
        prompt = self._coerce_prompt(query=query, contexts=contexts)
        return self.generate_prompt(prompt, system_prompt=system_prompt)

    def generate_prompt(self, prompt: str, system_prompt: str | None = None) -> str:
        tokenizer, model = self._load()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        if self._processor_mode:
            return self._generate_with_processor(tokenizer, model, messages)

        if getattr(tokenizer, "chat_template", None):
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            input_text = self._fallback_chat_text(messages)

        inputs = tokenizer(input_text, return_tensors="pt")
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        do_sample = self.config.temperature > 0

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": do_sample,
            "repetition_penalty": self.config.repetition_penalty,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = self.config.temperature
            generation_kwargs["top_p"] = self.config.top_p

        import torch

        with torch.inference_mode():
            output_ids = model.generate(**inputs, **generation_kwargs)

        prompt_length = inputs["input_ids"].shape[-1]
        generated_ids = output_ids[0][prompt_length:]
        return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def _generate_with_processor(self, processor, model, messages: list[dict[str, str]]) -> str:
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        )
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        do_sample = self.config.temperature > 0

        eos_token_id = getattr(processor, "eos_token_id", None)
        if eos_token_id is None and getattr(processor, "tokenizer", None) is not None:
            eos_token_id = getattr(processor.tokenizer, "eos_token_id", None)
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": do_sample,
            "repetition_penalty": self.config.repetition_penalty,
        }
        if eos_token_id is not None:
            generation_kwargs["pad_token_id"] = eos_token_id
        if do_sample:
            generation_kwargs["temperature"] = self.config.temperature
            generation_kwargs["top_p"] = self.config.top_p

        import torch

        with torch.inference_mode():
            output_ids = model.generate(**inputs, **generation_kwargs)

        prompt_length = inputs["input_ids"].shape[-1]
        generated_ids = output_ids[0][prompt_length:]
        if hasattr(processor, "decode"):
            return processor.decode(generated_ids, skip_special_tokens=True).strip()
        if getattr(processor, "tokenizer", None) is not None:
            return processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return str(generated_ids).strip()

    def _load(self):
        if self._tokenizer is not None and self._model is not None:
            return self._tokenizer, self._model

        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

        device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = self._resolve_torch_dtype(device)

        try:
            tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, trust_remote_code=True)
            self._processor_mode = False
        except Exception:
            tokenizer = AutoProcessor.from_pretrained(self.config.model_name, trust_remote_code=True)
            self._processor_mode = True
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        model.to(device)
        model.eval()

        self._tokenizer = tokenizer
        self._model = model
        self._device = device
        return tokenizer, model

    def _resolve_torch_dtype(self, device: str):
        import torch

        if self.config.torch_dtype != "auto":
            return getattr(torch, self.config.torch_dtype)
        if device == "cuda":
            return torch.float16
        return torch.float32

    @staticmethod
    def _coerce_prompt(query: str, contexts: list[str] | str) -> str:
        if isinstance(contexts, str):
            context_text = contexts
        else:
            context_text = "\n\n".join(contexts)
        return f"[질문]\n{query}\n\n[Context]\n{context_text}"

    @staticmethod
    def _fallback_chat_text(messages: list[dict[str, str]]) -> str:
        parts = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            parts.append(f"[{role}]\n{content}")
        parts.append("[assistant]\n")
        return "\n\n".join(parts)
