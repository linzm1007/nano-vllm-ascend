from nanovllm.engine.llm_engine import LLMEngine


class LLM(LLMEngine):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
