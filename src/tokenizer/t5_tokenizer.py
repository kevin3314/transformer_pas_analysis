from transformers import T5Tokenizer

from .utils import SpmMixin, TokenizeHandlerMeta


class T5TokenizeHandler(T5Tokenizer, SpmMixin, TokenizeHandlerMeta):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pad_id = self._convert_token_to_id(self.pad_token)

    @property
    def pad_id(self) -> int:
        return self._pad_id
