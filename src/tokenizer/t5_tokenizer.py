from typing import Tuple, List, Optional
from transformers import T5Tokenizer

from .utils import SpmMixin, EncDecTokenizeHandlerMeta


class T5TokenizeHandler(T5Tokenizer, SpmMixin, EncDecTokenizeHandlerMeta):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pad_id = self._convert_token_to_id(self.pad_token)

    @property
    def pad_id(self) -> int:
        return self._pad_id

    def get_decoder_tokenized_tokens(
        self,
        words: List[str]
    ) -> Tuple[List[str], List[Optional[int]], List[int]]:
        return self.get_encoder_tokenized_tokens(words)
