from typing import Tuple, List, Optional

from transformers import BartSPMTokenizer

from .utils import SpmMixin, EncDecTokenizeHandlerMeta


class BartSPMTokenizeHandler(BartSPMTokenizer, SpmMixin, EncDecTokenizeHandlerMeta):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pad_id = self._convert_token_to_id(self.pad_token)

    @property
    def pad_id(self) -> int:
        return self._pad_id

    def get_encoder_tokenized_tokens(
            self, words: List[str]
    ) -> Tuple[List[str], List[Optional[int]], List[int]]:
        all_tokens = []
        tok_to_orig_index = []
        orig_to_tok_index = []
        is_intermediate_list = []

        # <s> token
        all_tokens.append('<s>')
        tok_to_orig_index.append(None)
        is_intermediate_list.append(True)

        self.tokenize_content_words(
            words,
            all_tokens,
            tok_to_orig_index,
            orig_to_tok_index,
            is_intermediate_list,
            remove_alone_special=False
        )

        # </s> token
        all_tokens.append('</s>')
        tok_to_orig_index.append(None)
        is_intermediate_list.append(True)

        return all_tokens, tok_to_orig_index, orig_to_tok_index, is_intermediate_list

    def get_decoder_tokenized_tokens(
        self,
        words: List[str]
    ) -> Tuple[List[str], List[Optional[int]], List[int]]:
        return self.get_encoder_tokenized_tokens(words)
