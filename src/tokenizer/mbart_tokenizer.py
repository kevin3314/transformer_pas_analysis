from typing import Tuple, List, Optional

from transformers import MBartTokenizer

from .utils import SpmMixin, EncDecTokenizeHandlerMeta


class MBartTokenizerHandler(MBartTokenizer, SpmMixin, EncDecTokenizeHandlerMeta):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, src_lang='ja_XX', **kwargs)
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
            remove_alone_special=True
        )

        # </s> token
        all_tokens.append('</s>')
        tok_to_orig_index.append(None)
        is_intermediate_list.append(True)

        # 'ja_XX' token
        all_tokens.append('ja_XX')
        tok_to_orig_index.append(None)
        is_intermediate_list.append(True)

        return all_tokens, tok_to_orig_index, orig_to_tok_index, is_intermediate_list

    def get_decoder_tokenized_tokens(
        self,
        words: List[str]
    ) -> Tuple[List[str], List[Optional[int]], List[int]]:
        all_tokens = []
        tok_to_orig_index = []
        orig_to_tok_index = []
        is_intermediate_list = []

        # 'ja_XX' token
        all_tokens.append('ja_XX')
        tok_to_orig_index.append(None)
        is_intermediate_list.append(True)

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
            remove_alone_special=True
        )

        # </s> token
        all_tokens.append('</s>')
        tok_to_orig_index.append(None)
        is_intermediate_list.append(True)

        return all_tokens, tok_to_orig_index, orig_to_tok_index, is_intermediate_list
