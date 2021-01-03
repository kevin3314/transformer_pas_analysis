from typing import List, Optional, Tuple

from transformers import BertTokenizer

from .utils import TokenizeHandlerMeta


class BertTokenizeHandler(BertTokenizer, TokenizeHandlerMeta):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pad_id = self._convert_token_to_id(self.pad_token)

    @property
    def pad_id(self) -> int:
        return self._pad_id

    def get_tokenized_tokens(
            self, words: List[str]
    ) -> Tuple[List[str], List[Optional[int]], List[int]]:
        all_tokens = []
        tok_to_orig_index: List[Optional[int]] = []
        orig_to_tok_index: List[int] = []

        all_tokens.append("[CLS]")
        # There's no original token corresponding to [CLS] token
        tok_to_orig_index.append(None)

        for i, word in enumerate(words):
            orig_to_tok_index.append(len(all_tokens))  # assign head subword
            sub_tokens = self.tokenize(word)
            for sub_token in sub_tokens:
                all_tokens.append(sub_token)
                tok_to_orig_index.append(i)

        all_tokens.append("[SEP]")
        # There's no original token corresponding to [SEP] token
        tok_to_orig_index.append(None)

        is_intermediate_list = self.check_intermediate_tokens(all_tokens)

        return all_tokens, tok_to_orig_index, orig_to_tok_index, is_intermediate_list

    def check_intermediate(self, word: str) -> bool:
        return word.startswith("##")

    def check_intermediate_tokens(self, all_tokens: List[str]) -> List[bool]:
        """各トークンが単語の中間か、それ以外かを表す bool 値のリストを返す

        Args:
            all_tokens (List[str]): トークンのリスト

        Returns:
            List[bool]: 各トークンが単語の中間か、それ以外かを表すリスト
                        中間なら True, それ以外なら False
        """
        return [token.startswith("##") for token in all_tokens]
