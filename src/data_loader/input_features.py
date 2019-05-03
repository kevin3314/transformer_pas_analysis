from typing import NamedTuple, List


class InputFeatures:
    """A single set of features of data."""

    def __init__(self,
                 tokens: List[str],
                 orig_to_tok_index: List[int],  # use for output
                 tok_to_orig_index: List[int],  # use for output
                 input_ids: List[int],  # use for model
                 input_mask: List[int],  # use for model
                 segment_ids: List[int],  # use for model
                 arguments_set: List[List[int]] = None,  # use for model
                 ng_arg_ids_set: List[List[int]] = None,  # use for model
                 ):
        self.tokens = tokens
        self.orig_to_tok_index = orig_to_tok_index
        self.tok_to_orig_index = tok_to_orig_index
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.arguments_set = arguments_set
        self.ng_arg_ids_set = ng_arg_ids_set
