"""
Module that defines Annotator classes to apply a trained model to new documents.
"""
import iobes
from gatenlp.processing.annotator import Annotator
from gatenlp import Span


class TneTokenClassificationAnnotator(Annotator):
    """
    Annotator to apply a token classification model for chunking to the text in a document.
    """
    def __init__(
            self,
            model_dir,
            type_map=None,
            annset_name="",
            outset_name="",
            token_type=None,
            token_feature=None,
            sentence_type=None,
    ):
        """

        :param model_dir:
        :param type_map:
        :param annset_name:
        :param outset_name:
        :param token_type:
        :param token_feature:
        :param sentence_type:
        """
        self.model_dir = model_dir
        if type_map is None:
            type_map = {}
        if token_type is not None or token_feature is not None:
            raise Exception("Using token_type or token_feature is not implemented yet!")
        self.type_map = type_map
        self.annset_name = annset_name
        self.outset_name = outset_name
        self.sentence_type = sentence_type

    def __call__(self, doc, **kwargs):
        if self.sentence_type is None:
            spans = [Span(0, len(doc))]
        else:
            spans = [a.span for a in doc.annset(self.annset_name).with_type(self.sentence_type)]
        for span in spans:
            # TODO: if we have a token type construct txt from the tokens, optionally from the token feature
            txt = doc[span]
            # apply model to the text, then get the chunks spans
            # adapt span offsets by adding sentence span start offset and add to outset
            pass
