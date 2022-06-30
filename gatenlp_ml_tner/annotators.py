"""
Module that defines Annotator classes to apply a trained model to new documents.
"""
import iobes
import tner
from gatenlp.processing.annotator import Annotator
from gatenlp import Span


class TnerTokenClassificationAnnotator(Annotator):
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
        self.model = tner.TransformerNER(self.model_dir)

    def __call__(self, doc, **kwargs):
        if self.sentence_type is None:
            spans = [Span(0, len(doc))]
        else:
            spans = [a.span for a in doc.annset(self.annset_name).with_type(self.sentence_type)]
        if len(spans) == 0:
            return doc
        outset = doc.annset(self.outset_name)
        # NOTE/TODO: we may want to build the text using tokens and gap space/text here, keeping track
        # of offset changes?
        txts = [doc[s] for s in spans]
        # !!!! TODO: the offsets returned are INCORRECT if there is leading whitespace or multiple 
        # whitespace within each of the txts: apparently the model replaces those 

        # One possible workaround: require that we get the tokens covered by the sentence, not the 
        # sentence text, then construct the sentence "text" by joining the tokens with a single whitespace.
        # for each token remember the starting and ending offset in the generated sentence.
        # Then when we get back the pred offsets, the start offset should be one of our offsets in 
        # generated sentence, so we should be able to map back the pred offset to the token and then 
        # to the original document offset. 
        # Similar for the end offset we get back: each end offset can be mapped back to the in-sentence token
        preds = self.model.predict(txts)

        for span in spans:
            # TODO: if we have a token type construct txt from the tokens, optionally from the token feature
            txt = doc[span]
            # apply model to the text, then get the chunks spans
            preds = self.model.predict(txt)
            # adapt span offsets by adding sentence span start offset and add to outset
            for pred in preds:
                for ent in pred["entity"]:
                    start, end = ent["position"]
                    etype = ent["probability"]
                    prob = ent["probability"]
                    outset.add(start+span.start, end+span.end, etype, features=dict(probability=probability))
        return doc

