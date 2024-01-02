# https://github.com/potamides/uniformers/blob/b2644063cc1e2e66a443f1dbad083cdcf684486c/uniformers/utils/clean.py#L21
from string import punctuation
from typing import List
from collections.abc import Iterable
from nltk.tokenize import sent_tokenize
from sacremoses import MosesDetokenizer, MosesPunctNormalizer, MosesTokenizer
import nltk
#nltk.download('punkt')
import re

# Some English datasets we use are already tokenized, so we have to be careful
# with apostrophes. Generally, this change could lead to unintended
# consequences but for the poetry domain it should be fine
ENGLISH_SPECIFIC_APOSTROPHE = [
    (r"([{0}])\s[']\s([{0}])".format(MosesTokenizer.IsAlpha), r"\1'\2"),
    (r"([{isn}])\s[']\s([s])".format(isn=MosesTokenizer.IsN), r"\1'\2"),
] + MosesTokenizer.ENGLISH_SPECIFIC_APOSTROPHE

# in German poetry the apostrophe is also used for contraction (in contrast to
# prose), so we have to adapt that as well
NON_SPECIFIC_APOSTROPHE = r"\'", "'"  # pyright: ignore

punkt = '#$%&\()*+/<=>@[\\]"^_`\{|}~'
def clean_sentence(
    sentence: str,
    lang: str,
    protected = None,
    detokenize = True,
):
    pattern = r'[0-9]'
    sentence = re.sub(pattern, '', sentence)
    sentence = sentence.translate(str.maketrans('', '', punkt))
    mpn = MosesPunctNormalizer(lang=lang)
    md = MosesDetokenizer(lang=lang)
    mt = MosesTokenizer(lang=lang)
    mt.ENGLISH_SPECIFIC_APOSTROPHE = ENGLISH_SPECIFIC_APOSTROPHE
    mt.NON_SPECIFIC_APOSTROPHE = NON_SPECIFIC_APOSTROPHE  # pyright: ignore

    tokenized = mt.tokenize(mpn.normalize(sentence), protected_patterns=protected)
    sents = sent_tokenize(md.detokenize(tokenized))
    lst = []
    for sent in sents:
        if len(sent)<3:
            pass
        else:
            lst.append(sent)
    sents = "\n".join(lst)
    r = re.compile(r'([.,/#!$%^&*;:{}=_`~()-?])[.,/#!$%^&*;:{}=_`~()-]+')
    sents = r.sub(r'\1', sents)
    return sents