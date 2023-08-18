from example import example_sentence

from papygreektagger import tagger

from pprint import pprint

result = tagger.predict(example_sentence)

pprint(result)
