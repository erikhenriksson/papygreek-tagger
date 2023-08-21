from example import example_sentence

from papygreektagger import tag

from pprint import pprint

result = tag(example_sentence)

pprint(result)


"""
NOTE: If the model doesn't run, try changing the flair/nn/model.py load_state_dict to strict=False

"""
