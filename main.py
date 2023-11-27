# Dependencies
import logging
from semantic import SemanticModel

# Remove all handlers associated with the root logger object.
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

semantic_seg = SemanticModel()
semantic_seg.train_evaluate()
