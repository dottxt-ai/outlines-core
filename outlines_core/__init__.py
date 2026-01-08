import sys

# kernels is not reexported as it should remain an optional dependency
from . import _json_schema as json_schema
from .outlines_core import Guide, Index, Vocabulary

# Register json_schema in sys.modules so "from outlines_core.json_schema
# import ..." works
sys.modules["outlines_core.json_schema"] = json_schema
