from functools import singledispatch

import interegular

from outlines.fsm.guide import RegexGuide
from outlines.generate.api import SequenceGeneratorAdapter
from outlines.samplers import Sampler, multinomial


@singledispatch
def fsm(
    model, fsm: interegular.fsm.FSM, sampler: Sampler = multinomial()
) -> SequenceGeneratorAdapter:
    from outlines.processors import GuideLogitsProcessor

    guide = RegexGuide.from_interegular_fsm(fsm, model.tokenizer)
    logits_processor = GuideLogitsProcessor(tokenizer=model.tokenizer, guide=guide)
    return SequenceGeneratorAdapter(model, logits_processor, sampler)
