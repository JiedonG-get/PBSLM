"""Conservative exact-syntax dependency routing reference, not learned relevance.

Only explicit known entity/color updates for disjoint entities are background.
Unknown/ambiguous inputs fail closed to interrupt. Caller declares speech dependency
set; missing dependencies can make routing semantically unsafe. No truth validation.
"""
from .continuous_state import ExactReferenceExtractor


class DependencyRouter:
    def __init__(self,runner):
        self.runner=runner;self.dependencies=None;self.journal=[]
        self.extractor=ExactReferenceExtractor()

    def set_dependencies(self,entities):
        if not isinstance(entities,(set,frozenset)) or not entities or any(type(e) is not int or not 0<=e<16 for e in entities):
            raise ValueError('nonempty known entity dependency set required')
        self.dependencies=frozenset(entities)

    def clear_dependencies(self):self.dependencies=None

    async def observe(self,text,*,respond_on_interrupt=True):
        parsed=self.extractor(text)
        disjoint=(self.dependencies is not None and parsed.get('accepted',False)
                  and parsed['entity'] not in self.dependencies)
        reason='exact_disjoint_entity' if disjoint else 'relevant_or_unknown'
        # Record admission only after queue/hook accepts the event.
        if disjoint:
            revision=await self.runner.observe_background(text,relevance_assertion=
                           'exact reference parser: entity outside caller-declared speech dependencies')
        else:
            revision=await self.runner.observe(text,should_respond=respond_on_interrupt)
            # Previous speech dependencies cannot be assumed for a new answer.
            self.dependencies=None
        self.journal.append({'revision':revision,'text':text,'route':'background' if disjoint else 'interrupt',
                             'reason':reason,'parsed':parsed,
                             'policy':'caller dependencies + exact syntax, not model semantic relevance'})
        return revision
