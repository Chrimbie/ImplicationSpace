"""UCE analysis, Christopher's question 2026-10-21:
"With variables like infinite bits and the size of the model,
what does UCE say?"

Method per the UCE skill: define minimal structures, instantiate
across domains with honest bindings (unknown/none allowed), include
a negative control, run compare_instances, report what survives.
"""

import sys

sys.path.insert(0, '/home/hatch/workspace/custom-skills/local-uce/bin')
from uce import ConceptEngine  # noqa: E402

e = ConceptEngine()

# ============================================================ BITS ======
e.create_structure("bit_supply")
for role in ("origin", "stored", "access", "freedom"):
    e.add_role("bit_supply", role)
e.add_relation("bit_supply", "determines", "origin", "stored")
e.add_relation("bit_supply", "answers_through", "stored", "access")

e.create_domain("implication_space")
e.create_domain("ordinary_memory")
e.create_domain("mathematics")
e.create_domain("chance")

e.instantiate("bit_supply", "implication_space", "implied_bits")
e.bind("implied_bits", "origin", "finite: real bits + a fixed rule")
e.bind("implied_bits", "stored", "only the real bits")
e.bind("implied_bits", "access", "any bit derived on demand, instantly")
e.bind("implied_bits", "freedom", "none: every bit fixed by the origin")

e.instantiate("bit_supply", "ordinary_memory", "ram_bits")
e.bind("ram_bits", "origin", "whatever was written, bit by bit")
e.bind("ram_bits", "stored", "every bit")
e.bind("ram_bits", "access", "read where it sits")
e.bind("ram_bits", "freedom", "full: each bit set independently")

e.instantiate("bit_supply", "mathematics", "pi_digits")
e.bind("pi_digits", "origin", "finite: the definition of pi")
e.bind("pi_digits", "stored", "nothing (the definition only)")
e.bind("pi_digits", "access", "any digit computed on demand")
e.bind("pi_digits", "freedom", "none: every digit fixed by the origin")

e.instantiate("bit_supply", "chance", "coin_bits")   # negative control
e.bind("coin_bits", "origin", "none: no rule, no definition")
e.bind("coin_bits", "stored", "nothing until flipped")
e.bind("coin_bits", "access", "only by generating (flipping)")
e.bind("coin_bits", "freedom", "full: each bit independent of all others")

print("=" * 70)
print("STRUCTURE 1: bit_supply -- the concept behind 'infinite bits'")
print("=" * 70)
print(e.compare_instances("implied_bits", "pi_digits"))
print(e.compare_instances("implied_bits", "ram_bits"))
print(e.compare_instances("implied_bits", "coin_bits"))

# ============================================================ SIZE ======
e.create_structure("model_size")
for role in ("count", "stored", "learned", "capability"):
    e.add_role("model_size", role)
e.add_relation("model_size", "is_carried_by", "count", "stored")
e.add_relation("model_size", "grows_only_by", "stored", "learned")
e.add_relation("model_size", "shows_up_as", "learned", "capability")

e.create_domain("conventional_ml")
e.create_domain("implication_space")
e.create_domain("statistics")
e.create_domain("no_learning")

e.instantiate("model_size", "conventional_ml", "llm_70b")
e.bind("llm_70b", "count", "70 billion parameters")
e.bind("llm_70b", "stored", "70 billion values")
e.bind("llm_70b", "learned", "70 billion independent values")
e.bind("llm_70b", "capability", "high: fluent language")

e.instantiate("model_size", "implication_space", "enormous_in_space")
e.bind("enormous_in_space", "count", "10^41 nominal parameters")
e.bind("enormous_in_space", "stored", "1,056 values")
e.bind("enormous_in_space", "learned", "1,056 independent values")
e.bind("enormous_in_space", "capability", "seed-level: 93% on cancer")

e.instantiate("model_size", "implication_space", "text_10_80")
e.bind("text_10_80", "count", "10^80 nominal parameters")
e.bind("text_10_80", "stored", "35,328 values")
e.bind("text_10_80", "learned", "35,328 independent values")
e.bind("text_10_80", "capability", "20% ceiling on dialogue")

e.instantiate("model_size", "statistics", "bigram_table")
e.bind("bigram_table", "count", "about a million counted pairs")
e.bind("bigram_table", "stored", "the counted pairs")
e.bind("bigram_table", "learned", "the counted pairs")
e.bind("bigram_table", "capability", "20.4% on the same dialogue")

e.instantiate("model_size", "no_learning", "random_weights")  # control
e.bind("random_weights", "count", "10^12 random parameters")
e.bind("random_weights", "stored", "10^12 values")
e.bind("random_weights", "learned", "none: never trained")
e.bind("random_weights", "capability", "chance level")

print()
print("=" * 70)
print("STRUCTURE 2: model_size -- the concept behind 'size of the model'")
print("=" * 70)
print(e.compare_instances("llm_70b", "enormous_in_space"))
print(e.compare_instances("enormous_in_space", "text_10_80"))
print(e.compare_instances("text_10_80", "bigram_table"))
print(e.compare_instances("llm_70b", "random_weights"))
