# -*- coding: utf-8 -*-
# Extracted from: Chapter 123 — CLIP and Contrastive Vision–Language Alignment
# Source: src/.../ch123-clip.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Zero-shot classification is retrieval, and that changes what it depends on.

CLIP classifies without a classifier: embed the image, embed a text description of
each candidate class, and take the nearest (eq:zero-shot-as-retrieval). No head is
trained, no class list is fixed at training time, and the label set can change at
runtime -- which is the property that made the technique matter.

It also means the "classifier" is a set of text embeddings, so its decision
boundary is determined by how the classes were WORDED (eq:zero-shot-boundary).
This listing measures that dependence, and measures the standard mitigation.

The class-name geometry here is deliberately not uniform: some classes are near
neighbours in the shared space and some are isolated, because that is what a real
label set looks like and it is what makes prompt wording matter unevenly.
"""
import numpy as np

rng = np.random.default_rng(61)

N_CLASS, DIM = 40, 48
N_IMG = 4000
N_TEMPLATE = 8


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


# True class directions in the shared space. Half the classes are grouped into
# tight clusters of near-synonyms, which is where prompt wording does damage.
base = unit(rng.normal(size=(N_CLASS // 2, DIM)))
cls_dir = np.zeros((N_CLASS, DIM))
cls_dir[:N_CLASS // 2] = base
for i in range(N_CLASS // 2):
    cls_dir[N_CLASS // 2 + i] = unit(base[i] + 0.45 * rng.normal(size=DIM))
cls_dir = unit(cls_dir)

# Images: the class direction plus content the caption never mentions.
y = rng.integers(0, N_CLASS, size=N_IMG)
img = unit(cls_dir[y] + 0.42 * rng.normal(size=(N_IMG, DIM)))

# A prompt template shifts every class embedding by a shared, template-specific
# direction ("a photo of a {}", "a blurry photo of a {}", ...) and adds a small
# per-class wording effect (eq:template-model). The shared component is the
# interesting one: it is the same for every class, so it cannot carry class
# information, and it still moves the decision boundary.
TEMPLATES = []
for t in range(N_TEMPLATE):
    shared = unit(rng.normal(size=DIM)) * rng.uniform(0.15, 0.75)
    per_class = 0.22 * rng.normal(size=(N_CLASS, DIM))
    TEMPLATES.append(unit(cls_dir + shared + per_class))


def accuracy(text_emb):
    return float((img @ text_emb.T).argmax(1).__eq__(y).mean())


print(f"{N_CLASS} classes ({N_CLASS // 2} of them near-synonym pairs), "
      f"{N_IMG} images\n")
print(f"{'prompt template':<22}{'accuracy':>11}")
print("-" * 34)
accs = []
for t in range(N_TEMPLATE):
    a = accuracy(TEMPLATES[t])
    accs.append(a)
    print(f"{'template ' + str(t + 1):<22}{a:>11.3f}")

oracle = accuracy(cls_dir)
ens = accuracy(unit(np.mean(TEMPLATES, axis=0)))
best_single = max(accs)

print("-" * 34)
print(f"{'worst template':<22}{min(accs):>11.3f}")
print(f"{'best template':<22}{best_single:>11.3f}")
print(f"{'ENSEMBLE of all 8':<22}{ens:>11.3f}")
print(f"{'oracle class direction':<22}{oracle:>11.3f}")

print(f"""
Read the single-template rows first. They span {min(accs):.3f} to
{best_single:.3f} -- a spread of {best_single - min(accs):.3f} from nothing but
how the classes were phrased. Identical images, identical class set, identical
model. Only the sentences standing in for the classifier changed.

That is eq:zero-shot-as-retrieval's direct consequence and it is easy to miss.
There is no trained head, so the boundary between two classes is the
perpendicular bisector of their two TEXT embeddings, and a template moves those
embeddings. A zero-shot classifier IS a set of sentences, so its errors are
sentence errors.

Now compare every single template against the oracle row, {oracle:.3f}, which
uses the true class directions with no wording at all. The best template reaches
{best_single:.3f}. Wording is not costing a couple of points here -- it is costing
most of the available accuracy, because each template adds a direction to every
class embedding that has nothing to do with the class.

And now the row that changes what you should DO about it. The ensemble -- the
mean of all eight templates' embeddings, re-normalised -- scores {ens:.3f}. That
is {ens - best_single:.3f} above the best single template, while CHOOSING the
best template rather than the worst was worth only
{best_single - min(accs):.3f}. Averaging is worth {(ens - best_single) / max(best_single - min(accs), 1e-9):.0f} times as much as
selecting.

The mechanism is why this generalises. Each template contributes two nuisance
terms: a shared direction added to every class, and a per-class wording effect.
Neither carries class information, and both are independent across templates, so
their average shrinks like 1/sqrt(N) while the class direction -- common to every
template -- survives untouched (eq:ensembling-rate). Prompt ensembling is not a
trick; it is averaging out a nuisance variable, and the square-root law says the
returns diminish but do not stop.

The shared component deserves one note because it is counter-intuitive. It is the
SAME vector added to every class, so it cannot possibly carry information that
distinguishes classes -- and it still changes the answer, because adding a
constant vector to points on a sphere and re-normalising does not preserve their
ranking against a query (eq:renormalisation-not-rank-preserving). A perfectly
uninformative change to the prompt moves the decision boundary.

Note also where the damage concentrates. Half of these classes are near-synonym
pairs sitting close together in the space, and those are the pairs a template
shift can reorder; well-separated classes survive any wording. That is why prompt
sensitivity shows up in practice as confusion between specific confusable pairs
rather than as a uniform drop, and why a per-class error breakdown is the right
way to look for it.

The practical conclusion: a zero-shot classifier needs a validation set as much as
a trained one, not to fit weights but to choose sentences -- and the first thing
to do with it is not to pick the best prompt, it is to stop picking one.""")
