---
id: part-04-intro
status: final
---

## What this part is for

Twelve algorithms, derived rather than described.

These are the methods that still win on tabular data — which is most business
data — and they are also the conceptual foundation for everything after them. A
neural network is a stack of the linear models in {{ch:ml-linear-regression}}
and {{ch:ml-logistic}} with nonlinearities between. Attention is a similarity
computation of the kind {{ch:ml-knn-nb}} uses. The regularisation of
{{ch:ml-metrics}} is what keeps a 70-billion-parameter model from memorising its
training set. Skipping Part IV to get to deep learning produces someone who can
train a transformer and cannot tell whether it is overfitting.

The organising idea is the bias-variance decomposition, introduced in
{{ch:ml-metrics}} before most of the algorithms. Once you have it, the rest of
the part is legible as a set of answers to one question: **how do you get a
model complex enough to capture the pattern without capturing the noise?**
Bagging answers it by averaging away variance. Boosting answers it by
accumulating away bias. Regularisation answers it by shrinking. Trees answer it
by pruning. That framing is why {{ch:ml-metrics}} comes fourth rather than last.

## What is here

- **Chapters 31–34** — the frame. What learning is, the two models whose
  mathematics is fully derivable, and how to tell whether a model is any good.
- **Chapters 35–39** — the supervised algorithms. Instance-based, probabilistic,
  tree-based, ensembled, and margin-based, each with its own inductive bias.
- **Chapters 40–42** — unsupervised. Clustering, dimensionality reduction, and
  anomaly detection, where there is no ground truth and evaluation is genuinely
  harder.

```mermaid {#fig:part4-deps caption="Dependencies within Part IV. Chapter 34 is the hinge: every algorithm after it is discussed in terms of where it sits on the bias-variance trade-off."}
graph LR
  C31[31 · What ML is] --> C32[32 · Linear regression]
  C32 --> C33[33 · Logistic regression]
  C33 --> C34[34 · Metrics & bias-variance]
  C34 --> C35[35 · kNN & naive Bayes]
  C34 --> C36[36 · Trees]
  C36 --> C37[37 · Forests & bagging]
  C37 --> C38[38 · Boosting]
  C34 --> C39[39 · SVM & kernels]
  C35 --> C40[40 · Clustering]
  C40 --> C41[41 · PCA]
  C41 --> C42[42 · Anomaly detection]
  C36 --> C42
```

## Two things worth saying up front

**Every algorithm is an assumption.** There is no method that is best in
general — that is the content of the no-free-lunch result in
{{ch:ml-what-it-is}}. Linear regression assumes the relationship is a plane.
k-nearest neighbours assumes nearby points behave alike. Trees assume the
boundary is axis-aligned. Choosing an algorithm is choosing which assumption you
are willing to make about your data, and the chapters therefore spend as much
time on when each method fails as on how it works.

**The classical methods have not been superseded on tabular data.** This
surprises people arriving from the deep-learning literature. Gradient boosting
has held the top position on tabular problems for two decades, and
{{cite:grinsztajn2022}} identifies why in terms of concrete properties: trees
are robust to uninformative features, are unaffected by feature rotation, and
fit irregular functions that smooth models cannot.

That said, the position is genuinely moving for the first time.
{{cite:hollmann2025}} reports a tabular foundation model beating heavily-tuned
ensembles on datasets up to around ten thousand rows, in seconds rather than
hours, by in-context learning rather than fitting. {{ch:ml-boosting}} treats this
honestly: gradient boosting remains {{maturity:ESTABLISHED}} at medium and large
scale, and tabular foundation models are {{maturity:EMERGING}} and already the
better choice on small data. The *reasons* behind the trade-off are properties
of data rather than of the year, which is why that section should still be
useful when the recommendation flips.

## A note on implementation

Every important algorithm here is implemented from scratch before any library is
called, and the from-scratch version is checked against scikit-learn
{{cite:pedregosa2011}} where one exists. This is not ceremony. A gradient
boosting library is four hyperparameters and a fit method; the from-scratch
version is the one where you can see that each tree is fitted to the residual of
the last, which is the only thing that makes the hyperparameters meaningful.

The library versions are what you should ship. The scratch versions are what
make the library versions debuggable.

## What this part deliberately does not cover

VC theory and PAC bounds, referenced where generalisation guarantees come from
and not developed. Kernel theory beyond the trick itself. Manifold learning —
t-SNE and UMAP are named with their central caveat and left as visualisation
tools. Bayesian methods beyond naive Bayes. Each is a coherent body of material
that nothing later in this book requires.

## What you should be able to do at the end

Derive least squares from the projection argument and explain why the normal
equations are not how you should compute it. Derive the logistic gradient and
explain why the sigmoid pairs with cross-entropy. Decompose an error into bias,
variance and noise, and say which lever moves which. Read a learning curve and
diagnose the problem. Explain why bagging reduces variance and boosting reduces
bias. Implement a decision tree, a random forest and gradient boosting from
scratch. Explain the kernel trick without hand-waving. Cluster data and be
honest about whether the clusters mean anything. Apply PCA and say what the
components are. Choose an algorithm from the shape of the problem and defend the
choice.

The assignment at the end requires most of that on one dataset.
