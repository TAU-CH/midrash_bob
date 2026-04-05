<div align="center">

<h1>Bag of Bags: Adaptive Visual Vocabularies for Genizah Join Image Retrieval</h1>

[Sharva Gogawale](),
[Gal Grudka](),
[Daria Vasyutinsky-Shapira](),
[Omer Ventura](),
[Berat Kurar-Barakat](),
[Nachum Dershowitz]()

**School of Computer Science and AI, Tel Aviv University, Ramat Aviv, Israel**

</div>

## Abstract

A join is a set of manuscript fragments identified as originally emanating from the same manuscript. We study manuscript join retrieval: Given a query image of a fragment, retrieve other fragments originating from the same physical manuscript. We propose **Bag of Bags (BoB)**, an image-level representation that replaces the global-level visual codebook of classical Bag of Words (BoW) with a fragment-specific vocabulary of local visual words. Our pipeline trains a sparse convolutional autoencoder on binarized fragment patches, encodes connected components from each page, clusters the resulting embeddings with per-image k-means, and compares images using set-to-set distances between their local vocabularies. Evaluated on fragments from the Cairo Genizah, the best BoB variant (viz. Chamfer) achieves Hit@1 of 0.78 and MRR of 0.84, compared to 0.74 and 0.80, respectively, for the strongest BoW baseline (BoW-RawPatches-χ²), a 6.1% relative improvement in top-1 accuracy. We furthermore study a mass-weighted BoB-OT variant that incorporates cluster population into prototype matching, and present a formal approximation guarantee bounding its deviation from full component-level optimal transport. A two-stage pipeline using a BoW shortlist followed by BoB-OT reranking provides a practical compromise between retrieval strength and computational cost, supporting applicability to larger manuscript collections.

## Installation

*To be updated...*

## Getting Started

*To be updated...*

## Data

*To be updated...*

## Citation

If you use this work in your research, please cite:

```bibtex
@inproceedings{gogawale2026bob,
  title={Bag of Bags: Adaptive Visual Vocabularies for Genizah Join Image Retrieval},
  author={Gogawale, Sharva and Grudka, Gal and Vasyutinsky-Shapira, Daria and Ventura, Omer and Kurar-Barakat, Berat and Dershowitz, Nachum},
  year={2026}
}
```
