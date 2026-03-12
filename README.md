# Towards a Multimodal Framework for Remote Sensing Image Change Retrieval and Captioning (RSICRC)

<div align="center">
  <a href="https://arxiv.org/abs/2406.13424">
    <img src="https://img.shields.io/badge/Paper-arXiv-red" alt="Paper">
  </a>
  <a href="https://github.com/rogerferrod/RSICRC">
    <img src="https://img.shields.io/badge/Code-GitHub-black" alt="GitHub">
  </a>
  <a href="https://huggingface.co/RogerFerrod/RSICRC">
    <img src="https://img.shields.io/badge/Model-HuggingFace-yellow" alt="Hugging Face">
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/Conference-DS%202024-blue" alt="Discovery Science 2024">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  </a>
</div>

<br>

> **Towards a Multimodal Framework for Remote Sensing Image Change Retrieval and Captioning**
> 
> *Roger Ferrod, Luigi Di Caro and Dino Ienco*
>
> Published in **Discovery Science (DS) 2024**

---

## 📌 Overview

Earth observation systems provide a continuous stream of satellite imagery to monitor evolving landscapes. However, most current Vision-Language models in the Remote Sensing (RS) domain are focused on static data, lacking the ability to account for temporal changes between multiple observations. 

To address this gap, we introduce **RSICRC**, a novel multimodal model designed specifically for bi-temporal remote sensing image pairs. RSICRC jointly handles two tasks:
1. **Change Detection Captioning**: Accurately describing the changes between two timestamps in natural language.
2. **Bi-temporal Text-Image Retrieval**: Retrieving the correct before/after image pair based on a user's textual query.

By utilizing contrastive learning on the LEVIR-CC dataset, our model successfully bridges these paradigms, unlocking text-image retrieval capabilities while preserving state-of-the-art captioning performance.

---

## 🏗️ Architecture & Key Innovations

Our architecture is inspired by CoCa but explicitly adapted to handle bi-temporal image pairs. 

* **Pretrained Remote Sensing Backbone**: We utilize a ResNet-50 backbone fine-tuned from RemoteCLIP to independently extract features from the *before* and *after* images. 
* **Bi-temporal Encoder**: A hierarchical self-attention block combines the temporal features into a single, unified visual representation.
* **Decoupled Decoder**: The transformer-based decoder is split into two parts: an unimodal module that encodes textual queries for contrastive learning and a multimodal cross-attention module that generates the final caption.
* **False Negative Attraction (FNA)**: Adapting a captioning dataset (LEVIR-CC) for contrastive retrieval introduces "False Negatives" (different pairs with semantically identical captions). We implement a False Negative Attraction strategy, which dynamically compares caption embeddings and treats semantically similar false negatives as positive examples, significantly improving retrieval performance.

---

## 📖 Citation
If you find our work or the pre-trained weights helpful for your research, please consider citing our Discovery Science 2024 paper:
```text
@inproceedings{rsicrc,
  title={Towards a Multimodal Framework for Remote Sensing Image Change Retrieval and Captioning},
  author={Ferrod, Roger and Di Caro, Luigi and Ienco, Dino},
  booktitle={International Conference on Discovery Science},
  year={2024}
}
```