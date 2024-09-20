<div align="center">
  
## Beyond Accuracy Optimization: Computer Vision Losses for Large Language Model Fine-Tuning

[**Daniele Rege Cambrin**](https://darthreca.github.io/)<sup>1</sup> · [**Giuseppe Gallipoli**](https://github.com/gallipoligiuseppe)<sup>1</sup> · [**Irene Benedetto**](https://github.com/irenebenedetto)<sup>1</sup> · [**Luca Cagliero**](https://dbdmg.polito.it/dbdmg_web/people/luca-cagliero/)<sup>1</sup> · [**Paolo Garza**](https://dbdmg.polito.it/dbdmg_web/people/paolo-garza/)<sup>1</sup>

<sup>1</sup>Politecnico di Torino, Italy

**[EMNLP 2024 Findings](https://2024.emnlp.org/)**

<a href="https://arxiv.org/abs/2408.04523"><img src='https://img.shields.io/badge/arXiv-Beyond_Accuracy_Optimization-red' alt='Paper PDF'></a>
</div>

**This study investigates the use of established semantic segmentation loss functions in natural language generation to create a versatile, practical, and scalable solution for fine-tuning different architectures.**
We evaluate their effectiveness in solving Math Word Problems and question answering across different models of varying sizes. For the analyzed tasks, we found that the traditional Cross-Entropy loss represents a sub-optimal choice, while models trained to minimize alternative (task-dependent) losses, such as Focal or Lovász, achieve a mean improvement of +42\% on exact match without requiring additional data or human feedback. These findings suggest a promising pathway for more efficient and accessible training processes.

*REPOSITORY IN CONSTRUCTION SOME FILES COULD BE MISSING*

### Getting Started

Install the dependencies of the *requirements.txt* file. Make sure to edit the config files in the `configs/` folder. Then simply run *improved_loss.py*

With *baseline_inference.py* you can run the baseline models used for comparison.

### Datasets and Models

In this work, we employed the following datasets:

We also compared these models:

## License

This project is licensed under the **Apache 2.0 license**. See [LICENSE](LICENSE) for more information.

## Citation

If you find this project useful, please consider citing:

```bibtex

```
