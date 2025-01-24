<div align="center">
  
## Beyond Accuracy Optimization: Computer Vision Losses for Large Language Model Fine-Tuning

[**Daniele Rege Cambrin**](https://darthreca.github.io/)<sup>1</sup> · [**Giuseppe Gallipoli**](https://github.com/gallipoligiuseppe)<sup>1</sup> · [**Irene Benedetto**](https://github.com/irenebenedetto)<sup>1</sup> · [**Luca Cagliero**](https://dbdmg.polito.it/dbdmg_web/people/luca-cagliero/)<sup>1</sup> · [**Paolo Garza**](https://dbdmg.polito.it/dbdmg_web/people/paolo-garza/)<sup>1</sup>

<sup>1</sup>Politecnico di Torino, Italy

**[EMNLP 2024 Findings](https://2024.emnlp.org/)**

<a href="https://arxiv.org/abs/2409.13641"><img src='https://img.shields.io/badge/ArXiv-Beyond_Accuracy_Optimization-red' alt='Paper PDF'></a>
<a href="https://github.com/DarthReca/segmentation-losses-nlp/blob/main/loss/focal_loss.py"><img src='https://img.shields.io/badge/Loss-Focal-blue' alt='Focal'></a>
<a href="https://github.com/DarthReca/segmentation-losses-nlp/blob/main/loss/dice_nlp_loss.py"><img src='https://img.shields.io/badge/Loss-SADL-blue' alt='Focal'></a>
<a href="https://github.com/DarthReca/segmentation-losses-nlp/blob/main/loss/dice_loss.py"><img src='https://img.shields.io/badge/Loss-GDice-blue' alt='Focal'></a>
<a href="https://github.com/DarthReca/segmentation-losses-nlp/blob/main/loss/lovasz.py"><img src='https://img.shields.io/badge/Loss-Lovasz-blue' alt='Focal'></a>
</div>

**This study investigates the use of established semantic segmentation loss functions in natural language generation to create a versatile, practical, and scalable solution for fine-tuning different architectures.**
We evaluate their effectiveness in solving Math Word Problems and question answering across different models of varying sizes. For the analyzed tasks, we found that the traditional Cross-Entropy loss represents a sub-optimal choice, while models trained to minimize alternative (task-dependent) losses, such as Focal or Lovász, achieve a mean improvement of +42\% on exact match without requiring additional data or human feedback. These findings suggest a promising pathway for more efficient and accessible training processes.

*REPOSITORY IN CONSTRUCTION SOME FILES COULD BE MISSING*

## Getting Started

Install the dependencies of the *requirements.txt* file. Make sure to edit the config files in the `configs/` folder. Then simply run *improved_loss.py*

With *baseline_inference.py*, you can run the baseline models used for comparison.

*WORKING ON SIMPLIFY THE TRAINING*

## Resources
This section summarizes all models, datasets, and losses we employed during training.

### Dataset
*SOON*

### Base Models
This is the list of the base models used for the finetuning. 
They are only pre-trained on a list of known datasets (generally in the report) if the *Pre-Training Dataset* is *Well-Defined* in Table.
This was done to avoid any overlapping with the finetuning data.

|                   | Size | License      | Pre-Training Dataset | Link |
|-------------------|------|--------------|----------------------|------|
| RedPajama-Incite  | 3B   | Apache 2.0   | Well-Defined         | [link](https://huggingface.co/togethercomputer/RedPajama-INCITE-Base-3B-v1) |
| StableLM          | 3B   | CC BY-SA-4.0 | Well-Defined         | [link](https://huggingface.co/stabilityai/stablelm-3b-4e1t) |
| RedPajama-Incite  | 7B   | Apache 2.0   | Well-Defined         | [link](https://huggingface.co/togethercomputer/RedPajama-INCITE-7B-Base) |
| Falcon            | 7B   | Apache 2.0   | Well-Defined (90%)   | [link](https://huggingface.co/tiiuae/falcon-7b) |
| Llama-2           | 7B   | Llama-2      | Public               | [link](https://huggingface.co/meta-llama/Llama-2-7b-hf) |

### Losses
These are the losses analyzed in the paper and the original papers (read them to understand better how they work).
You can find the code for the losses in this repository in the *loss* folder (and the licenses in *loss_licenses* folder).
The *Type* taxonomy follows the one proposed by [Jun Ma](https://arxiv.org/abs/2005.13449).

| Loss                     |     Type     | Link |
|--------------------------|:------------:|:----:|
| Cross-Entropy Loss       | Distribution |   -  |
| Focal Loss               | Distribution | [link](https://arxiv.org/abs/1708.02002) |
| Generalized Dice Loss    |    Region    | [link](https://arxiv.org/abs/1707.03237) |
| Self-Adjusting Dice Loss |     Combo    | [link](https://arxiv.org/abs/1911.02855) |
| Lovasz Loss              |    Region    | [link](https://arxiv.org/abs/1705.08790) |

## License

This project is licensed under the **Apache 2.0 license**. See [LICENSE](LICENSE) for more information.

## Citation

If you find this project useful, please consider citing:

```bibtex
@inproceedings{Rege_Cambrin_2024,
   title={Beyond Accuracy Optimization: Computer Vision Losses for Large Language Model Fine-Tuning},
   url={http://dx.doi.org/10.18653/v1/2024.findings-emnlp.704},
   DOI={10.18653/v1/2024.findings-emnlp.704},
   booktitle={Findings of the Association for Computational Linguistics: EMNLP 2024},
   publisher={Association for Computational Linguistics},
   author={Rege Cambrin, Daniele and Gallipoli, Giuseppe and Benedetto, Irene and Cagliero, Luca and Garza, Paolo},
   year={2024},
   pages={12060–12079}
}
```
