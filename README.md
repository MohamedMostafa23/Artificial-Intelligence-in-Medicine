# Artificial Intelligence in Medicine I & II

Practical assignments, Jupyter notebooks, and reports for the **Artificial Intelligence in Medicine** and **Artificial Intelligence in Medicine II** courses at the **Technical University of Munich (TUM)**.

The material covers machine learning for medical imaging and clinical text, from classification, segmentation, and transfer learning to trustworthy AI, generative models, MRI reconstruction, and survival analysis. This repository brings together the original course collection and the material from [Artificial Intelligence in Medicine II](https://github.com/MohamedMostafa23/Artificial-Intelligence-in-Medicine-II).

## Artificial Intelligence in Medicine I

| Topic | Focus | Materials |
| --- | --- | --- |
| [Classification](Classification/) | Breast ultrasound and chest nodule classification with 2D and 3D convolutional neural networks. | [Notebook](Classification/AI-in-Med-EX1.ipynb) |
| [Segmentation](Segmentation/) | Brain tissue segmentation with unsupervised methods and deep supervised learning. | [Notebook](Segmentation/AI-in-Med-EX2-segmentation_main.ipynb), [PDF report](Segmentation/AI-in-Med-EX2-Report.pdf) |
| [Transfer Learning](Transfer%20Learning/) | Learning with limited data, autoencoders, and transfer learning with frozen and trainable layers. | [Notebook](Transfer%20Learning/AI-in-Med-EX3-transferlearning.ipynb), [PDF report](Transfer%20Learning/AI-in-Med-EX3-Report.pdf) |
| [Trustworthy ML](Trustworthy%20ML/) | Model explanations using class activation maps (CAMs) and uncertainty estimation with Monte Carlo dropout. | [CAM notebook](Trustworthy%20ML/task%201%20-%20AI-in-Med-EX4-Report_Trustworthy_AI_CAM_based_explanations.ipynb), [MC dropout notebook](Trustworthy%20ML/task%202%20-%20MC_DropOut.ipynb), [Word report](Trustworthy%20ML/AI-in-Med-EX4-Report.docx) |
| [Trustworthy AI: Privacy, Security, and Attacks](Trustworthy%20AI%20-%20Privacy%2C%20security%20and%20attacks/) | Federated and privacy-preserving machine learning. | [PDF report](Trustworthy%20AI%20-%20Privacy%2C%20security%20and%20attacks/AI-in-Med-EX5-Report.pdf) |

## Artificial Intelligence in Medicine II

| Topic | Focus | Materials |
| --- | --- | --- |
| [Natural Language Processing](NLP/) | Clinical text preprocessing, vocabulary construction, text classification with pretrained language models, and text generation. | [Notebook](NLP/go56cuh_ge94naz_ge83muj_ge83cak_AIMed2_NLP.ipynb) |
| [Deep Generative Models](Deep%20Generative%20Models/) | Brain MRI analysis with variational autoencoders (VAEs), generative adversarial networks (GANs), and diffusion models. | [Notebook](Deep%20Generative%20Models/go56cuh_ge94naz_ge83muj_ge83cak_AIMed2_GEN.ipynb), [PDF report](Deep%20Generative%20Models/go56cuh_ge94naz_ge83muj_ge83cak_AIMed2_GEN.pdf) |
| [Inverse Problems](Inverse%20Problems/) | MRI reconstruction from k-space, coil compression and sensitivity maps, undersampling, and regularized reconstruction. | [Notebook](Inverse%20Problems/Exercise_original_%281%29.ipynb) |
| [Survival Analysis](Survival%20Analysis/) | Survival analysis assignment. | [PDF report](Survival%20Analysis/ge83muj_ge83cak_go56cuh_ge94naz_survival_analysis_assignment.pdf) |

## Working with the notebooks

1. Clone the repository:

   ```bash
   git clone https://github.com/MohamedMostafa23/Artificial-Intelligence-in-Medicine.git
   cd Artificial-Intelligence-in-Medicine
   ```

2. Open the relevant `.ipynb` file in JupyterLab, Jupyter Notebook, or another compatible editor.
3. Read the notebook's setup and data-loading cells before running it. Install the dependencies required for that assignment and obtain the referenced datasets, supporting code, and pretrained weights where needed.
4. Adjust data and output paths to match your environment, then work through the notebook in order. Accompanying reports contain written answers and results where provided.

These are course assignment notebooks with assignment-specific dependencies; the repository does not include a shared environment specification or bundled datasets. Libraries used across the notebooks include PyTorch, torchvision, NumPy, scikit-learn, Matplotlib, MedMNIST, MONAI, Hugging Face Transformers, and MRI reconstruction tools. Consult each notebook for its particular requirements.

## Contributors

### Artificial Intelligence in Medicine I

- Mohammed Elbushnaq
- Muhammed Elmasry
- Evangelos Fragkiadakis
- Nada Elsherbeny

### Artificial Intelligence in Medicine II

- Mohammed Elbushnaq
- Marina Aoki
- Sama Elbaroudy
- Sarah Berbuir

## Course resources

- [TUM Moodle](https://www.moodle.tum.de/): Course announcements, additional materials, and discussions, subject to course access.
- [TUM Library](https://www.ub.tum.de/en): Academic journals, books, and research databases.
