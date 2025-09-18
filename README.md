# Music Genre Classifier & Semantic Recommender
This project combines audio classification and semantic search to build an intelligent music genre analysis and recommendation pipeline.

## Overview
Fine-tuned a pre-trained Vision Transformer (ViT) on the [ccmusic-database/music_genre](https://huggingface.co/datasets/ccmusic-database/music_genre) dataset using Google Colab. You can download and use it [here](https://huggingface.co/vaveyko/vit-music-genre). 

## Metrics:
* Accuracy: 90%
* F1 Macro: 91%
* Macro Recall: 91%
* Macro Precision: 92%

## Pipeline
1. Audio Classification Each audio sample is classified into its genre using the fine-tuned ViT model.
2. Semantic Embedding The predicted genre and track name are passed through a SentenceTransformer model to generate semantic embeddings.
3. Similarity Search These embeddings are compared against a custom dataset of over 60,000 entries. The system retrieves the top 5 most semantically similar tracks. 

## Technologies
* Vision Transformer (ViT)
* SentenceTransformer
* Google Colab
* Python, PyTorch
* Hugging Face Datasets