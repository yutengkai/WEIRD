# Detailed Discussion on Topological Data Analysis (TDA) in Transformer-based Language Models

## Introduction

- We started our discussion by exploring the idea of using a pre-trained BERT model as an encoder and training a decoder to recover the original input text from the encoded representations. This led us to the topic of the "word embedding inversion problem", which is a challenging task due to the loss of information during encoding.

## Why We Need TDA in NLP

- Transformer-based models like BERT, BART, T5, and Pegasus have been very successful in various NLP tasks. However, understanding the internal workings of these models remains a challenge. 

- The internal representations (embeddings) of these models are high-dimensional vectors, which are difficult to interpret directly. 

- TDA provides a way to study the "shape" of the data in these high-dimensional spaces. It can help us understand the structure and connectivity of the embeddings, which can provide insights into how the model is processing and understanding the input text.

## Difficulties in Contrastive or Self-Contrastive Learning

- Contrastive learning is a powerful technique for learning useful representations by encouraging the model to produce similar embeddings for "similar" inputs and different embeddings for "different" inputs.

- However, defining what constitutes "similar" and "different" inputs in the context of NLP can be challenging. For example, two sentences can have very different wordings but express the same meaning, and vice versa.

- Furthermore, the discrete nature of text data makes it difficult to apply techniques like data augmentation, which are commonly used in contrastive learning for continuous data like images.

- Self-contrastive learning, where the model learns to distinguish between its own outputs, can be a way to overcome these challenges. However, this approach requires careful design to ensure that the model is learning meaningful distinctions.

## TDA in Transformer-based Models

- The embeddings produced by transformer-based models can be thought of as points in a high-dimensional space. The "valid" embeddings (those that correspond to coherent outputs) may form a complex shape in this space.

- TDA can be used to study this shape, which can provide insights into the model's understanding of the input text. For example, it can help us identify clusters of similar embeddings, or detect "holes" or "gaps" in the space of valid embeddings.

- However, applying TDA to these models presents several challenges. The high dimensionality of the embeddings makes them difficult to visualize and analyze. Furthermore, the embeddings are affected by various factors such as the model's architecture, the training data, and the specific task at hand.

- Despite these challenges, TDA has the potential to provide valuable insights into the workings of transformer-based models, and can be a useful tool for researchers in the field.

## Our Thoughts and Assumptions

- We proposed the idea of using a pre-trained BERT model as a teacher model and a pre-trained BART model as a student model, with the goal of forcing the BART model to produce the same encodings as the BERT model for the same input text.

- We also suggested the possibility of using the weights of a fine-tuned BERT model in different subclasses of BERT for different tasks.

- We expressed interest in understanding the importance of positional encoding in transformer-based models and how it affects the model's ability to understand the order of words in a sentence.

- We proposed the idea of manipulating the input data in some way, such as adding noise, to make the final embeddings fall into both the true and false clusters. We also mentioned the possibility of using topological data analysis to understand the shape of the valid embedding region.

## Questions Raised

- We asked about the possibility of using the same weights from a fine-tuned BERT model in different subclasses of BERT for different tasks.

- We asked about the importance of positional encoding in transformer-based models and how it affects the model's ability to understand the order of words in a sentence.

- We asked about the possibility of manipulating the input data in some way, such as adding noise, to make the final embeddings fall into both the true and false clusters.

- We asked about the possibility of using topological data analysis to understand the shape of the valid embedding region in the internal layers of a transformer-based model.

- We asked about the possibility of using finite samples to make topological analyses about the regions in a Euclidean space or some other kind of space.

- We asked about the possibility of making some conclusions about topological properties when the dataset is big enough, and whether there are existing researches on this topic.

- We asked about the possibility of recording the entire discussion into a markdown file for future reference.

## Future Directions

- We expressed interest in exploring the use of topological data analysis in transformer-based models, and in understanding how to generate valid and invalid embeddings.

- We also expressed interest in understanding whether the valid embedding region is continuous or discontinuous, and what its shape might be.

- We proposed the idea of using a small model like BERT-base-uncased for our experiments, and expressed interest in understanding how to generate valid and invalid embeddings.

- We also proposed the idea of using contrastive learning or self-contrastive learning to manipulate the input data in some way, such as adding noise, to make the final embeddings fall into both the true and false clusters.

- We expressed interest in understanding whether it is possible to make some conclusions about topological properties when the dataset is big enough, and whether there are existing researches on this topic.
