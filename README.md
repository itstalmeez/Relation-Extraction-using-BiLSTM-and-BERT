**Relation Extraction using BiLSTM and BERT**
This repository contains PyTorch implementations of relation extraction models using Bidirectional Long Short-Term Memory (BiLSTM) and BERT (Bidirectional Encoder Representations from Transformers). The models are trained on a dataset with annotated sentences and their corresponding relations.

**Dataset**
The dataset used for training and evaluation is stored in JSON format, with each data point having a "sentence" and a "relation" field. The relation field indicates the relationship between entities in the sentence.

**Models**
BiLSTM Model
The BiLSTM model employs a bidirectional LSTM architecture to capture contextual information from sentences. The model is trained to predict relations based on tokenized input sentences.

**BERT Model**
The BERT model utilizes the pre-trained BERT (bert-base-uncased) architecture for relation extraction. It leverages the transformer-based architecture to understand contextual embeddings and predict relations.

**Contributions**
Contributions are welcome! Feel free to open issues for bug reports or new feature suggestions. Pull requests are also encouraged.
