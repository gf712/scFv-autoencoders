# scFv-autoencoders
This is the repo with a PhD project I have been working on to create an embedding for [scFv](https://en.wikipedia.org/wiki/Single-chain_variable_fragment) sequences using [autoencoders](https://en.wikipedia.org/wiki/Autoencoder).
### Models:
- all the models are in the `models/` directory
- they are all written in python using [Keras](https://keras.io/) and [tensorflow](https://www.tensorflow.org/)
- the scripts were written to run on local GPUs (usage shown in jupyter notebooks in `notebooks/` directory)
- there are two main model types in this project:
    - multi input and output autoencoders
        - the autoencoder takes **two** inputs: VH and VL sequences
        - the autoencoder outputs **two** sequences, VH and VL
        - the rationale is that each domain has it's own local features which together interact with the local features of the other domain
    - single input and output autoencoders
        - the autoencoder takes in a concatenated VH/VL sequence and tries to reproduce it
        - in this case the autoencoder does not know that it has two domains, but with LSTMs it might be able to find local features specific to each domain, which are then combined in the hidden layers
- each model is being trialed with different architectures: LSTMs, GRUs, simple RNNs, bidirectional RNNs
- early results show that adding 1D convolutions doesn't help, but it will be tried later on again

### What is the end goal?
The task here is to find an embedding that can represent any scFv sequence in a lower dimensional space, i.e. from about 300 amino acids (timesteps) to 2 dimensions (possibly more). Two dimensions is ideal as it is easy to visualise the projection of each sequence with scatter plots and clusters are easy to identify. However, higher dimensions allow the model to encode sequences with a lower information loss.
The embedding can then be fed to another algorithm, i.e. SVM, ANN, Random Forest, etc. to predict certain features. In my work I am specifically looking at predicting scFv thermostability.

### Example:
- AutoencoderV6:

![Alt text](images/autoencoderV6.png?raw=true "Autoencoder")

- Scatter plot looking at germlines:

The autoencoder projects the sequences into clusters corresponding to their germlines, without any sequence alignment.
![Alt text](images/autoencoderV6_germline.png?raw=true "Scatter plot")