### Setup
* Install python3 requirements: `pip3 install -r requirements.txt`
* Initialize GloVe (global vectors for word representation) as follows:
```bash
wget https://github.com/explosion/spacy-models/releases/download/en_vectors_web_lg-2.3.0/en_vectors_web_lg-2.3.0.tar.gz -O en_vectors_web_lg-2.3.0.tar.gz
pip install en_vectors_web_lg-2.3.0.tar.gz
```

For overfitting a single batch, set the `--DEBUG` flag to `True`

Other important parameters:
1. `--RESUME`, start training with saved checkpoint. You should assign checkpoint version `--VERSION int` and resumed epoch `--CKTP_E int`.
2. `--VERSION`, to assign seed and version of the model.
3. `--CPU`, to force training and/or inference using CPU.