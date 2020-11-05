# Keyword detection
### data 
get data from link: https://drive.google.com/file/d/1ala2gdO2LHFHomw3TjWNMvRaBVERgF7K/view?usp=sharing
copy file xeco.csv to data-bin/raw
### Build environment using python env

```bash
pip install -r requirements.txt

# Install coc coc tokenizer 
git clone https://github.com/coccoc/coccoc-tokenizer.git
cd coccoc-tokenizer
mkdir build && cd build
cmake -DBUILD_PYTHON=1 ..
make install
cd ../python
python setup.py install
```

### Build environment using Docker
```bash
docker build -t transfer-keyword .
```

### Get data for training model

- Before building the model, make sure folder './data-bin/raw' contain csv data files. For example:

```bash
docker run -it --rm -v E:/Caohoc/SearchMultiMedia/keyword_detection/data-bin/raw:/usr/src/app/data-bin/raw -d transfer-keyword
```

### Create keyword modeling

```bash
python build_keyword_modeling.py
```

### Infer model

```bash
python infer_keyword_modeling.py
```

