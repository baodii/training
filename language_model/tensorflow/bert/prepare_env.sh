# prepare conda env
source activate
conda create -n pre-training python=3.7 --yes
conda activate pre-training
pip install tensorflow-gpu==1.15

# get mlcommom repo and data 
git clone https://github.com/baodii/training.git mlcommon_training
git clone https://github.com/sgpyc/training sgpyc_training
cd mlcommon_training
git checkout bert_pretraining
cd ../sgpyc_training
git checkout bert_fix
cd ../mlcommon_training
cp language_model/tensorflow/bert/pre_*.sh ../sgpyc_training/language_model/tensorflow/bert
cp language_model/tensorflow/bert/cleanup_scripts/create_pre*.sh ../sgpyc_training/language_model/tensorflow/bert/cleanup_scripts
cp language_model/tensorflow/bert/cleanup_scripts/download_and_uncompress.sh ../sgpyc_training/language_model/tensorflow/bert/cleanup_scripts
cp language_model/tensorflow/bert/cleanup_scripts/Makefile ../sgpyc_training/language_model/tensorflow/bert/cleanup_scripts
cd ../sgpyc_training
cd language_model/tensorflow/bert/cleanup_scripts
mkdir tfrecord
pip install gdown
conda deactivate
source activate
conda activate pre-training
source download_and_uncompress.sh
cd wiki/tf1_ckpt
mv model.ckpt-28252.data-00000-of-00001 model.ckpt-28252
cd ../..

# process data
git clone https://github.com/attardi/wikiextractor.git
cd wikiextractor
git checkout 3162bb6c3c9ebd2d15be507aa11d6fa818a454ac
cd ..
python wikiextractor/WikiExtractor.py wiki/enwiki-20200101-pages-articles-multistream.xml

./process_wiki.sh './text/*/wiki_??'

# process pretraining data
pip install absl-py
make -j128
bash create_preeval_data.sh

cd ../../../../../mlcommon_training/language_model/tensorflow/bert/
rm -rf ./cleanup_scripts
ln -s ../../../../sgpyc_training/language_model/tensorflow/bert/cleanup_scripts/ ./cleanup_scripts
# pre-training
# cd ..
git clone https://github.com/mlperf/logging.git mlperf-logging
pip install -e mlperf-logging
bash pre_training.sh