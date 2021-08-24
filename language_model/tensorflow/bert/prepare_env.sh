# prepare conda env
conda activate
conda create -n pre-training python=3.7 --yes
conda activate pre-training
pip install tensorflow-gpu==1.15

# get mlcommom repo and data 
git clone https://github.com/baodii/training.git mlcommon_training
git clone https://github.com/sgpyc/training sgpyc_training
cd mlcommon_training
git checkout bert_pretraining
cp language_model/tensorflow/bert/pre_*.sh ../sgpyc_training/language_model/tensorflow/bert
cp language_model/tensorflow/bert/cleanup_scripts/create_pre*.sh ../sgpyc_training/language_model/tensorflow/bert/cleanup_scripts
cd ../sgpyc_training
git checkout bert_fix
cd language_model/tensorflow/bert/cleanup_scripts
mkdir tfrecord
source download_and_umcompress.sh
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
bash create_pretraining_data.sh
bash create_preeval_data.sh

# pre-training
cd ..
bash pre_training.sh