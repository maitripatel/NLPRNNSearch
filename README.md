Neural Machine Translation
=====================================================================

### Installation
The following packages are needed:
* [Pytorch](https://github.com/pytorch/pytorch) >= 0.4.0
* NLTK

### Training
Training the RNNSearch on English-French translation datasets as follows:
```
python train.py \
--src_vocab corpus/frr.voc3.pkl --trg_vocab corpus/een.voc3.pkl \
--train_src corpus/train.fr-en.fr --train_trg corpus/train.fr-en.en \
--valid_src corpus/test.fr-en.fr.txt \
--valid_trg corpus/test.fr-en.en.txt \
--eval_script scripts/validate.sh \
--model RNNSearch \
--optim RMSprop \
--batch_size 80 \
--half_epoch \
--cuda \
--info RMSprop-half_epoch
```