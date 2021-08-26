SHELL := /bin/bash

seq := $(shell echo part-{00000..00010}-of-00500)
# files = $(foreach var, $(seq), part_$(var)_of_00500) 
files = $(seq)

tfrecord/%-of-00500 : results/%-of-00500
	python create_pretraining_data.py \
		   --input_file=$< \
		   --output_file=$@ \
		   --vocab_file=./wiki/vocab.txt \
		   --do_lower_case=True \
		   --max_seq_length=512 \
		   --max_predictions_per_seq=76 \
		   --masked_lm_prob=0.15 \
	       --random_seed=12345 \
		   --dupe_factor=10

all: $(addprefix tfrecord/, $(files))

clean:
	rm -rf tfrecord/*
