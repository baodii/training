# Generate one TFRecord for each part-00XXX-of-00500 file. The following command is for generating one corresponding TFRecord
for i in {00000..00499}
do
	python3 create_pretraining_data.py \
		   --input_file=./results/part-${i}-of-00500 \
		   --output_file=./tfrecord/part-${i}-of-00500 \
		   --vocab_file=./wiki/vocab.txt \
		   --do_lower_case=True \
		   --max_seq_length=512 \
		   --max_predictions_per_seq=76 \
		   --masked_lm_prob=0.15 \
	           --random_seed=12345 \
		   --dupe_factor=10
done
