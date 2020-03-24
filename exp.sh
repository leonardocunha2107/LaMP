python main.py -dataset reuters -batch_size 32 -d_model 512 -d_inner_hid 512 -n_layers_enc 2 \
-n_layers_dec 2 -n_head 4 -epoch 2 -dropout 0.2 -dec_dropout 0.2 -lr 0.0002 -encoder 'graph' -decoder 'graph' -label_mask 'prior'

