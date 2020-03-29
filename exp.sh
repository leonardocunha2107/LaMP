python main.py -dataset rcv1 -batch_size 32 -d_model 512 -d_inner_hid 512 -n_layers_enc 2 -n_layers_dec 2 -n_head 4 -epoch 2 -dropout 0.2 -dec_dropout 0.2 -lr 0.0002\
 -encoder 'graph' -decoder 'graph' -label_mask 'inveye' -exo_name p1_eminus3 -ploss 1 -p_coef 1e-3

python main.py -dataset rcv1 -batch_size 32 -d_model 512 -d_inner_hid 512 -n_layers_enc 2 -n_layers_dec 2 -n_head 4 -epoch 2 -dropout 0.2 -dec_dropout 0.2 -lr 0.0002\
 -encoder 'graph' -decoder 'graph' -label_mask 'inveye' -exo_name p1_eminus5 -ploss 1 -p_coef 1e-5

python main.py -dataset rcv1 -batch_size 32 -d_model 512 -d_inner_hid 512 -n_layers_enc 2 -n_layers_dec 2 -n_head 4 -epoch 2 -dropout 0.2 -dec_dropout 0.2 -lr 0.0002\
 -encoder 'graph' -decoder 'graph' -label_mask 'inveye' -exo_name pdot9_eminus3 -ploss 0.9 -p_coef 1e-3
 
python main.py -dataset rcv1 -batch_size 32 -d_model 512 -d_inner_hid 512 -n_layers_enc 2 -n_layers_dec 2 -n_head 4 -epoch 2 -dropout 0.2 -dec_dropout 0.2 -lr 0.0002\
 -encoder 'graph' -decoder 'graph' -label_mask 'inveye' -exo_name pdot9_eminus5 -ploss 0.9 -p_coef 1e-5
 
