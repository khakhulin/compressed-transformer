import dill as pickle
import matplotlib.pyplot as plt


with open("logs/iwslt-noncompressed", 'rb') as f_in:
    log_parse_full = pickle.load(f_in)

with open("logs/iwslt14_compress", 'rb') as f_in:
    log_parse_comp3 = pickle.load(f_in)

with open("logs/iwslt14_compress5", 'rb') as f_in:
    log_parse_comp5 = pickle.load(f_in)

bleu_log_full = log_parse_full['bleu']
bleu_log_comp3 = log_parse_comp3['bleu']
bleu_log_comp5 = log_parse_comp5['bleu']

loss_full = log_parse_full['loss']
loss_comp3 = log_parse_comp3['loss']
loss_comp5 = log_parse_comp5['loss']

#
# plt.title("BLEU for IWSLT14")
# #
# plt.plot(bleu_log_full, label='uncompressed', c='g')
# plt.plot(bleu_log_comp3, label='compressed N=3', c='r')
# plt.plot(bleu_log_comp5, label='compressed N=5', c='y')

plt.title("Loss IWSLT14")
plt.plot(loss_full, label='uncompressed', c='g')
plt.plot(loss_comp3, label='compressed N=3', c='r')
plt.plot(loss_comp5, label='compressed N=5', c='y')

plt.legend()
plt.xlabel('validation iterations')
plt.ylabel('loss')
plt.show()