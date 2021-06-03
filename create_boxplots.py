import matplotlib.pyplot as plt
import numpy as np

#        [original_simple, original_simpleGRU, original_seq2seq]
#        [new_simple, new_simpleGRU, new_seq2seq]
data_a = [[0.7853-0.9449,0.7794-0.9233,0.7668-0.9151,0.8004-0.9385,0.7683-0.9799,0.7929-0.9228,0.7878-0.9516,0.7742-0.9678,0.7816-0.8977,0.7658-0.9303], [0.4677-0.6175,0.4880-0.5885,0.4863-0.5740,0.4829-0.5751,0.4828-0.5901,0.4674-0.5969,0.4811-0.6161,0.4731-0.6193,0.4688-0.5920,0.5039-0.5664], [0.4246-0.4275,0.4204-0.4270,0.4341-0.4310,0.4127-0.4293,0.4032-0.4165,0.4207-0.4108,0.4236-0.4541,0.4174-0.4272,0.4290-0.4210,0.3943-0.3918]]
data_b = [[0.6904-0.7975,0.7053-0.7806,0.7149-0.7964,0.7048-0.7794,0.7363-0.7888,0.7195-0.8115,0.6984-0.7793,0.6951-0.7913,0.7096-0.8120,0.7213-0.7697], [0.3702-0.4204,0.4158-0.4136,0.3859-0.4028,0.3826-0.3959,0.4193-0.4174,0.3723-0.4060,0.3686-0.4204,0.3634-0.3954,0.3944-0.4222,0.3754-0.3979], [0.3187-0.2985,0.3195-0.2930,0.3285-0.2882,0.3219-0.2909,0.3209-0.2942,0.3256-0.2887,0.3176-0.2850,0.3339-0.2927,0.3242-0.2962,0.3158-0.2904]]

ticks = ['Feed-forward', 'GRU', 'Seq2seq']


def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

plt.figure()

bpl = plt.boxplot(data_a, positions=np.array(range(len(data_a)))*2.0-0.4, sym='', widths=0.6)
bpr = plt.boxplot(data_b, positions=np.array(range(len(data_b)))*2.0+0.4, sym='', widths=0.6)
set_box_color(bpl, '#D7191C')  # colors are from http://colorbrewer2.org/
set_box_color(bpr, '#2C7BB6')

# draw temporary red and blue lines and use them to create a legend
plt.plot([], c='#D7191C', label='Vanilla configuration')
plt.plot([], c='#2C7BB6', label='SOTA configuration')
plt.legend()
plt.xlabel('Model')
plt.ylabel('SSL performance')
plt.title("Vanilla vs SOTA configuration (25k data, 10 simulations for each)")

plt.xticks(range(0, len(ticks) * 2, 2), ticks)
plt.xlim(-2, len(ticks)*2)
plt.ylim(-0.2, 0.1)
plt.tight_layout()
plt.savefig('boxcompare.png')
