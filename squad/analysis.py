import matplotlib.pyplot as plt
import numpy as np

def analysis(gold_dict, pred_dict):
    fig, ax = plt.subplots()
    count_dict = dict([(i, []) for i in range(26)])
    for key, value in pred_dict.items():
        if len(gold_dict[key]['answers']) > 0:
            ground_truth = gold_dict[key]['answers'][0].split(' ')
        else:
            ground_truth = ''
        if len(value) > 0:
            prediction = value.split(' ')
        else:
            prediction = ''
        
        if len(ground_truth) <= 25:
            count_dict[len(ground_truth)].append(len(prediction))

    plot_x_list = []
    plot_y_list = []
    plot_bar_list = []
    for l in sorted(count_dict):
        if len(count_dict[l]) > 0:
            plot_bar_list.append(len(count_dict[l]))
            plot_x_list.append(l)
            plot_y_list.append(np.mean(count_dict[l]))

    ax.bar(plot_x_list, plot_bar_list, color='tab:blue')
    ax.set_xlabel('Ground Truth Length')
    ax.set_ylabel('Ground Truth Length Freq.', color='tab:blue')
    ax.tick_params(axis='y', labelcolor='tab:blue')
    
    ax2 = ax.twinx()
    ax2.plot([0, 25], [0, 25], linestyle='dashed', color='tab:red', label='Accuracy Reference')
    ax2.plot(plot_x_list, plot_y_list, color='tab:red')
    ax2.set_xlim(-0.5, 25)
    ax2.set_ylim(0, 25)
    ax2.set_ylabel('Mean Prediction Length', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    # ax2.legend()

    fig.tight_layout()
    fig.savefig('./new_length_analysis.png')
    print('Length analysis saved')
