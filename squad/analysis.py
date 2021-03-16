import matplotlib.pyplot as plt
import numpy as np

def analysis(gold_dict, pred_dict):
    fig, ax = plt.subplots()
    count_mat = np.zeros((26, 26))
    for key, value in pred_dict.items():
        if len(gold_dict[key]['answers']) > 0:
            ground_truth = gold_dict[key]['answers'][0].split(' ')
        else:
            ground_truth = ''
        if len(value) > 0:
            prediction = value.split(' ')
        else:
            prediction = ''
        
        if len(ground_truth) <= 25 and len(ground_truth) > 0 \
            and len(prediction) <= 25 and len(prediction) > 0:
            count_mat[len(ground_truth), len(prediction)] += 1

    # count_mat = np.flipud(count_mat)
    plt.imshow(count_mat)
    plt.gca().invert_yaxis()
    plt.colorbar()
    # ax.plot([0.5, 25], [0.5, 25], color='black')
    ax.set_xlim(0, 25)
    ax.set_ylim(0, 25)
    ax.set_xlabel('Ground Truth Length')
    ax.set_ylabel('Prediction Length')
    ax.set_aspect('equal', 'box')
    fig.savefig('./length_analysis.png')
