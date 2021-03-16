import matplotlib.pyplot as plt

def analysis(gold_dict, pred_dict):
    fig, ax = plt.subplots()
    for key, value in pred_dict.items():
        if len(gold_dict[key]['answers']) > 0:
            ground_truth = gold_dict[key]['answers'][0].split(' ')
        else:
            ground_truth = ''
        if len(value) > 0:
            prediction = value.split(' ')
        else:
            prediction = ''
        # print(ground_truth, len(ground_truth), '|', prediction, len(prediction))
        ax.scatter(len(ground_truth), len(prediction), color='red', s=2)
    ax.plot([0, 25], [0, 25], color='black')
    ax.set_xlim(0, 25)
    ax.set_ylim(0, 25)
    ax.set_xlabel('Ground Truth Length')
    ax.set_ylabel('Prediction Length')
    ax.set_aspect('equal', 'box')
    fig.savefig('./length_analysis.png')
