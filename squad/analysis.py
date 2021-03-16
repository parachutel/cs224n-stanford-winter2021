import matplotlib.pyplot as plt

def analysis(gold_dict, pred_dict):
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
        plt.scatter(len(ground_truth), len(prediction), color='red', s=2)
    plt.plot([0, 25], [0, 25], color='black')
    plt.xlim(0, 25)
    plt.ylim(0, 25)
    plt.xlabel('Ground Truth Length')
    plt.ylabel('Prediction Length')
    plt.set_aspect('equal', 'box')
    plt.savefig('./length_analysis.png')
