import matplotlib.pyplot as plt

def analysis(gold_dict, pred_dict):
    for key, value in pred_dict.items():
        if len(gold_dict[key]['answers']) > 0:
            ground_truth = gold_dict[key]['answers'][0]
        else:
            ground_truth = ''
        prediction = value
        plt.scatter(len(ground_truth), len(prediction), color='red')
    plt.savefig('./length_analysis.png')