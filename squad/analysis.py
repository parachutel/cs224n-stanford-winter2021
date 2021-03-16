import matplotlib.pyplot as plt

def analysis(gold_dict, pred_dict):
    for key, value in pred_dict.items():
        ground_truth = gold_dict[key]['answers'][0]
        prediction = value
        plt.scatter(len(ground_truth), len(prediction), color='red')
plt.save('length_analysis.png')