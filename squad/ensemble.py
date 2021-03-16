import csv
import glob
import numpy as np

save_path = './save/test/'

def parse_scores(fname, maximize_F1=True):
    # f'ensumble_F1=({results['F1']:05.2f})_EM=({results['EM']:05.2f}).csv'
    str_list = fname.split('(')
    F1 = float(str_list[-2][:5])
    EM = float(str_list[-1][:5])
    if maximize_F1:
        return F1
    else:
        return EM

def max_element_idxs(counts):
    m = max(counts)
    return np.where(np.array(counts) == m)[0]


class VotingEnsemble:

    def __init__(self, exp_names, split='dev', save_name='submission', maximize_F1=True):
        print(f'Generating ensemble for {split}.')
        self.vote_dict = {}
        self.save_path = save_path + split + '_' + save_name + '.csv'

        for exp_name in exp_names:
            if split == 'dev':
                csv_paths = glob.glob(save_path + exp_name + '/val*.csv')
            else:
                csv_paths = glob.glob(save_path + exp_name + '/test*.csv')
            for csv_path in csv_paths:
                if csv_path[-5] != ')':
                    continue
                score = parse_scores(csv_path, maximize_F1)
                with open(csv_path, mode='r') as csv_file:
                    csv_reader = csv.DictReader(csv_file)
                    for row in csv_reader:
                        uuid, pred = row['Id'], row['Predicted']
                        if uuid not in self.vote_dict.keys():
                            self.vote_dict[uuid] = {score: pred}
                        else:
                            self.vote_dict[uuid].update({score: pred})         

    def ensemble(self):
        # self.vote_dict = {id: {score: pred, score: ...}, id: ...}
        for uuid in sorted(self.vote_dict):
            scores = []
            preds = []
            for s, p in self.vote_dict[uuid].items():
                scores.append(s)
                preds.append(p)
            
            preds_count = [preds.count(p) for p in preds]

            max_count_idxs = max_element_idxs(preds_count)
            if len(max_count_idxs) > 1:
                scores_in_count_tie = np.array(scores)[max_count_idxs]
                idx_max_score_in_count_tie = np.argmax(scores_in_count_tie)
                final_pred_id = max_count_idxs[idx_max_score_in_count_tie]
            else:
                final_pred_id = max_count_idxs[0]
            self.vote_dict[uuid] = preds[final_pred_id]
    
        with open(self.save_path, 'w', newline='', encoding='utf-8') as csv_fh:
            csv_writer = csv.writer(csv_fh, delimiter=',')
            csv_writer.writerow(['Id', 'Predicted'])
            for uuid in sorted(self.vote_dict):
                csv_writer.writerow([uuid, self.vote_dict[uuid]])
        

if __name__ == '__main__':
    exp_names = ['qanet_D=128_encblk=7_head=8_bs=24_run-04-dev-ensemble-course', # qanet-large [best]
                 'qanet_D=96_encblk=5_head=6_bs=64_run-01-dev-emsemble-course', # qanet-mid
                 'qanet_D=128_encblk=7_head=8_bs=24_run-01-dev-ensemble-myazaure', # qanet-large
                 'bidaf_D=100_charEmb=True_fusion=True_bs=64_run-01-dev-ensemble-course', # bidaf+char_emb+fusion
                 'qanet_D=128_encblk=7_head=8_bs=24_run-02-dev-ensemble-myazure', # qanet-large
                 'qanet_D=96_encblk=5_head=6_bs=32_run-01-dev-F1-67.98-EM-64.27-course', # qanet-mid best
                 'qanet_D=128_encblk=7_head=8_bs=24_run-01-dev-F1-70.38-EM-66.81-course', # qanet-large 2nd best
                ]
    voting_ensemble = VotingEnsemble(exp_names, split='dev')
    voting_ensemble.ensemble()