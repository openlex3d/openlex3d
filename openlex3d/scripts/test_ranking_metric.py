from typing import List

import numpy as np

set_rank_l, set_rank_r = 3, 5
pred_ranks_a = [0,3,4]
pred_ranks_b = [3,4,11]
pred_ranks_c = [1,3,4]
pred_ranks_d = [2,3,4]
pred_ranks_e = [3,4,5]
pred_ranks_f = [3,4,6]


L = 11

def compute_set_ranking_score(ranks: List[int], set_rank_l=3, set_rank_r=5, max_label_idx=11):
    scores = []
    for rank in ranks:
        left_box_constr = 1 + min((0, (rank - set_rank_l)/set_rank_l))
        right_box_constr = 1-max((0, (rank - set_rank_r)/(max_label_idx - set_rank_r)))
        scores.append(min(left_box_constr, right_box_constr))
    return scores

rank_a_scores = compute_set_ranking_score(pred_ranks_a)
rank_b_scores = compute_set_ranking_score(pred_ranks_b)
rank_c_scores = compute_set_ranking_score(pred_ranks_c)
rank_d_scores = compute_set_ranking_score(pred_ranks_d)
rank_e_scores = compute_set_ranking_score(pred_ranks_e)
rank_f_scores = compute_set_ranking_score(pred_ranks_f)

print(rank_a_scores, 
      rank_b_scores,
      rank_c_scores, 
      rank_d_scores,
      rank_e_scores,
      rank_f_scores,
      )
print(np.mean(rank_a_scores), 
      np.mean(rank_b_scores), 
      np.mean(rank_c_scores), 
      np.mean(rank_d_scores), 
      np.mean(rank_e_scores),
      np.mean(rank_f_scores),
      )
