import argparse
import pickle
import os
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":

    r1 = "work_dir/ctrgcn_B_joint_motion"
    r2 = "work_dir/ctrgcn_B_bone_motion"
    r3 = "work_dir/ctrgcn_Bbone"
    r4 = "work_dir/ctrgcn_Bjoint"


    sr1 = "work_dir/mix_Bjoint_motion"
    sr2 = "work_dir/mix_Bbone_motion"
    sr3 = "work_dir/mix_Bjoint"
    sr4 = "work_dir/mix_Bbone"


    with open('data/test_B_label.npy', 'rb') as f:
        label = np.load(f)

    with open(os.path.join(r1, 'epoch1_test_score.pkl'), 'rb') as r1:
        r1 = list(pickle.load(r1).items())

    with open(os.path.join(r2, 'epoch1_test_score.pkl'), 'rb') as r2:
        r2 = list(pickle.load(r2).items())


    with open(os.path.join(r3, 'epoch1_test_score.pkl'), 'rb') as r3:
        r3 = list(pickle.load(r3).items())

    with open(os.path.join(r4, 'epoch1_test_score.pkl'), 'rb') as r4:
        r4 = list(pickle.load(r4).items())



    with open(os.path.join(sr1, 'epoch1_test_score.pkl'), 'rb') as sr1:
        sr1 = list(pickle.load(sr1).items())
    with open(os.path.join(sr2, 'epoch1_test_score.pkl'), 'rb') as sr2:
        sr2 = list(pickle.load(sr2).items())
    with open(os.path.join(sr3, 'epoch1_test_score.pkl'), 'rb') as sr3:
        sr3 = list(pickle.load(sr3).items())
    with open(os.path.join(sr4, 'epoch1_test_score.pkl'), 'rb') as sr4:
        sr4 = list(pickle.load(sr4).items())

    right_num = total_num = right_num_5 = 0
    best = 0.0

    total_num = 0
    right_num = 0

    alpha = [0.2, 0.2, 1.2, 1.2]
    alpha2 = [0.2, 0.2, 1.2, 1.2]


    # 创建一个列表来存储融合结果
    fused_results = []

    for i in tqdm(range(len(label))):
        l = label[i]
        _, r11 = r1[i]
        _, r22 = r2[i]
        _, r33 = r3[i]
        _, r44 = r4[i]


        _, sr11 = sr1[i]
        _, sr22 = sr2[i]
        _, sr33 = sr3[i]
        _, sr44 = sr4[i]




        result1 = r11 * alpha[0] + r22 * alpha[1] + r33 * alpha[2] + r44 * alpha[3]

        result2 = sr11 * alpha2[0] + sr22 * alpha2[1] + sr33 * alpha2[2] + sr44 * alpha2[3]

        r = result1 + result2

        # 将融合结果添加到列表中
        fused_results.append(r)

        rank_5 = r.argsort()[-5:]
        right_num_5 += int(int(l) in rank_5)
        r_max = np.argmax(r)
        right_num += int(r_max == int(l))
        total_num += 1

    # 将融合结果列表转换为numpy数组
    fused_results_array = np.array(fused_results)

    # 保存融合结果为prey.npy文件
    np.save('pred.npy', fused_results_array)

    acc = right_num / total_num
    print(acc, alpha)
    if acc > best:
        best = acc
        best_alpha = alpha
    acc5 = right_num_5 / total_num

    print(best, best_alpha)
    print('Top1 Acc: {:.4f}%'.format(acc * 100))
    print('Top5 Acc: {:.4f}%'.format(acc5 * 100))
    print('Fusion results saved to prey.npy')