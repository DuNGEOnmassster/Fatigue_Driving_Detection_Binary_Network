import os

if __name__ == '__main__':

    fc_pred = list()
    with open("FC_result.txt", 'r') as f:
        lines = f.readlines()
        gt_0 = [int(i) for i in lines[0].split(",")[1:]]
        pred_test = [int(i) for i in lines[-1].split(",")[1:]]

    cnt = 0

    for i in range(len(gt_0)):
        if gt_0[i] == pred_test[i]:
            cnt += 1

    print(f"Test Acc = {cnt} / {len(gt_0)}, Acc rate = {cnt/len(gt_0)}")