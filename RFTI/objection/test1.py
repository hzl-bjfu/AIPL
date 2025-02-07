

import glob
# import random
import os
import torch

# root = 'F:/'
# f1 = open(root + 'images.txt', 'w')
#
# # # a = sorted(glob.glob(os.path.join(root) + "/*.*"))
# # # # a = glob.glob(os.path.join(root) + "/*.*")
# # #
# # # print(len(a))
# idx = 0
# for r, d, files in os.walk(root):
#     if files != []:
#         for i in files:
#             if i.split('.')[-1] == 'JPG':
#                 fp = os.path.join(r, i)
#                 print(fp)
#                 f1.write('{}\n'.format(fp))
#                 idx += 1
# print(idx)
logits = torch.tensor([[0.2,0.1]])
a,b = logits.topk(1, 1, True, True)
print(int(b))