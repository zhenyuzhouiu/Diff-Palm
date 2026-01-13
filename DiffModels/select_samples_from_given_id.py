'''
The DiffPalm synthesizes and save the palmprint images into './Project/ZZY/Diff-Palm/DiffModels/output_zhenyu/test-large/results/.
And the name of samples is start from 0.png to ***.png, each 20 images are from the same identity, such as 0-19.png.
'''

import os
import shutil
from tqdm import tqdm

no_sample = 20
start_id = 20000  # the first id is 000000
end_id = 40000 # the last id is 99999
src_folder = "./output_zhenyu/test-large/results"
dst_folder = "./select/"
for id in tqdm(range(start_id, end_id)):
    sample_list = range(id * no_sample, (id+1) * no_sample)
    dst_id = os.path.join(dst_folder, str(id).zfill(6))
    os.makedirs(dst_id, exist_ok=True)
    for sample in sample_list:
        shutil.copy(os.path.join(src_folder, str(sample) + '.png'),
                    os.path.join(dst_id, str(sample)+'.jpg'))
        