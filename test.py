from trainer import Trainer
from utils import load_image
import torch
import numpy as np
import time
gpu = True

path = "/home/common_gpu0/corpora/vision/liveness/rose/images/2/Mu_T_HS_g_E_2_93__0_down_gam1/Mu_T_HS_g_E_2_93__0_0.jpg,/home/common_gpu0/corpora/vision/liveness/rose/images/2/Mu_T_HS_g_E_2_93__0_down_gam1/Mu_T_HS_g_E_2_93__0_1.jpg,/home/common_gpu0/corpora/vision/liveness/rose/images/2/Mu_T_HS_g_E_2_93__0_down_gam1/Mu_T_HS_g_E_2_93__0_2.jpg,/home/common_gpu0/corpora/vision/liveness/rose/images/2/Mu_T_HS_g_E_2_93__0_down_gam1/Mu_T_HS_g_E_2_93__0_3.jpg,/home/common_gpu0/corpora/vision/liveness/rose/images/2/Mu_T_HS_g_E_2_93__0_down_gam1/Mu_T_HS_g_E_2_93__0_4.jpg,/home/common_gpu0/corpora/vision/liveness/rose/images/2/Mu_T_HS_g_E_2_93__0_down_gam1/Mu_T_HS_g_E_2_93__0_5.jpg,/home/common_gpu0/corpora/vision/liveness/rose/images/2/Mu_T_HS_g_E_2_93__0_down_gam1/Mu_T_HS_g_E_2_93__0_6.jpg,/home/common_gpu0/corpora/vision/liveness/rose/images/2/Mu_T_HS_g_E_2_93__0_down_gam1/Mu_T_HS_g_E_2_93__0_7.jpg,/home/common_gpu0/corpora/vision/liveness/rose/images/2/Mu_T_HS_g_E_2_93__0_down_gam1/Mu_T_HS_g_E_2_93__0_8.jpg"

X = [load_image(tmp) for tmp in path.split(',')[:8]]

X = np.array([X]).transpose(0,4,1,2,3)

X = torch.Tensor(X)
print(X.shape)

model_path = '/home/datvt/liveness/result/init/model_epoch40'
trainer = Trainer('data', model_path, gpu)
trainer.set_model()
start_time = time.time()

print(trainer.predict(X))

print("--- %s seconds ---" % (time.time() - start_time))

