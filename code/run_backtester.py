from backtester.backTester import *
from backtester.singleModel import *


base_directory = '/home/tonnonssi/YOLO-Futures'        # 파일 위치 
file_name = 'GOT_KL_hybrid/55_scaling_MultiState_17'   # 백테스트하고 싶은 파일 지정 
target_directory = base_directory + '/logs/' + file_name

# None이면 main_backTester에서 돌아간 모델 중 가장 pnl이 높은 애로
# 다른 애를 쓰고 싶다면, 특정 신경망 위치를 지정 
# --------------------------------------------------------
model_path = target_directory + '/models/I12_10steps.pth'

# model_path = None 

# main_backTester(target_directory)                      # 전체 모델에 대한 검증 시작

# 하나의 가중치에 대해 여러번 valid를 돌림 : 정책이 greedy하지 않기 때문  
main_single_backTester(target_directory, 
                       model_path=model_path,          # 위에서 지정 
                       n_runs=30)                      # 신뢰도를 위해 적어도 30회 

lst = ['I10latest.pth', 'I3latest.pth', 'I11latest.pth', 'I7latest.pth', 'I10_4steps.pth']

for fname in lst:
    model_path = target_directory + '/models/' + fname
    main_single_backTester(target_directory, 
                            model_path=model_path,          # 위에서 지정 
                            n_runs=30)     