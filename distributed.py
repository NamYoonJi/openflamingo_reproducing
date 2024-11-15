import torch

def init_single_device():
    # 항상 단일 GPU를 사용하도록 설정하는 함수
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    return device

def is_global_master(args):
    return True  # 단일 GPU이므로 항상 True

def is_local_master(args):
    return True  # 단일 GPU이므로 항상 True

def is_master(args, local=False):
    return True  # 단일 GPU이므로 항상 True

# 필요한 경우에만 호출하도록 수정된 main 함수 내의 코드
def main():
    # 장치 초기화
    device = init_single_device()
    # 나머지 학습 코드에서 device 사용
    print(f"Using device: {device}")

if __name__ == "__main__":
    main()
