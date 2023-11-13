import re
import time
from torch.utils.tensorboard import SummaryWriter

# TensorBoard writer 초기화
writer = SummaryWriter('/data/keti/syh/checkpoints/CLIP_ReID_taskawarePromptBase_on_MSMT17/runs/log_analysis')

# 마지막으로 읽은 라인의 번호를 추적
last_line_read = 0

try:
    while True:
        with open('/data/keti/syh/checkpoints/CLIP_ReID_taskawarePromptBase_on_MSMT17/train_log.txt', 'r') as file:
            # 파일의 시작으로 이동
            file.seek(0)

            # 이미 읽은 라인은 건너뛰기
            for _ in range(last_line_read):
                next(file)

            # 새로운 라인 읽기
            for line in file:
                match = re.search(r'Epoch\[(\d+)\] Iteration\[(\d+)/\d+\] Loss: ([\d.]+), Base Lr: ([\d.e-]+)', line)
                if match:
                    epoch = int(match.group(1))
                    iteration = int(match.group(2))
                    loss = float(match.group(3))
                    lr = float(match.group(4))

                    # TensorBoard에 기록
                    writer.add_scalar('Loss/train', loss, epoch * 510 + iteration)
                    writer.add_scalar('Learning Rate', lr, epoch * 510 + iteration)

                # 현재 라인 번호 업데이트
                last_line_read += 1

        # 10초 간격으로 반복
        time.sleep(10)
except KeyboardInterrupt:
    # 사용자가 Ctrl+C를 누르면 실행됨
    print("KeyboardInterrupt received, closing writer.")
    writer.close()
