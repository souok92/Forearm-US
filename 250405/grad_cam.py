import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from torchvision import transforms, models

# GradCAM 클래스 정의
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.hooks = []
        self.gradients = None
        self.activations = None
        
        # 레이어의 출력과 그래디언트를 저장하기 위한 훅 등록
        def save_gradient(grad):
            self.gradients = grad.detach()
            
        def forward_hook(module, input, output):
            self.activations = output.detach()
            
        # 타겟 레이어에 훅 등록
        handle = self.target_layer.register_forward_hook(forward_hook)
        self.hooks.append(handle)
        handle = self.target_layer.register_full_backward_hook(
            lambda module, grad_input, grad_output: save_gradient(grad_output[0])
        )
        self.hooks.append(handle)
    
    def __call__(self, x, class_idx=None):
        # 모델 출력 가져오기
        self.model.eval()
        self.model.zero_grad()
        
        # 정방향 전파
        output = self.model(x)
        
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1)
        
        # 타겟 클래스에 대한 점수
        target = output[0][class_idx]
        
        # 역방향 전파
        target.backward()
        
        # 그래디언트와 활성화 맵 가져오기
        gradients = self.gradients[0]
        activations = self.activations[0]
        
        # 각 채널별 그래디언트 가중치 계산 (Global Average Pooling)
        weights = torch.mean(gradients, dim=(1, 2), keepdim=True)
        
        # 가중치와 활성화 맵 조합
        cam = torch.sum(weights * activations, dim=0).cpu().detach().numpy()
        
        # ReLU 적용
        cam = np.maximum(cam, 0)
        
        # 정규화
        if np.max(cam) > 0:
            cam = cam / np.max(cam)
        
        # 클린업
        for hook in self.hooks:
            hook.remove()
        
        return cam
    
    def __del__(self):
        for hook in self.hooks:
            hook.remove()

# 히트맵 생성 및 원본 이미지에 오버레이하는 함수
def generate_cam_image(model, img_tensor, original_img, target_layer, class_idx=None, alpha=0.5):
    # GradCAM 초기화
    grad_cam = GradCAM(model, target_layer)
    
    # CAM 생성
    cam = grad_cam(img_tensor.unsqueeze(0), class_idx)
    
    # 원본 이미지 크기로 CAM 리사이징
    orig_size = original_img.size
    cam = cv2.resize(cam, (orig_size[0], orig_size[1]))
    
    # 히트맵 생성
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # 원본 이미지를 RGB로 변환
    if original_img.mode != 'RGB':
        original_img = original_img.convert('RGB')
    original_img_np = np.array(original_img)
    
    # 히트맵과 원본 이미지 결합
    superimposed_img = heatmap * alpha + original_img_np * (1 - alpha)
    superimposed_img = np.uint8(superimposed_img)
    
    return superimposed_img, cam

def apply_gradcam_to_image(model, img_path, transform, target_layer, device, class_names):
    # 이미지 로드 및 전처리
    original_img = Image.open(img_path).convert('L')
    img_tensor = transform(original_img).to(device)
    
    # 예측 클래스 가져오기
    model.eval()
    with torch.no_grad():
        output = model(img_tensor.unsqueeze(0))
        predicted_class = torch.argmax(output, dim=1).item()
    
    # GradCAM 적용
    superimposed_img, _ = generate_cam_image(model, img_tensor, original_img, target_layer)
    
    # 시각화 - 오직 오버레이 이미지만 표시
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 오버레이 이미지
    ax.imshow(superimposed_img)
    ax.set_title(f'Class: {class_names[predicted_class]}', fontsize=14)
    ax.axis('off')
    
    plt.tight_layout()
    return fig, predicted_class


def main():
    # 랜덤 시드 설정으로 재현성 보장
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 클래스 정의
    CLASSES = ['open', 'index', 'mid', 'ring', 'pinky', 'fist']
    
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    
    # 모델 타입 선택
    model_type = 'resnet18'  # 'resnet18' 또는 'mobilenet'
    
    # 모델 정의 (train.py에서 정의한 모델 클래스 사용)
    class HandGestureModel(nn.Module):
        def __init__(self, model_type='resnet18', num_classes=len(CLASSES)):
            super(HandGestureModel, self).__init__()
            
            if model_type == 'resnet18':
                # ResNet18 기반 모델, 첫 레이어를 그레이스케일 입력에 맞게 수정
                self.model = models.resnet18(weights=None)
                self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
                self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
            elif model_type == 'mobilenet':
                # MobileNetV2 기반 모델
                self.model = models.mobilenet_v2(weights=None)
                # 그레이스케일 입력에 맞게 첫 번째 레이어 수정
                self.model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
                # 마지막 분류기 레이어 수정
                self.model.classifier[1] = nn.Linear(self.model.last_channel, num_classes)
            
        def forward(self, x):
            return self.model(x)
    
    # 모델 생성 및 가중치 로드
    model = HandGestureModel(model_type=model_type).to(device)
    model_path = f'best_hand_gesture_{model_type}_model.pth'
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # 타겟 레이어 설정 (모델에 따라 다름)
    if model_type == 'resnet18':
        target_layer = model.model.layer4[-1]
    else:  # mobilenet
        target_layer = model.model.features[-1]
    
    # 변환 정의
    transform = transforms.Compose([
        transforms.Resize((302, 317)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # 테스트할 이미지 경로
    data_root = "./cat_gest"
    results_dir = "./gradcam_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # 각 클래스에서 몇 개의 이미지를 랜덤으로 샘플링하여 Grad-CAM 적용
    samples_per_class = 2
    
    for class_name in CLASSES:
        class_dir = os.path.join(data_root, class_name)
        if not os.path.isdir(class_dir):
            print(f"Directory not found: {class_dir}")
            continue
        
        img_files = [f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # 랜덤으로 이미지 선택 (NumPy 사용)
        if len(img_files) > samples_per_class:
            indices = np.random.choice(len(img_files), samples_per_class, replace=False)
            selected_files = [img_files[i] for i in indices]
        else:
            selected_files = img_files
        
        for img_file in selected_files:
            img_path = os.path.join(class_dir, img_file)
            print(f"Applying Grad-CAM to {img_path}")
            
            fig, pred_class = apply_gradcam_to_image(model, img_path, transform, target_layer, device, CLASSES)
            
            # 결과 저장 - 클래스 이름을 파일명 맨 앞에 배치
            filename = f"{CLASSES[pred_class]}_{os.path.splitext(img_file)[0]}_gradcam.png"
            output_path = os.path.join(results_dir, filename)
            fig.savefig(output_path, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()