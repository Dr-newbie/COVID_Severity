import torch.nn as nn

class ModelWrapper(nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, img_inputs):
        # img_inputs에서 필요한 데이터를 분리
        img_data = img_inputs[:, :-self.model.cyto_input_size]
        cyto_data = img_inputs[:, -self.model.cyto_input_size:]

        # img_data를 3D 입력 형태로 변환
        img_shape = (self.model.in_channels, self.model.img_size, self.model.img_size, self.model.img_size)
        img_data = img_data.view(-1, *img_shape)

        # 모델 호출
        return self.model(img_data, cyto_data)
