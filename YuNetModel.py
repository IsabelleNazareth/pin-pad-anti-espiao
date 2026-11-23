import cv2

class YuNetModel:
    def __init__(self, model_path="face_detection_yunet_2023mar.onnx", threshold=0.6):
        # Inicializa o modelo 
        self.model = cv2.FaceDetectorYN.create(
            model_path,   # Caminho do modelo
            "",           # Config (vazio)
            [320, 320],   # Tamanho de entrada inicial
            threshold,    # Score threshold
            0.3,          # NMS threshold
            5000          # Top K
        )

    def detect(self, frame):
        # Atualiza o tamanho de entrada do modelo para bater com a resolução da câmera
        height, width, _ = frame.shape
        self.model.setInputSize([width, height])

        # Executa a detecção no frame
        _, faces = self.model.detect(frame)
        return faces if faces is not None else []