import cv2
from YuNetModel import YuNetModel

def main():
    # Inicializa a câmera 0 (padrão do Windows) e carrega o modelo
    cap = cv2.VideoCapture(0)
    model = YuNetModel(threshold=0.6)

    if not cap.isOpened():
        print("Erro ao acessar a webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Faz a detecção de rostos
        faces = model.detect(frame)

        # Desenha as marcações para cada rosto detectado
        for face in faces:
            # Extrai a caixa delimitadora (x, y, largura, altura)
            box = list(map(int, face[:4]))
            cv2.rectangle(frame, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)

            # Extrai os 5 landmarks (olhos, nariz e cantos da boca)
            landmarks = list(map(int, face[4:14]))
            for i in range(5):
                cv2.circle(frame, (landmarks[i*2], landmarks[i*2+1]), 2, (0, 0, 255), -1)

        # Exibe a contagem de pessoas detectadas (Meta 1)
        cv2.putText(frame, f"Rostos detectados: {len(faces)}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        cv2.imshow("PIN Pad Anti-Espiao", frame)

        # Pressione 'q' para encerrar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()