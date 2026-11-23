import cv2
from YuNetModel import YuNetModel

def main():
    # Inicializa a câmera 0 (padrão do Windows) e carrega o modelo
    camera = cv2.VideoCapture(0)
    modelo = YuNetModel(threshold=0.6)

    if not camera.isOpened():
        print("Erro ao acessar a webcam.")
        return

    while True:
        sucesso, img = camera.read()
        if not sucesso:
            break

        # Faz a detecção de rostos
        rostos = modelo.detect(img)

        # Desenha as marcações para cada rosto detectado
        for rosto in rostos:
            # Extrai a caixa delimitadora (x, y, largura, altura)
            caixa = list(map(int, rosto[:4]))
            cv2.rectangle(img, (caixa[0], caixa[1]), (caixa[0] + caixa[2], caixa[1] + caixa[3]), (0, 255, 0), 2)

            # Extrai os 3 landmarks (os pontos de referência) que estamos interessados que são os olhos e o nariz para saber se a pessoa está olhando para frente
            landmarks = list(map(int, rosto[4:14]))
            for i in range(3):
                cv2.circle(img, (landmarks[i*2], landmarks[i*2+1]), 2, (0, 0, 255), -1)

        # Exibe a contagem de pessoas detectadas
        cv2.putText(img, f"Rostos detectados: {len(rostos)}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        cv2.imshow("PIN Pad Anti-Espiao", img)

        # Pressione 'esc' para encerrar
        tecla = cv2.waitKey(1)
        if tecla == 27:
            break
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()