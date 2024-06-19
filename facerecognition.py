import cv2
from deepface import DeepFace

pessoas_cadastradas = [{'nome' : 'Brad Pitt', 'img_ref': 'known_faces/pitt.jpg'},
                       {'nome' : 'Jason Momoa', 'img_ref': 'known_faces/momoa.jpg'}]

input_img_path = "input_imgs/input.jpg"

# Realizando a análise da imagem de entrada
recognized_person = None

for pessoa in pessoas_cadastradas:
    result = DeepFace.verify(img1_path = pessoa['img_ref'], img2_path = input_img_path)
    if (result['verified'] == True):
        #print(result)
        recognized_person = pessoa

if (recognized_person != None):
    print(f"Bem vindo {recognized_person['nome']} !")
    input_img = cv2.imread(input_img_path)
    user_emotion_prob = DeepFace.analyze(input_img, actions=("emotion"))
    #print(user_emotion_prob)
    print(f"Sua emoção atual é: {user_emotion_prob[0]['dominant_emotion']}")

else:
    print("Pessoa não reconhecida...")


