import cv2
import mediapipe as mp
from math import sqrt
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

def distance(x1, y1, x0, y0):
   return sqrt((x1-x0)**2 + (y1-y0)**2)

# For static images:
IMAGE_FILES = ['selfie.jpeg']
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
eye_dots = [130, 145, 159, 243, 359, 374, 386, 463]
with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5) as face_mesh:
  for idx, file in enumerate(IMAGE_FILES):
    image = cv2.imread(file)
    # Convert the BGR image to RGB before processing.
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Print and draw face mesh landmarks on the image.
    if not results.multi_face_landmarks:
      continue
    annotated_image = image.copy()
    annotated_image1 = image.copy()
    for face_landmarks in results.multi_face_landmarks:
      i=0
      positionsx = {}
      positionsy = {}
      for landmarks in face_landmarks.landmark: #marcando cada ponto de landmark no rosto
        x = landmarks.x
        y = landmarks.y
        shape = annotated_image.shape
        relative_x = int(x*shape[1])
        relative_y = int(y*shape[0])
        cv2.putText(annotated_image, str(i),(relative_x, relative_y), cv2.FONT_HERSHEY_COMPLEX, 0.3, color=(255,0,100), thickness=1)
        if i in eye_dots:
          positionsx[str(i)] = relative_x
          positionsy[str(i)] = relative_y
          cv2.putText(annotated_image1, str(i),(relative_x, relative_y), cv2.FONT_HERSHEY_COMPLEX, 0.7, color=(255,0,100), thickness=1)
        i+=1
      distv1 = distance(positionsx["159"], positionsy["159"], positionsx["145"], positionsy["145"])
      disth1 = distance(positionsx["130"], positionsy["130"], positionsx["243"], positionsy["243"])
      distv2 = distance(positionsx["386"], positionsy["386"], positionsx["374"], positionsy["374"])
      disth2 = distance(positionsx["463"], positionsy["463"], positionsx["359"], positionsy["359"])
      print("dist vertical esquerda:", distv1)
      print("dist horizontal esquerda:", disth1)
      #relacao pra olho aberto
      rel1 = distv1/disth1
      print("dist vertical direita:", distv2)
      print("dist horizontal direita:", disth2)
      rel2 = distv2/disth2
      print(rel1, rel2)
      #relacao aurea ~ 0.15
      
# cv2.imshow('Press q to kill', annotated_image)
cv2.imshow('Press q to kill1', annotated_image1)
# print(len(face_landmarks.landmark))
# print(annotated_image.shape)
# cv2.imshow('2',results)
cv2.imshow('Press q to kill1', annotated_image1)
while True:     
    if cv2.waitKey(1) == ord('q'):
        break
cv2.waitKey(3000)
cv2.destroyAllWindows() 