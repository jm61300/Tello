import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# For static images:
IMAGE_FILES = ['selfie.jpeg']
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
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
      print('face_landmarks:', face_landmarks)
      mp_drawing.draw_landmarks(
          image=annotated_image,
          landmark_list=face_landmarks,
          connections=mp_face_mesh.FACEMESH_TESSELATION,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp_drawing_styles
          .get_default_face_mesh_tesselation_style())
      mp_drawing.draw_landmarks(
          image=annotated_image,
          landmark_list=face_landmarks,
          connections=mp_face_mesh.FACEMESH_CONTOURS,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp_drawing_styles
          .get_default_face_mesh_contours_style())
      mp_drawing.draw_landmarks(
          image=annotated_image,
          landmark_list=face_landmarks,
          connections=mp_face_mesh.FACEMESH_IRISES,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp_drawing_styles
          .get_default_face_mesh_iris_connections_style())
    cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)
    #Tentando fazer apenas os pontos desejados
    for face_landmarks in results.multi_face_landmarks:
      ##########################################
      for landmarks in face_landmarks.landmark:
         x = landmarks.x
         y = landmarks.y
         shape = annotated_image1.shape
         relative_x = int(x*shape[1])
         relative_y = int(y*shape[0])
         cv2.circle(annotated_image1, (relative_x, relative_y), radius=1, color=(255,0,100), thickness=1)
      ##########################################
    #   print('face_landmarks:', face_landmarks)
    #   mp_drawing.draw_landmarks(
    #       image=annotated_image1,
    #       landmark_list=face_landmarks,
    #       connections=mp_face_mesh.FACEMESH_TESSELATION, #essa parte faz o tecido 
    #       landmark_drawing_spec=None,
    #       connection_drawing_spec=mp_drawing_styles
    #       .get_default_face_mesh_tesselation_style())
    #   mp_drawing.draw_landmarks(
    #       image=annotated_image1,
    #       landmark_list=face_landmarks,
    #       connections=mp_face_mesh.FACEMESH_CONTOURS, #essa parte faz o contorno da face apenas
    #       landmark_drawing_spec=None,
    #       connection_drawing_spec=mp_drawing_styles
    #       .get_default_face_mesh_contours_style())
    #   mp_drawing.draw_landmarks(
    #       image=annotated_image1,
    #       landmark_list=face_landmarks,
    #       connections=mp_face_mesh.FACEMESH_IRISES, #essa parte faz o contorno da iris
    #       landmark_drawing_spec=None,
    #       connection_drawing_spec=mp_drawing_styles
    #       .get_default_face_mesh_iris_connections_style())
    cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image1)
cv2.imshow('Press q to kill', annotated_image)
cv2.imshow('Press q to kill1', annotated_image1)
print(len(face_landmarks.landmark))
# cv2.imshow('2',results)
while True:     
    if cv2.waitKey(1) == ord('q'):
        break
cv2.waitKey(3000)
cv2.destroyAllWindows() 

# # For webcam input:
# drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
# cap = cv2.VideoCapture(0)
# with mp_face_mesh.FaceMesh(
#     max_num_faces=1,
#     refine_landmarks=True,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5) as face_mesh:
#   while cap.isOpened():
#     success, image = cap.read()
#     if not success:
#       print("Ignoring empty camera frame.")
#       # If loading a video, use 'break' instead of 'continue'.
#       continue

#     # To improve performance, optionally mark the image as not writeable to
#     # pass by reference.
#     image.flags.writeable = False
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     results = face_mesh.process(image)

#     # Draw the face mesh annotations on the image.
#     image.flags.writeable = True
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     if results.multi_face_landmarks:
#       for face_landmarks in results.multi_face_landmarks:
#         mp_drawing.draw_landmarks(
#             image=image,
#             landmark_list=face_landmarks,
#             connections=mp_face_mesh.FACEMESH_TESSELATION,
#             landmark_drawing_spec=None,
#             connection_drawing_spec=mp_drawing_styles
#             .get_default_face_mesh_tesselation_style())
#         mp_drawing.draw_landmarks(
#             image=image,
#             landmark_list=face_landmarks,
#             connections=mp_face_mesh.FACEMESH_CONTOURS,
#             landmark_drawing_spec=None,
#             connection_drawing_spec=mp_drawing_styles
#             .get_default_face_mesh_contours_style())
#         mp_drawing.draw_landmarks(
#             image=image,
#             landmark_list=face_landmarks,
#             connections=mp_face_mesh.FACEMESH_IRISES,
#             landmark_drawing_spec=None,
#             connection_drawing_spec=mp_drawing_styles
#             .get_default_face_mesh_iris_connections_style())
#     # Flip the image horizontally for a selfie-view display.
#     cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
#     if cv2.waitKey(5) & 0xFF == 27:
#       break
# cap.release()