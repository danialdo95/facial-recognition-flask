# This is a _very simple_ example of a web service that recognizes faces in uploaded images.
#
# $ curl -XPOST -F "file=@obama2.jpg" http://127.0.0.1:5001
#
# This example is based on the Flask file upload example: http://flask.pocoo.org/docs/0.12/patterns/fileuploads/
#
# NOTE: This example requires flask to be installed! You can install it with pip:
# $ pip3 install flask

import face_recognition, os, base64
from flask import Flask, jsonify, request, redirect, render_template

# You can change this to any folder on your system
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_image():
    print(request.files)
    # Check if a valid image file was uploaded
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            # The image file seems valid! Detect faces and return the result.
            return detect_faces_in_image(file)

    # If no valid image file was uploaded, show the file upload form:
    return render_template('index.html')


def detect_faces_in_image(file_stream):

    #file_stream
    # print(base64.b64encode(file_stream.read()))
    final_data = base64.b64encode(file_stream.read()).decode("utf-8")
    # img_base64 = img_base.decode('ascii')
    # data["data"] = img_base64
    # flist.append(data)
    # final_data = {'files':flist}

    # Load the uploaded image file
    img = face_recognition.load_image_file(file_stream)
    # Get face encodings for any faces in the uploaded image
    unknown_face_encodings = face_recognition.face_encodings(img)

    face_found = False
    is_match = False
    fileImg = "Not Recognize"

    if len(unknown_face_encodings) > 0:
        face_found = True

        
        for filename in os.listdir('./static/known'):
            if filename.endswith(".jpg") or filename.endswith(".png"): 
                # print(os.path.join(directory, filename))
                image_of_barrack = face_recognition.load_image_file('./static/known/' + filename)
                print(filename)
                # print("File known Image", image_of_barrack)
                known_face_encoding = face_recognition.face_encodings(image_of_barrack)[0]

                match_results = face_recognition.compare_faces([known_face_encoding], unknown_face_encodings[0], tolerance=0.5)
                if match_results[0]:
                    is_match = True
                    fileImg = os.path.splitext(filename)[0]
                    print("result", match_results)
                    break
                continue
        # See if the first face in the uploaded image matches the known face of Obama
      

    # Return the result as json
    result = {
        "face_found_in_image": face_found,
        "known_image": is_match,
        "is_picture_of": fileImg
    }
    #return jsonify(result)
    return render_template('result.html', is_match=is_match, fileImg=fileImg, sourceImg=final_data, face_found=face_found)


if __name__ == "__main__":
    app.run()
