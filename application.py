# coding=utf8
import base64
import io
import json
import os
import pickle
from threading import Thread
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from face.face_model_util import train_model
import face_recognition
from flask import Flask, render_template, request, make_response

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

face_img_fix = "jpg"
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
APP_DATA = os.path.join(APP_ROOT, 'face', 'data')
# executor = ThreadPoolExecutor(2)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/do_search/<string:pname>', methods=['GET'])
def do_search(pname):
    result = []
    if pname is None or pname == "" or len(pname) < 1:
        return "请输入姓名"
    if not os.path.exists(os.path.join(APP_DATA, 'train/%s' % pname)):
        pass
    else:
        result = os.listdir(os.path.join(APP_DATA, 'train/%s' % pname))
    return json.dumps(result)


@app.route('/do_save', methods=['POST'])
def do_save():
    if len(request.files) < 1:
        return "请上传照片"
    pname = request.form['pname']
    if pname is None or pname == "" or len(pname) < 1:
        return "请输入姓名"

    file_data = request.files["file_pic"]
    if not allowed_file(file_data.filename) or not file_data:
        return "图片格式有误，请重新上传"

    pic_path = os.path.join(APP_DATA, 'train', pname)
    if not os.path.exists(pic_path):
        os.mkdir(pic_path)
    img_path = os.path.join(APP_DATA, 'train', pname, str(uuid.uuid4()) + ".jpg")
    file_data.save(img_path)
    result = "上传成功"
    return result


@app.route('/do_train', methods=['POST'])
def do_train():
    result = "训练失败"
    try:
        total=train_model(os.path.join(APP_DATA, "model", "trained_knn_model.clf"), os.path.join(APP_DATA, "train"))
        result = "训练成功耗时："+total+"秒"
    except:
        pass
    return result


@app.route('/do_upload_origin', methods=['POST'])
def do_upload_origin():
    if len(request.files) < 1:
        return "请上传文件"
    file_data = request.files["file_pic"]
    if file_data and allowed_file(file_data.filename):
        img_path = os.path.join(APP_DATA, "file_pic." + face_img_fix)
        file_data.save(img_path)
    result = "成功"
    return result


@app.route('/show_photo_pname/<string:pname>/<string:filename>', methods=['GET'])
def show_photo_pname(pname, filename):
    if request.method == 'GET':
        if filename is None:
            pass
        else:
            image_data = open(os.path.join(APP_DATA, "train", pname, filename, ), "rb").read()
            response = make_response(image_data)
            response.headers['Content-Type'] = 'image/png'
            return response
    else:
        pass


@app.route('/show_photo/<string:filename>', methods=['GET'])
def show_photo(filename):
    if request.method == 'GET':
        if filename is None:
            pass
        else:
            image_data = open(os.path.join(APP_DATA, filename, ), "rb").read()
            response = make_response(image_data)
            response.headers['Content-Type'] = 'image/png'
            return response
    else:
        pass


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


#rest begin
@app.route('/detect_faces_in_image', methods=['POST'])
def detect_faces_in_image():
    print(time.time())
    name = ""
    try:
        distance_threshold = 0.4
        face_str = request.form['face_str']
        # face_str = "/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAAyACwDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwDChiSe0QCGPhfvBOSa9Jg0LTpIQf7LssKgLMYUAHHUkiuP8OQB5beQwyyxxRmUpEuWkYDIRfUkkVr3OoDV7OOa/P8AZ9oXdFsWLPIWRipL/cw2QeOcDFcrpxcVoPDQgqa91fcbQ0rQlPzWul591jqcaJpGAf7MsSD38hP8KwbC28OXKrJYWgW4U7WaW23gMM8bjJkGri6u6zPZDTrhmW4jtvtCToIleQqFzn5urr0B61n7ON9jsjCnbWK+42l8O6WV3DR7Mj1+yr/hTDoWkA4Ok2QP/Xsn+FYUul6Mly5utRe2uMbt6I+M+m7eOavwQ6lFHtstehkg7NKzZ/UNx+NUqUTOSp7cq+4xvDun6fHJaymxt0naxDsRk7s7PnwSQG68gDrW8qtBLMLaJZI5X3lXkKFW2gHDKrcHaOMdc1U0i0dNKsbrKsktpEoZeRwq9/YggjsRVuEsrYJ4pJ+6i8PGPLH0Qsj3mwkwQDHeS6kcD8DGv86ktbMS6ZNavIWabc7SqNpEpO4OB2wwGP8AdFE2548RsFYMGBIyOCD0/Cq0dnq6hxDrKhnyQ81upZfwUBTj3H50jqsTeXNcsrXEFm0oIJkDlfmHcKY3wfxqZNNi25Ntaknkk3bj9BDU0azKA1w8bzEDe0a4DHpnGBTzLg9KadmZzgmcP4V0K6024tr9b3fZXFqxe33uNrOoYEL93IOOevJrqMjrVTQWjn0vTYElXebdMggjAVMsenOAD0qMX0asFZsZ5Ge9U0+VM5MJJKC9Cxd/ayo+zTxR/wB7fGWJ+mCMVUhN0Jdo1aIOTkxfZZB+okq6k0EgHIP41IkMBbPT3zUKVj0IyViW0lveY71YCR92SFyQw9wQMH86t7RVZ3ihHLAAetNN+8eAum3lwCMh49oH6kGqhHmZhVnyo8gi1TUICnk31zHsG1dkzDaMYwMH04qM3t0VVTczFVGAPMOBRRXL0R8zS2FF/eL927nH0kNP/tTUP+f+6/7/ADf40UVDNRDqd+3W+uT9ZW/xqRNZ1SPds1K8XcxZts7DJPUnnrRRVRJmf//Z"
        img_b64decode = base64.b64decode(face_str)
        img = face_recognition.load_image_file(io.BytesIO(img_b64decode))
        # Get face encodings for any faces in the uploaded image
        face_encodings = face_recognition.face_encodings(img)
        # load knn
        with open(os.path.join(APP_DATA, 'model', 'trained_knn_model.clf'), 'rb') as f:
            knn_clf = pickle.load(f, encoding='bytes')
            closest_distances = knn_clf.kneighbors(face_encodings, n_neighbors=1)
            if closest_distances[0][0][0] <= distance_threshold:
                re = knn_clf.predict(face_encodings)
                if len(re) > 0:
                    name = re[0]
        print(time.time())
        print(name)
        print(closest_distances)
    except:
        pass
    return name


@app.route('/do_add_face', methods=['POST'])
def do_add_face():
    fstr = request.form['fstr']
    if fstr is None or fstr == "" or len(fstr) < 1:
        return "需要传入图片"
    pname = request.form['pname']
    if pname is None or pname == "" or len(pname) < 1:
        return "需要传入姓名"
    try:
        img_b64decode = base64.b64decode(fstr)
        pic_path = os.path.join(APP_DATA, 'train', pname)
        if not os.path.exists(pic_path):
            os.mkdir(pic_path)
        img_path = os.path.join(APP_DATA, 'train', pname, str(uuid.uuid4()) + ".jpg")
        open(img_path,'wb').write(img_b64decode)
        result = "上传成功"
        thr = Thread(target = train_model, args =(os.path.join(APP_DATA, "model", "trained_knn_model.clf"), os.path.join(APP_DATA, "train")))
        thr.start()
    except:
        result="上传失败"
    return result
#rest end

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8088', debug=False, threaded=True)
    # detect_faces_in_image()
