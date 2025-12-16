from flask import Flask, render_template, request, session, redirect, url_for, Response, jsonify, send_file
import mysql.connector
import cv2
from PIL import Image
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.lib.units import inch
import os
import time
from datetime import date
from datetime import datetime
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
 
app = Flask(__name__)
app.secret_key = 'sincontraseña'
 
cnt = 0
pause_cnt = 0
justscanned = False
 
mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="",
    database="flask_db"
)
mycursor = mydb.cursor()
 
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        mycursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        user = mycursor.fetchone()
        
        if user:
            if user[2] == password: 
                session['logged_in'] = True
                session['user_id'] = user[0]
                session['username'] = user[1]
                return redirect(url_for('home'))
            elif check_password_hash(user[2], password):
                session['logged_in'] = True
                session['user_id'] = user[0]
                session['username'] = user[1]
                return redirect(url_for('home'))
            else:
                error = 'Credenciales inválidas. Por favor intente nuevamente.'
        else:
            error = 'Credenciales inválidas. Por favor intente nuevamente.'
    
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Generate dataset >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def generate_dataset(nbr):
    face_classifier = cv2.CascadeClassifier("C:/Users/Milton Montiel/Downloads/titulacion 1/proyecto_ia2/resources/haarcascade_frontalface_default.xml")
 
    def face_cropped(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
 
        if faces is ():
            return None
        for (x, y, w, h) in faces:
            cropped_face = img[y:y + h, x:x + w]
        return cropped_face
 
    cap = cv2.VideoCapture(0)
 
    mycursor.execute("select ifnull(max(img_id), 0) from img_dataset")
    row = mycursor.fetchone()
    lastid = row[0]
 
    img_id = lastid
    max_imgid = img_id + 100
    count_img = 0
    dataset_complete = False
 
    while True:
        ret, img = cap.read()
        if face_cropped(img) is not None:
            count_img += 1
            img_id += 1
            face = cv2.resize(face_cropped(img), (200, 200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
 
            file_name_path = "dataset/"+nbr+"."+ str(img_id) + ".jpg"
            cv2.imwrite(file_name_path, face)

            cv2.putText(face, str(count_img), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
 
            mycursor.execute("""INSERT INTO `img_dataset` (`img_id`, `img_person`) VALUES
                                ('{}', '{}')""".format(img_id, nbr))
            mydb.commit()
 
            if int(img_id) == int(max_imgid):
                dataset_complete = True
                completion_img = np.zeros((200, 200), np.uint8)
                cv2.putText(completion_img, "Dataset", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(completion_img, "Completo!", (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(completion_img, "Presione", (40, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(completion_img, "Regresar", (40, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                frame = cv2.imencode('.jpg', completion_img)[1].tobytes()
            else:
                frame = cv2.imencode('.jpg', face)[1].tobytes()
            
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
 
            if cv2.waitKey(1) == 13 or dataset_complete:
                break
                cap.release()
                cv2.destroyAllWindows()
 
 
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Register Users >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
@app.route('/register', methods=['GET', 'POST'])
def register():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        if not username or not password:
            error = 'Todos los campos son requeridos.'
        elif password != confirm_password:
            error = 'Las contraseñas no coinciden.'
        else:
            try:
                mycursor.execute("SELECT username FROM users WHERE username = %s", (username,))
                if mycursor.fetchone():
                    error = 'El nombre de usuario ya está en uso.'
                else:
                    mycursor.execute(
                        "INSERT INTO users (username, password) VALUES (%s, %s)",
                        (username, password)
                    )
                    mydb.commit()
                    return redirect(url_for('login'))
            except Exception as e:
                error = 'Error al registrar el usuario. Por favor, intente nuevamente.'
                print(e) 
    
    return render_template('register.html', error=error)
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Train Classifier >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
@app.route('/train_classifier/<nbr>')
def train_classifier(nbr):
    dataset_dir = "C:/Users/Milton Montiel/Downloads/titulacion 1/proyecto_ia2/proyecto_ia2/dataset"
 
    path = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir)]
    faces = []
    ids = []
 
    for image in path:
        img = Image.open(image).convert('L');
        imageNp = np.array(img, 'uint8')
        id = int(os.path.split(image)[1].split(".")[1])
 
        faces.append(imageNp)
        ids.append(id)
    ids = np.array(ids)
 
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)
    clf.write("classifier.xml")
 
    return redirect(url_for('home'))

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< delete users >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

@app.route('/delete_user/<user_id>', methods=['POST'])
@login_required
def delete_user(user_id):
    try:
        mycursor.execute("SELECT img_id FROM img_dataset WHERE img_person = %s", (user_id,))
        images = mycursor.fetchall()
        
        for img in images:
            img_path = f"dataset/{user_id}.{img[0]}.jpg"
            if os.path.exists(img_path):
                os.remove(img_path)
        
        mycursor.execute("DELETE FROM img_dataset WHERE img_person = %s", (user_id,))
        mycursor.execute("DELETE FROM accs_hist WHERE accs_prsn = %s", (user_id,))
        mycursor.execute("DELETE FROM prs_mstr WHERE prs_nbr = %s", (user_id,))
        mydb.commit()
        
        return jsonify({'success': True})
    except Exception as e:
        mydb.rollback()
        return jsonify({'success': False, 'error': str(e)})

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Face Recognition >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def face_recognition():  
    def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf):
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features = classifier.detectMultiScale(gray_image, scaleFactor, minNeighbors)
        
        global justscanned
        global pause_cnt
        
        pause_cnt += 1
        
        coords = []
        
        for (x, y, w, h) in features:
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            id, pred = clf.predict(gray_image[y:y + h, x:x + w])
            confidence = int(100 * (1 - pred / 300))
            
            if confidence > 80 and not justscanned: 
                global cnt
                cnt += 1
                
                n = (100 / 30) * cnt
                w_filled = (cnt / 30) * w
                
                cv2.putText(img, str(int(n))+' %', (x + 20, y + h + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (153, 255, 255), 2, cv2.LINE_AA)
                
                cv2.rectangle(img, (x, y + h + 40), (x + w, y + h + 50), color, 2)
                cv2.rectangle(img, (x, y + h + 40), (x + int(w_filled), y + h + 50), (153, 255, 255), cv2.FILLED)
                
                if int(cnt) == 30:
                    confirmations = 0
                    for _ in range(5):
                        _, conf = clf.predict(gray_image[y:y + h, x:x + w])
                        if int(100 * (1 - conf / 300)) > 80:
                            confirmations += 1
                    
                    if confirmations >= 4:
                        mycursor.execute("select a.img_person, b.prs_name, b.prs_skill "
                                     "  from img_dataset a "
                                     "  left join prs_mstr b on a.img_person = b.prs_nbr "
                                     " where img_id = " + str(id))
                        row = mycursor.fetchone()
                        if row:
                            pnbr = row[0]
                            pname = row[1]
                            pskill = row[2]
                            
                            mycursor.execute("insert into accs_hist (accs_date, accs_prsn) values('"+str(date.today())+"', '" + pnbr + "')")
                            mydb.commit()
                            
                            cv2.putText(img, pname + ' | ' + pskill, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (153, 255, 255), 2, cv2.LINE_AA)
                    
                    cnt = 0
                    time.sleep(1)
                    justscanned = True
                    pause_cnt = 0
            
            else:
                if not justscanned:
                    cv2.putText(img, 'DESCONOCIDO', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    cv2.putText(img, ' ', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                
                if pause_cnt > 80:
                    justscanned = False
            
            coords = [x, y, w, h]
        return coords
 
    def recognize(img, clf, faceCascade):
        coords = draw_boundary(img, faceCascade, 1.1, 10, (255, 255, 0), "Face", clf)
        return img
 
    faceCascade = cv2.CascadeClassifier("C:/Users/Milton Montiel/Downloads/titulacion 1/proyecto_ia2/proyecto_ia2/resources/haarcascade_frontalface_default.xml")
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.read("classifier.xml")
 
    wCam, hCam = 400, 400
 
    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)
 
    while True:
        ret, img = cap.read()
        img = recognize(img, clf, faceCascade)
 
        frame = cv2.imencode('.jpg', img)[1].tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
 
        key = cv2.waitKey(1)
        if key == 27:
            break
 
@app.route('/')
def landing():  
    if 'logged_in' in session:
        return redirect(url_for('home'))
    return render_template('landing.html')
 
@app.route('/home')
def home():
    mycursor.execute("select prs_nbr, prs_name, prs_skill, prs_active, prs_added from prs_mstr")
    data = mycursor.fetchall()
 
    return render_template('index.html', data=data)

@app.route('/addprsn')
@login_required
def addprsn():
    mycursor.execute("select ifnull(max(prs_nbr) + 1, 1) from prs_mstr")
    row = mycursor.fetchone()
    nbr = row[0]
 
    return render_template('addprsn.html', newnbr=int(nbr))
 
@app.route('/addprsn_submit', methods=['POST'])
def addprsn_submit():
    prsnbr = request.form.get('txtnbr')
    prsname = request.form.get('txtname')
    prsskill = request.form.get('optskill')
 
    mycursor.execute("""INSERT INTO `prs_mstr` (`prs_nbr`, `prs_name`, `prs_skill`) VALUES
                    ('{}', '{}', '{}')""".format(prsnbr, prsname, prsskill))
    mydb.commit()
 
    return redirect(url_for('vfdataset_page', prs=prsnbr))
 
@app.route('/vfdataset_page/<prs>')
def vfdataset_page(prs):
    return render_template('gendataset.html', prs=prs)
 
@app.route('/vidfeed_dataset/<nbr>')
def vidfeed_dataset(nbr):
    return Response(generate_dataset(nbr), mimetype='multipart/x-mixed-replace; boundary=frame')
 
 
@app.route('/video_feed')
def video_feed():
    return Response(face_recognition(), mimetype='multipart/x-mixed-replace; boundary=frame')
 
@app.route('/fr_page')
def fr_page():
    """Video streaming home page."""
    mycursor.execute("select a.accs_id, a.accs_prsn, b.prs_name, b.prs_skill, a.accs_added "
                     "  from accs_hist a "
                     "  left join prs_mstr b on a.accs_prsn = b.prs_nbr "
                     " where a.accs_date = curdate() "
                     " order by 1 desc")
    data = mycursor.fetchall()
    
    return render_template('fr_page.html', data=data, page='fr_page')
 
 
@app.route('/countTodayScan')
def countTodayScan():
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        passwd="",
        database="flask_db"
    )
    mycursor = mydb.cursor()
 
    mycursor.execute("select count(*) "
                     "  from accs_hist "
                     " where accs_date = curdate() ")
    row = mycursor.fetchone()
    rowcount = row[0]
 
    return jsonify({'rowcount': rowcount})
 
 
@app.route('/loadData', methods = ['GET', 'POST'])
def loadData():
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        passwd="",
        database="flask_db"
    )
    mycursor = mydb.cursor()
 
    mycursor.execute("select a.accs_id, a.accs_prsn, b.prs_name, b.prs_skill, date_format(a.accs_added, '%H:%i:%s') "
                     "  from accs_hist a "
                     "  left join prs_mstr b on a.accs_prsn = b.prs_nbr "
                     " where a.accs_date = curdate() "
                     " order by 1 desc")
    data = mycursor.fetchall()
 
    return jsonify(response = data)

@app.route('/generate_report', methods=['POST'])
@login_required
def generate_report():
    try:
        mycursor.execute("""
            select a.accs_prsn, b.prs_name, b.prs_skill, date_format(a.accs_added, '%H:%i:%s')
            from accs_hist a
            left join prs_mstr b on a.accs_prsn = b.prs_nbr
            where a.accs_date = curdate()
            order by a.accs_added
        """)
        data = mycursor.fetchall()

        if not data:
            return jsonify({'success': False, 'message': 'No hay registros para generar el reporte.'})
        
        filename = f"registro_reconocimiento_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        doc = SimpleDocTemplate(f"static/reports/{filename}", pagesize=letter)
        elements = []
        
        table_data = [['ID Persona', 'Nombre', 'Cargo', 'Hora']]
        for row in data:
            table_data.append([str(x) for x in row])
            
        table = Table(table_data, colWidths=[1*inch, 2*inch, 2*inch, 1*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        
        elements.append(table)
        doc.build(elements)
        

        return jsonify({'success': True})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/clear_records', methods=['POST'])
@login_required
def clear_records():
    try:
        mycursor.execute("DELETE FROM accs_hist WHERE accs_date = curdate()")
        mydb.commit()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
    
if not os.path.exists('static/reports'):
    os.makedirs('static/reports')
 
if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=True)
