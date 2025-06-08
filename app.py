from PIL import Image
import io # Also needed for BytesIO


from flask import Flask, render_template, request, redirect, url_for, jsonify
from database import init_db, register_teacher, get_teacher, register_student, recognize_face_and_mark_attendance, get_attendance
from datetime import datetime


# ... (your existing imports) ...

@app.route('/webcam')
def webcam_page():
    """Renders the webcam capture page."""
    return render_template('webcam.html')

# ... (rest of your Flask app code) ...
app = Flask(__name__)
init_db()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register/teacher', methods=['GET', 'POST'])
def register_teacher_route():
    if request.method == 'POST':
        name = request.form['name']
        subject = request.form['subject']
        class_name = request.form.get("class")

        register_teacher(name, subject, class_name)
        return redirect(url_for('login_teacher_route'))
    return render_template('teacher_register.html')

@app.route('/login/teacher', methods=['GET', 'POST'])
def login_teacher_route():
    if request.method == 'POST':
        name = request.form['name']
        teacher = get_teacher(name)
        if teacher:
            records = get_attendance()
            return render_template('attendance.html', records=records)
        else:
            return "Teacher not found", 404
    return render_template('teacher_login.html')

@app.route('/register/student', methods=['GET', 'POST'])
def register_student_route():
    if request.method == 'POST':
        name = request.form['name']
        roll = request.form['roll']
        class_name = request.form.get("class")

        # Save student face encoding here (to be added)
        register_student(name, roll, class_name)
        return "Student registered!"
    return render_template('student_register.html')

@app.route('/login/student', methods=['GET', 'POST'])
def login_student_route():
    if request.method == 'POST':
        # image = request.files['image']
        # Match face, record attendance
        success, name = recognize_face_and_mark_attendance()
        if success:
            return jsonify({'message': f'Attendance marked for {name}'})
        return jsonify({'message': 'Face not recognized'}), 401
    return render_template('student_login.html')

if __name__ == '__main__':
    app.run(debug=True)
