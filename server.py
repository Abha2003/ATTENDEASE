import traceback
from flask import Flask, render_template, redirect, url_for, session, jsonify, request, flash
from database import init_db, register_teacher, validate_teacher_login, get_students_by_class
import threading
import webbrowser
import os
import sqlite3
from datetime import datetime, date, timedelta
import cv2 # Still needed for image processing (e.g., BGR to RGB if done locally before sending to service)
import numpy as np # Still needed for array operations
from PIL import Image
import io
import base64
import time
import shutil # Added for file operations (e.g., renaming)

# Import the new face recognition service
import face_recognition_service as fr_service

# Ensure directories exist (these are still relevant for student images and DB)
os.makedirs('face_data', exist_ok=True)
os.makedirs('database', exist_ok=True)
os.makedirs('static/student_images', exist_ok=True)

app = Flask(__name__)
app.secret_key = 'YOUR-SUPER-SECRET-AND-RANDOM-KEY' # IMPORTANT: Change this to a strong, random key in production!

# --- INCREASE MAX CONTENT LENGTH FOR FILE UPLOADS/WEBCAM DATA ---
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024 # 200 Megabytes

# The global model loading is now handled by face_recognition_service.py itself
# You can optionally call fr_service.init_face_recognition_models() here
# to ensure models are loaded when the Flask app starts, though imports
# usually handle initial loading within the service file.

# --- Simulated BLE Presence Dictionary ---
student_presence = {}
PRESENCE_TIMEOUT_SECONDS = 60

# Database initialization (now includes explicit column definitions and unique constraints)
def init_db():
    conn = sqlite3.connect('database/attendance.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS teachers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            class TEXT NOT NULL,
            subject TEXT NOT NULL,
            UNIQUE(class, subject)
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            class TEXT NOT NULL,
            roll_no TEXT NOT NULL,
            parent_email TEXT,
            parent_phone TEXT,
            address TEXT,
            dob TEXT,
            UNIQUE(roll_no, class)
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            status TEXT NOT NULL,
            FOREIGN KEY(student_id) REFERENCES students(id),
            UNIQUE(student_id, date)
        )
    ''')
    conn.commit()

    # Add new columns to students table if they don't exist (for backward compatibility)
    try:
        c.execute("ALTER TABLE students ADD COLUMN parent_email TEXT")
    except sqlite3.OperationalError: pass
    try:
        c.execute("ALTER TABLE students ADD COLUMN parent_phone TEXT")
    except sqlite3.OperationalError: pass
    try:
        c.execute("ALTER TABLE students ADD COLUMN address TEXT")
    except sqlite3.OperationalError: pass
    try:
        c.execute("ALTER TABLE students ADD COLUMN dob TEXT")
    except sqlite3.OperationalError: pass

    conn.close()

init_db()

# --- Flask Routes ---
@app.route('/')
def home():
    return render_template('home.html')

@app.route("/register/teacher", methods=["GET", "POST"])
def register_teacher_route():
    if request.method == "POST":
        name = request.form["name"].upper()
        class_name = request.form["class"].upper()
        subject = request.form["subject"].upper()

        conn = sqlite3.connect('database/attendance.db')
        c = conn.cursor()
        c.execute("SELECT * FROM teachers WHERE name = ? AND class = ? AND subject = ?", (name, class_name, subject))
        if c.fetchone() is None:
            c.execute("INSERT INTO teachers (name, class, subject) VALUES (?, ?, ?)", (name, class_name, subject))
            conn.commit()
            conn.close()
            flash("✅ TEACHER REGISTERED SUCCESSFULLY. PLEASE LOG IN.", "success")
            return redirect(url_for('login_teacher'))
        else:
            conn.close()
            flash("❌ THIS TEACHER IS ALREADY REGISTERED FOR THIS CLASS AND SUBJECT COMBINATION.", "error")
            return render_template("register_teacher.html")
    return render_template("register_teacher.html")

@app.route("/login/teacher", methods=["GET", "POST"])
def login_teacher():
    if request.method == "POST":
        class_name = request.form["class"].upper()
        subject = request.form["subject"].upper()
        if validate_teacher_login(class_name, subject):
            flash("LOGIN SUCCESSFUL!", "success")
            session['teacher_class'] = class_name
            session['teacher_subject'] = subject
            return redirect(url_for('teacher_dashboard'))
        else:
            flash("❌ NOT REGISTERED", "error")
            return render_template("login_teacher.html")
    return render_template("login_teacher.html")

@app.route('/dashboard/teacher')
def teacher_dashboard():
    if 'teacher_class' not in session:
        flash("PLEASE LOG IN TO ACCESS THE DASHBOARD.", "info")
        return redirect(url_for('login_teacher'))

    class_ = session['teacher_class'].upper()
    conn = sqlite3.connect('database/attendance.db')
    c = conn.cursor()
    
    teacher_name = "TEACHER"
    c.execute("SELECT name FROM teachers WHERE class=? AND subject=?", (class_, session.get('teacher_subject', '').upper()))
    result = c.fetchone()
    if result:
        teacher_name = result[0].upper()
    
    c.execute("SELECT id, name, roll_no FROM students WHERE class=?", (class_,))
    students_raw = c.fetchall()

    student_data = []
    current_date_str = datetime.now().strftime("%Y-%m-%d")

    present_today_count = 0
    absent_today_count = 0

    for student in students_raw:
        student_id, name, roll_no = student
        c.execute("SELECT COUNT(*) FROM attendance WHERE student_id=? AND status='Present'", (student_id,))
        present_days = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM attendance WHERE student_id=?", (student_id,))
        total_days = c.fetchone()[0]
        attendance_percentage = (present_days / total_days) * 100 if total_days > 0 else 0

        c.execute("SELECT status FROM attendance WHERE student_id=? AND date=?", (student_id, current_date_str))
        today_status_record = c.fetchone()
        today_status = "ABSENT"
        if today_status_record and today_status_record[0] == 'Present':
            today_status = "PRESENT"
            present_today_count += 1
        else:
            absent_today_count += 1

        student_data.append({
            'id': student_id,
            'name': name.upper(),
            'roll_no': roll_no.upper(),
            'attendance': f"{attendance_percentage:.2f}%",
            'today_status': today_status
        })
    conn.close()
    return render_template('dashboard_teacher.html', teacher_name=teacher_name, students=student_data, class_name=class_,
                           present_today_count=present_today_count, absent_today_count=absent_today_count)

@app.route('/register/student', methods=["GET", "POST"])
def register_student():
    if request.method == "POST":
        name = request.form["name"].upper()
        class_ = request.form["class_"].upper()
        roll_no = request.form["roll_no"].upper()
        parent_email = request.form.get("parent_email", "").upper()
        parent_phone = request.form.get("parent_phone", "").upper()
        address = request.form.get("address", "").upper()
        dob = request.form.get("dob", "").upper()

        print("\n--- DEBUG: REGISTER STUDENT POST REQUEST ---")
        print(f"FORM DATA RECEIVED: {request.form}")
        print(f"FILES RECEIVED: {request.files}")

        image_data = request.form.get("image_data")
        image_file = request.files.get("image_file")

        print(f"IMAGE_DATA (FROM HIDDEN INPUT): {image_data[:50] + '...' if image_data else 'NONE'}")
        print(f"IMAGE_FILE (FROM FILE UPLOAD): {image_file.filename if image_file else 'NONE'}")

        filename = f"{roll_no}.PNG"
        image_path = os.path.join("static", "student_images", filename)

        pil_image = None

        if image_data:
            print("--- REGISTER: PROCESSING IMAGE FROM WEBCAM DATA ---")
            try:
                if "," in image_data:
                    image_data = image_data.split(",")[1]
                image_bytes = base64.b64decode(image_data)
                
                pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                print(f"DEBUG REGISTER: PIL IMAGE FROM WEBCAM - MODE={pil_image.mode}, SIZE={pil_image.size}")
            except Exception as e:
                flash(f"❌ ERROR DECODING WEBCAM IMAGE DATA OR OPENING WITH PIL: {e}", "error")
                print(f"DEBUG REGISTER: WEBCAM IMAGE PROCESSING ERROR: {e}")
                return render_template("register_student.html")
        elif image_file:
            print("--- REGISTER: PROCESSING IMAGE FROM FILE UPLOAD ---")
            try:
                pil_image = Image.open(image_file.stream).convert('RGB')
                print(f"DEBUG REGISTER: PIL IMAGE FROM FILE - MODE={pil_image.mode}, SIZE={pil_image.size}")
            except Exception as e:
                flash(f"❌ ERROR OPENING UPLOADED IMAGE FILE WITH PIL: {e}", "error")
                print(f"DEBUG REGISTER: FILE UPLOAD IMAGE PROCESSING ERROR: {e}")
                return render_template("register_student.html")
        else:
            flash("❌ NO IMAGE DATA OR FILE PROVIDED. PLEASE UPLOAD A PHOTO OR USE THE WEBCAM.", "error")
            print("DEBUG REGISTER: NEITHER IMAGE_DATA NOR IMAGE_FILE WAS PROVIDED.")
            return render_template("register_student.html")

        if pil_image: # Check if PIL image was successfully created
            success, embedding_data, message = fr_service.verify_student_registration_image(pil_image, roll_no, class_)
            
            if not success:
                flash(f"❌ {message}", "error")
                return render_template("register_student.html")
            
            # Save the image and embedding only if verification was successful
            pil_image.save(image_path, format='PNG')
            print(f"DEBUG REGISTER: ORIGINAL PIL IMAGE SAVED TO: {image_path}")
            np.save(os.path.join(fr_service.FACE_DATA_DIR, f'{roll_no}.npy'), embedding_data)
            print(f"DEBUG REGISTER: FACENET EMBEDDING SAVED FOR {roll_no}.")
            
        else:
            flash("❌ IMAGE PROCESSING FAILED. PLEASE PROVIDE A VALID PHOTO.", "error")
            return render_template("register_student.html")

        conn = sqlite3.connect('database/attendance.db')
        c = conn.cursor()
        try:
            c.execute("INSERT INTO students (name, class, roll_no, parent_email, parent_phone, address, dob) VALUES (?, ?, ?, ?, ?, ?, ?)",
                      (name, class_, roll_no, parent_email, parent_phone, address, dob))
            conn.commit()
            flash("✅ STUDENT REGISTERED SUCCESSFULLY!", "success")
            return redirect(url_for('login_student_page'))
        except sqlite3.IntegrityError:
            flash("❌ STUDENT WITH THIS ROLL NUMBER AND CLASS COMBINATION ALREADY EXISTS.", "error")
            print(f"DEBUG REGISTER: INTEGRITYERROR FOR {roll_no} IN {class_}. Cleaning up face data.")
            # Clean up files if DB insertion fails due to integrity error
            if os.path.exists(image_path): os.remove(image_path)
            if os.path.exists(os.path.join(fr_service.FACE_DATA_DIR, f'{roll_no}.npy')): os.remove(os.path.join(fr_service.FACE_DATA_DIR, f'{roll_no}.npy')) 
            return render_template("register_student.html")
        finally:
            conn.close()
    return render_template("register_student.html")

@app.route('/login/student', methods=['GET'])
def login_student_page():
    return render_template('login_student_webcam.html')

@app.route('/login/student/verify', methods=['POST'])
def login_student_verify():
    try:
        data = request.get_json()
        if not data or 'images' not in data:
            print("ERROR: NO IMAGE DATA PROVIDED IN JSON.")
            return jsonify({"success": False, "message": "NO IMAGE DATA PROVIDED."}), 400
        
        image_data_list = data['images']
        
        if not image_data_list or len(image_data_list) == 0:
            print("ERROR: EMPTY IMAGE DATA LIST RECEIVED.")
            return jsonify({"success": False, "message": "NO IMAGE CAPTURED FOR VERIFICATION."}), 400

        image_data_base64 = image_data_list[0]
        
        # Call the recognition service
        success, recognized_info, message = fr_service.recognize_face_from_base64_image(image_data_base64)
        
        conn = sqlite3.connect('database/attendance.db')
        c = conn.cursor()

        if success and recognized_info:
            predicted_roll_no = recognized_info['roll_no']
            
            # Fetch full student details from DB using the recognized roll number
            c.execute("SELECT id, name, class, roll_no FROM students WHERE roll_no = ?", (predicted_roll_no,))
            matched_student_db = c.fetchone()
            
            if matched_student_db:
                student_id, name_from_db_raw, class_from_db_raw, roll_no_from_db_raw = matched_student_db
                matched_student = {
                    'id': student_id,
                    'name': name_from_db_raw.upper(),
                    'class': class_from_db_raw.upper(),
                    'roll_no': roll_no_from_db_raw.upper()
                }

                date_str = datetime.now().strftime("%Y-%m-%d")
                c.execute("SELECT * FROM attendance WHERE student_id=? AND date=?", (matched_student['id'], date_str))
                if not c.fetchone():
                    c.execute("INSERT INTO attendance (student_id, date, status) VALUES (?, ?, ?)", (matched_student['id'], date_str, 'Present'))
                    conn.commit()
                    conn.close()
                    return jsonify({"success": True, "message": f"ATTENDANCE MARKED FOR {matched_student['name']}!", "redirect_url": url_for('student_ble_advertiser', student_id=matched_student['id'])})
                else:
                    conn.close()
                    return jsonify({"success": True, "message": f"ATTENDANCE ALREADY MARKED FOR {matched_student['name']} TODAY.", "redirect_url": url_for('student_ble_advertiser', student_id=matched_student['id'])})
            else:
                conn.close()
                return jsonify({"success": False, "message": f"RECOGNIZED ROLL NO '{predicted_roll_no}' NOT FOUND IN DATABASE. {message}"}), 401
        else:
            conn.close()
            return jsonify({"success": False, "message": message}), 401
            
    except Exception as general_e:
        print(f"CRITICAL ERROR: GENERAL ERROR IN LOGIN_STUDENT_VERIFY: {general_e}")
        traceback.print_exc()
        return jsonify({"success": False, "message": "AN UNEXPECTED SERVER ERROR OCCURRED."}), 500

@app.route('/student/profile/<int:student_id>', methods=['GET'])
def student_profile(student_id):
    if 'teacher_class' not in session:
        flash("PLEASE LOG IN TO ACCESS STUDENT PROFILES.", "info")
        return redirect(url_for('login_teacher'))

    conn = sqlite3.connect('database/attendance.db')
    c = conn.cursor()
    c.execute("SELECT id, name, class, roll_no, parent_email, parent_phone, address, dob FROM students WHERE id = ?", (student_id,))
    student = c.fetchone()
    conn.close()

    if student:
        student_data = {
            'id': student[0],
            'name': student[1].upper(),
            'class': student[2].upper(),
            'roll_no': student[3].upper(),
            'parent_email': student[4].upper() if student[4] else '',
            'parent_phone': student[5].upper() if student[5] else '',
            'address': student[6].upper() if student[6] else '',
            'dob': student[7].upper() if student[7] else ''
        }
        student_image_path = url_for('static', filename=f'student_images/{student_data["roll_no"]}.PNG')
        return render_template('student_profile.html', student=student_data, student_image_path=student_image_path, edit_mode=False)
    else:
        flash("❌ STUDENT NOT FOUND.", "error")
        return redirect(url_for('teacher_dashboard'))

@app.route('/student/edit/<int:student_id>', methods=['GET', 'POST'])
def edit_student(student_id):
    if 'teacher_class' not in session:
        flash("PLEASE LOG IN TO EDIT STUDENT PROFILES.", "info")
        return redirect(url_for('login_teacher'))

    conn = sqlite3.connect('database/attendance.db')
    c = conn.cursor()

    if request.method == 'POST':
        name = request.form["name"].upper()
        class_ = request.form["class_"].upper()
        roll_no = request.form["roll_no"].upper()
        parent_email = request.form.get("parent_email", "").upper()
        parent_phone = request.form.get("parent_phone", "").upper()
        address = request.form.get("address", "").upper()
        dob = request.form.get("dob", "").upper()

        c.execute("SELECT id, roll_no FROM students WHERE id = ?", (student_id,))
        current_student_info = c.fetchone()

        if current_student_info:
            old_roll_no = current_student_info[1].upper()
            if old_roll_no != roll_no:
                c.execute("SELECT id FROM students WHERE roll_no = ? AND class = ? AND id != ?", (roll_no, class_, student_id))
                if c.fetchone():
                    conn.close()
                    flash("❌ NEW ROLL NUMBER AND CLASS COMBINATION ALREADY EXISTS FOR ANOTHER STUDENT. UPDATE FAILED.", "error")
                    return redirect(url_for('student_profile', student_id=student_id))

        try:
            c.execute("UPDATE students SET name = ?, class = ?, roll_no = ?, parent_email = ?, parent_phone = ?, address = ?, dob = ? WHERE id = ?",
                      (name, class_, roll_no, parent_email, parent_phone, address, dob, student_id))
            conn.commit()

            if current_student_info and current_student_info[1].upper() != roll_no:
                old_roll_no_for_files = current_student_info[1].upper()
                new_roll_no_for_files = roll_no

                old_npy_path = os.path.join(fr_service.FACE_DATA_DIR, f'{old_roll_no_for_files}.npy')
                new_npy_path = os.path.join(fr_service.FACE_DATA_DIR, f'{new_roll_no_for_files}.npy')

                old_img_path = os.path.join("static", "student_images", f"{old_roll_no_for_files}.PNG")
                new_img_path = os.path.join("static", "student_images", f"{new_roll_no_for_files}.PNG")

                if os.path.exists(old_npy_path):
                    if not os.path.exists(new_npy_path):
                        try:
                            os.rename(old_npy_path, new_npy_path)
                            print(f"RENAMED .NPY: {old_npy_path} -> {new_npy_path}")
                        except OSError as e:
                            print(f"ERROR RENAMING .NPY FILE DIRECTLY ({old_npy_path} TO {new_npy_path}): {e}. ATTEMPTING COPY/DELETE FOR NPY.")
                            try: shutil.copy2(old_npy_path, new_npy_path); os.remove(old_npy_path); print(f"COPIED AND DELETED .NPY: {old_npy_path} -> {new_npy_path}")
                            except Exception as copy_e: print(f"ERROR: FAILED TO COPY/DELETE .NPY: {copy_e}. THE FACE ENCODING MIGHT BE BROKEN."); flash(f"WARNING: FAILED TO UPDATE FACE DATA FILE FOR {name}. FACE RECOGNITION MIGHT NOT WORK CORRECTLY. ERROR: {copy_e}", "warning")
                    else: print(f"DEBUG: NEW NPY PATH '{new_npy_path}' ALREADY EXISTS. SKIPPING RENAME FOR {old_npy_path}.")
                else: print(f"DEBUG: OLD NPY PATH '{old_npy_path}' DOES NOT EXIST. NO RENAME FOR NPY NEEDED.")

                if os.path.exists(old_img_path):
                    if not os.path.exists(new_img_path):
                        try:
                            os.rename(old_img_path, new_img_path)
                            print(f"RENAMED IMAGE: {old_img_path} -> {new_img_path}")
                        except OSError as e:
                            print(f"ERROR RENAMING IMAGE FILE DIRECTLY ({old_img_path} TO {new_img_path}): {e}. ATTEMPTING COPY/DELETE FOR IMAGE.")
                            try: shutil.copy2(old_img_path, new_img_path); os.remove(old_img_path); print(f"COPIED AND DELETED IMAGE: {old_img_path} -> {new_img_path}")
                            except Exception as copy_e: print(f"ERROR: FAILED TO COPY/DELETE IMAGE: {copy_e}. STUDENT IMAGE MIGHT NOT DISPLAY."); flash(f"WARNING: FAILED TO UPDATE STUDENT IMAGE FILE FOR {name}. STUDENT IMAGE MIGHT NOT DISPLAY CORRECTLY. ERROR: {copy_e}", "warning")
                    else: print(f"DEBUG: NEW IMAGE PATH '{new_img_path}' ALREADY EXISTS. SKIPPING RENAME FOR {old_img_path}.")
                else: print(f"DEBUG: OLD IMAGE PATH '{old_img_path}' DOES NOT EXIST. NO RENAME FOR IMAGE NEEDED.")

            flash("✅ STUDENT DETAILS UPDATED SUCCESSFULLY!", "success")
            conn.close()
            return redirect(url_for('student_profile', student_id=student_id))
        except Exception as e:
            conn.close()
            traceback.print_exc()
            flash(f"❌ ERROR UPDATING STUDENT DETAILS: {e}. PLEASE CHECK SERVER LOGS FOR MORE INFO.", "error")
            return redirect(url_for('student_profile', student_id=student_id))

    c.execute("SELECT id, name, class, roll_no, parent_email, parent_phone, address, dob FROM students WHERE id = ?", (student_id,))
    student = c.fetchone()
    conn.close()

    if student:
        student_data = {
            'id': student[0], 'name': student[1].upper(), 'class': student[2].upper(), 'roll_no': student[3].upper(),
            'parent_email': student[4].upper() if student[4] else '', 'parent_phone': student[5].upper() if student[5] else '',
            'address': student[6].upper() if student[6] else '', 'dob': student[7].upper() if student[7] else ''
        }
        student_image_path = url_for('static', filename=f'student_images/{student_data["roll_no"]}.PNG')
        return render_template('student_profile.html', student=student_data, student_image_path=student_image_path, edit_mode=True)
    else:
        flash("❌ STUDENT NOT FOUND FOR EDITING.", "error")
        return redirect(url_for('teacher_dashboard'))

@app.route('/student/delete/<int:student_id>', methods=['POST'])
def delete_student(student_id):
    if 'teacher_class' not in session:
        flash("PLEASE LOG IN TO DELETE STUDENT PROFILES.", "info")
        return redirect(url_for('login_teacher'))

    conn = sqlite3.connect('database/attendance.db')
    c = conn.cursor()

    try:
        c.execute("SELECT roll_no FROM students WHERE id = ?", (student_id,))
        student_record = c.fetchone()

        if student_record:
            roll_no_to_delete = student_record[0].upper()

            c.execute("DELETE FROM attendance WHERE student_id = ?", (student_id,))
            c.execute("DELETE FROM students WHERE id = ?", (student_id,))
            conn.commit()

            npy_path = os.path.join(fr_service.FACE_DATA_DIR, f'{roll_no_to_delete}.npy')
            if os.path.exists(npy_path): os.remove(npy_path); print(f"DELETED .NPY FILE: {npy_path}")
            img_path = os.path.join("static", "student_images", f"{roll_no_to_delete}.PNG")
            if os.path.exists(img_path): os.remove(img_path); print(f"DELETED IMAGE FILE: {img_path}")

            flash("✅ STUDENT AND ALL ASSOCIATED DATA DELETED SUCCESSFULLY!", "success")
        else:
            flash("❌ STUDENT NOT FOUND FOR DELETION.", "error")
    except Exception as e:
        flash(f"❌ ERROR DELETING STUDENT: {e}", "error")
        traceback.print_exc()
    finally:
        conn.close()
    return redirect(url_for('teacher_dashboard'))

@app.route('/teacher/profile', methods=['GET', 'POST'])
def teacher_profile():
    if 'teacher_class' not in session:
        flash("PLEASE LOG IN TO VIEW YOUR PROFILE.", "info")
        return redirect(url_for('login_teacher'))

    class_name = session['teacher_class'].upper()
    subject = session.get('teacher_subject', '').upper()

    conn = sqlite3.connect('database/attendance.db')
    c = conn.cursor()

    if request.method == 'POST':
        new_teacher_name = request.form["name"].upper()

        try:
            c.execute("UPDATE teachers SET name = ? WHERE class = ? AND subject = ?", (new_teacher_name, class_name, subject))
            conn.commit()
            flash("✅ TEACHER NAME UPDATED SUCCESSFULLY!", "success")
        except Exception as e:
            flash(f"❌ ERROR UPDATING TEACHER NAME: {e}", "error")
            traceback.print_exc()
        finally:
            conn.close()
        return redirect(url_for('teacher_profile'))

    c.execute("SELECT name, class, subject FROM teachers WHERE class = ? AND subject = ?", (class_name, subject))
    teacher_info = c.fetchone()
    conn.close()

    if teacher_info:
        teacher_data = {'name': teacher_info[0].upper(), 'class': teacher_info[1].upper(), 'subject': teacher_info[2].upper()}
        return render_template('teacher_profile.html', teacher=teacher_data)
    else:
        flash("❌ TEACHER PROFILE NOT FOUND.", "error")
        return redirect(url_for('teacher_dashboard'))

@app.route('/attendance/report', methods=['GET'])
def attendance_report():
    if 'teacher_class' not in session:
        flash("PLEASE LOG IN TO VIEW ATTENDANCE REPORTS.", "info")
        return redirect(url_for('login_teacher'))

    class_ = session['teacher_class'].upper()
    start_date_str = request.args.get('start_date')
    end_date_str = request.args.get('end_date')

    if not start_date_str or not end_date_str:
        end_date = date.today(); start_date = end_date - timedelta(days=6)
        start_date_str = start_date.strftime("%Y-%m-%d"); end_date_str = end_date.strftime("%Y-%m-%d")
    else:
        try:
            start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
            end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
            if start_date > end_date: start_date, end_date = end_date, start_date
        except ValueError:
            flash("❌ INVALID DATE FORMAT. PLEASE USEYYYY-MM-DD.", "error")
            end_date = date.today(); start_date = end_date - timedelta(days=6)
            start_date_str = start_date.strftime("%Y-%m-%d"); end_date_str = end_date.strftime("%Y-%m-%d")

    conn = sqlite3.connect('database/attendance.db')
    c = conn.cursor()

    c.execute("SELECT id, name, roll_no FROM students WHERE class = ?", (class_,))
    students_in_class = c.fetchall()

    report_data = []

    for student_id, student_name, student_roll_no in students_in_class:
        student_attendance_records = []; present_count_range = 0; absent_count_range = 0
        current_date = start_date

        while current_date <= end_date:
            current_date_str = current_date.strftime("%Y-%m-%d")
            c.execute("SELECT status FROM attendance WHERE student_id = ? AND date = ?", (student_id, current_date_str))
            record = c.fetchone()

            status = record[0].upper() if record else "ABSENT"
            if status == "PRESENT": present_count_range += 1
            else: absent_count_range += 1

            student_attendance_records.append({'date': current_date_str, 'status': status})
            current_date += timedelta(days=1)

        total_days_in_range = (end_date - start_date).days + 1
        attendance_percentage_range = (present_count_range / total_days_in_range) * 100 if total_days_in_range > 0 else 0

        report_data.append({
            'name': student_name.upper(), 'roll_no': student_roll_no.upper(),
            'attendance_records': student_attendance_records,
            'present_count_range': present_count_range, 'absent_count_range': absent_count_range,
            'attendance_percentage_range': f"{attendance_percentage_range:.2f}%"
        })
    conn.close()

    return render_template('attendance_report.html',
                           class_name=class_, start_date=start_date_str, end_date=end_date_str, report_data=report_data)

@app.route('/student_ble_advertiser/<int:student_id>')
def student_ble_advertiser(student_id):
    conn = sqlite3.connect('database/attendance.db')
    c = conn.cursor()
    c.execute("SELECT name, roll_no FROM students WHERE id = ?", (student_id,))
    student = c.fetchone()
    conn.close()
    if student:
        return render_template('student_ble_advertiser.html', student_id=student_id, student_name=student[0].upper(), student_roll_no=student[1].upper())
    else:
        flash("STUDENT NOT FOUND FOR BLE ADVERTISING.", "error")
        return redirect(url_for('home'))

@app.route('/api/student_heartbeat', methods=['POST'])
def api_student_heartbeat():
    data = request.get_json()
    student_id = data.get('student_id')
    current_time = time.time()

    if student_id:
        student_presence[student_id] = current_time
        print(f"DEBUG: STUDENT {student_id} HEARTBEAT RECEIVED AT {datetime.fromtimestamp(current_time).strftime('%H:%M:%S')}")
        return jsonify({"status": "SUCCESS", "message": "HEARTBEAT RECORDED."})
    return jsonify({"status": "ERROR", "message": "INVALID STUDENT ID."}), 400

@app.route('/teacher_ble_scanner')
def teacher_ble_scanner():
    if 'teacher_class' not in session:
        flash("PLEASE LOG IN TO SCAN FOR STUDENT PRESENCE.", "info")
        return redirect(url_for('login_teacher'))
    return render_template('teacher_ble_scanner.html')

@app.route('/api/get_present_students')
def api_get_present_students():
    current_time = time.time()
    present_students_data = []

    conn = sqlite3.connect('database/attendance.db')
    c = conn.cursor()

    active_student_ids_set = set()
    for student_id, last_seen_time in list(student_presence.items()):
        if (current_time - last_seen_time) < PRESENCE_TIMEOUT_SECONDS:
            active_student_ids_set.add(student_id)

    if active_student_ids_set:
        teacher_class = session['teacher_class'].upper()
        c.execute("SELECT id, name, roll_no FROM students WHERE class = ?", (teacher_class,))
        students_in_class = c.fetchall()

        for student_id_db, name_db, roll_no_db in students_in_class:
            if student_id_db in active_student_ids_set:
                present_students_data.append({
                    'id': student_id_db,
                    'name': name_db.upper(),
                    'roll_no': roll_no_db.upper(),
                    'last_seen': f"{int(current_time - student_presence[student_id_db])} SECONDS AGO"
                })
    conn.close()
    return jsonify(present_students_data)

@app.route('/webcam')
def webcam_page():
    return render_template('webcam.html')

@app.route('/verify_photo', methods=['GET', 'POST'])
def verify_photo():
    if 'teacher_class' not in session:
        flash("PLEASE LOG IN TO ACCESS THIS FEATURE.", "info")
        return redirect(url_for('login_teacher'))

    result_message = None
    recognized_name = None
    original_image_url = None
    
    if request.method == 'POST':
        image_file = request.files.get('image_file')
        
        if image_file and image_file.filename != '':
            # Save the uploaded image temporarily
            temp_dir = 'temp_uploads'
            os.makedirs(temp_dir, exist_ok=True)
            temp_image_path = os.path.join(temp_dir, image_file.filename)
            image_file.save(temp_image_path)
            original_image_url = url_for('static', filename=f'../{temp_image_path}') # For displaying

            print(f"DEBUG: Processing uploaded image: {temp_image_path}")

            # Step 1: Single Recognition
            success_single, info_single, message_single = fr_service.recognize_face_from_image_path(temp_image_path)
            
            if success_single and info_single:
                recognized_name = info_single['roll_no'] # Using roll_no as name for display
                flash(f"Initial Recognition: Identified as {recognized_name}. Now verifying consistency...", "info")
                
                # Step 2: Consistency Verification
                verified, final_roll_no, verification_message = fr_service.verify_face_consistency_from_image_path(
                    temp_image_path,
                    required_matches=3,  # Example: Must recognize the same person at least 3 times
                    total_attempts=5     # Example: Over 5 attempts
                )
                
                if verified:
                    result_message = f"✅ VERIFICATION SUCCESS: Consistently identified as {final_roll_no}. {verification_message}"
                    flash(result_message, "success")
                    # Optionally, you could mark attendance here if this is an entry point
                else:
                    result_message = f"❌ VERIFICATION FAILED: {verification_message}. Initial match was {recognized_name}."
                    flash(result_message, "error")
            else:
                result_message = f"❌ INITIAL RECOGNITION FAILED: {message_single}"
                flash(result_message, "error")
            
            # Clean up temporary image
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
                print(f"DEBUG: Cleaned up temporary image: {temp_image_path}")
        else:
            flash("❌ NO IMAGE FILE SELECTED.", "error")

    return render_template('verify_photo.html', result_message=result_message, recognized_name=recognized_name, original_image_url=original_image_url)


def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000/")

if __name__ == '__main__':
    threading.Timer(1.0, open_browser).start()
    app.run(debug=True, use_reloader=False)

